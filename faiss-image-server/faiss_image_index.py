# -*- coding: utf-8 -*-
import os
import time
import logging
import glob
import random
import shutil
from dateutil import tz

import re
import faiss
import gevent
from gevent.pool import Pool
from gevent.threadpool import ThreadPool
import numpy as np
from tensorflow.python.lib.io import file_io
from sklearn.metrics.pairwise import cosine_similarity

import boto3
import botocore

import faissimageindex_pb2 as pb2
import faissimageindex_pb2_grpc as pb2_grpc

from embeddings import ImageEmbeddingService
from remote_embeddings import ImageEmbeddingService as RemoteImageEmbeddingService
from faiss_index import FaissIndex, FaissShrinkedIndex, FaissTrainIndex

# Disable debug logs of the boto lib
logging.getLogger('botocore').setLevel(logging.WARN)
logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def path_to_embedding(filepath):
    return np.fromstring(file_io.read_file_to_string(filepath, True),
            dtype=np.float32)

def url_to_s3_info(url):
    m = re.match('s3://([^/]+)/(.+)', url)
    bucket_name = m.group(1)
    key = m.group(2)
    return (bucket_name, key)


class FaissImageIndex(pb2_grpc.ImageIndexServicer):
    def _client(self):
        return self._s3_client

    def __init__(self, args):
        config = botocore.client.Config(connect_timeout=2, read_timeout=5)
        session = boto3.session.Session()
        self._s3_client = session.client('s3', config=config)

        if args.save_filepath.startswith('s3://'):
            bucket_name, key = url_to_s3_info(args.save_filepath)
            self.remote_save_info = (bucket_name, key)
            self.save_filepath = '/tmp/server.index'

            client = self._client()
            res = client.head_object(Bucket=bucket_name, Key=key)
            logging.info('save_file ContentLength: %d, LastModified: %s',
                    res['ContentLength'],
                    res['LastModified'].astimezone(tz.gettz('Asia/Seoul')).isoformat())

            client.download_file(bucket_name, key, self.save_filepath)
        else:
            self.remote_save_info = None
            self.save_filepath = args.save_filepath

        if args.remote_embedding_path:
            self.remote_embedding_info = url_to_s3_info(args.remote_embedding_path)
        else:
            self.remote_embedding_info = None

        self.max_train_count = args.train_count
        self._max_nlist = args.max_nlist
        t0 = time.time()
        if args.remote_embedding_host:
            self.embedding_service = RemoteImageEmbeddingService(args.remote_embedding_host,
                    load_balancing=args.lb, timeout=args.remote_embedding_timeout)
            logging.info("remote embedding service loaded, %s" % args.remote_embedding_host)
        else:
            self.embedding_service = ImageEmbeddingService(args.model)
            logging.info("embedding service loaded %.2f s" % (time.time() - t0))

        if file_io.file_exists(self.save_filepath) and not args.train_only:
            self.faiss_index = FaissTrainIndex()
            logging.info(self.faiss_index.__class__)
            t0 = time.time()
            self.faiss_index.restore(self.save_filepath)
            self.faiss_index.set_nprobe(args.nprobe)
            logging.info("%d items restored %.2f s", self.faiss_index.ntotal(), time.time() - t0)
        else:
            self.faiss_index = self._new_trained_index()

        t0 = time.time()
        if args.kmeans_filepath and file_io.file_exists(args.kmeans_filepath):
            self._kmeans_index = faiss.read_index(args.kmeans_filepath)
            logging.info("kmeans_index loaded %.2f s", time.time() - t0)
        elif args.kmeans_filepath and args.ncentroids > 0:
            logging.info("kmeans_index training.. ncentroids: %d", args.ncentroids)
            self._kmeans_index = self._train_kmeans(args.ncentroids)
            faiss.write_index(self._kmeans_index, args.kmeans_filepath)
            logging.info("kmeans_index loaded %.2f s", time.time() - t0)

    def Migrate(self, request, context):
        logging.info('Migrating...')
        t0 = time.time()
        self._sync()
        logging.info("Migrated %.2f s", time.time() - t0)
        return pb2.SimpleReponse(message='Migrated.')

    def _new_index(self, nlist=100):
        d = self.embedding_service.dim()
        faiss_index = FaissShrinkedIndex(d, nlist=nlist)
        logging.info(faiss_index.__class__)
        logging.info("nlist: %d", nlist)
        return faiss_index

    def Train(self, request, context):
        pre_index = self.faiss_index
        self.faiss_index = self._new_trained_index()
        pre_index.reset()
        return pb2.SimpleReponse(message='Trained')

    def Reset(self, request, context):
        d = self.embedding_service.dim()
        embedding = np.zeros([1, d], dtype=np.float32)
        count = 0
        while True:
            D, I = self.faiss_index.search(embedding, 100000)
            if I[0][0] == -1:
                break
            self.faiss_index.remove_ids(I[0])
            count += len(I[0])
            logging.info('removed %d ids', count)

        all_filepaths = glob.glob('embeddings/*/*.emb')
        files_count = len(all_filepaths)
        logging.info('emb files count: %d', files_count)
        p = Pool(12)
        p.map(file_io.delete_file, all_filepaths)
        return pb2.SimpleReponse(message=('Reset and deleted %d emb files' % files_count))

    # args: paths (list)
    def _path_to_xb(self, paths):
        d = self.embedding_service.dim()
        xb = np.ndarray(shape=(len(paths), d), dtype=np.float32)
        p = Pool(12)
        for i, emb in enumerate(p.imap(path_to_embedding, paths)):
            xb[i] = emb
        return xb

    def export_ids(self):
        logging.info("Id loading...")
        t0 = time.time()
        ids = self._get_ids()
        total_count = len(ids)
        logging.info("%d ids %.3f s", total_count, time.time() - t0)

        logging.info("Writing...")
        with open('models/added_ids.txt', 'w') as f:
            for id in ids:
                f.write("%d\n" % id)
        logging.info("Exported.")

    def _get_ids(self):
        def path_to_id(filepath):
            pos = filepath.rindex('/') + 1
            return int(filepath[pos:-4])

        filepaths = glob.glob('embeddings/*/*.emb')
        return [path_to_id(x) for x in filepaths]

    def _read_added_ids(self, filepath):
        if not file_io.file_exists(filepath):
            return []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return [int(line.rstrip()) for line in lines]

    def _sync(self):
        added_ids_filepath = 'models/added_ids.txt'
        added_ids = set(self._read_added_ids(added_ids_filepath))
        if not added_ids:
            return
        target_ids = set(self._get_ids())
        logging.info('added_ids count: %d', len(added_ids))
        logging.info('target_ids count: %d', len(target_ids))

        remove_ids = list(added_ids - target_ids)
        add_ids = list(target_ids - added_ids)
        logging.info('remove_ids count: %d', len(remove_ids))
        logging.info('add_ids count: %d', len(add_ids))

        if remove_ids:
            for ids in chunks(remove_ids, 20000):
                ids = np.array(ids, dtype=np.int64)
                self.faiss_index.remove_ids(ids)
            logging.info("removed")

        if add_ids:
            for ids in chunks(add_ids, 20000):
                t0 = time.time()
                filepaths = [self._get_filepath(id) for id in ids]
                xb = self._path_to_xb(filepaths)
                ids = np.array(ids, dtype=np.int64)
                self.faiss_index.add(xb, ids)
                logging.info("%d embeddings added %.3f s", xb.shape[0], time.time() - t0)

        file_io.delete_file(added_ids_filepath)
        logging.info("Synced. ntotal: %d", self.faiss_index.ntotal())

    def _train_kmeans(self, ncentroids):
        def path_to_id(filepath):
            pos = filepath.rindex('/') + 1
            return int(filepath[pos:-4])

        logging.info("File loading...")
        t0 = time.time()
        all_filepaths = glob.glob('embeddings/*/*.emb')
        total_count = len(all_filepaths)
        logging.info("%d files %.3f s", total_count, time.time() - t0)

        train_count = min(total_count, self.max_train_count)
        if train_count <= ncentroids:
            return

        logging.info("shuffling...")
        random.shuffle(all_filepaths)

        logging.info("embedings loading...")
        filepaths = all_filepaths[:train_count]
        t0 = time.time()
        xb = self._path_to_xb(filepaths)
        ids = np.array(list(map(path_to_id, filepaths)), dtype=np.int64)
        logging.info("%d embeddings loaded %.3f s", xb.shape[0], time.time() - t0)

        niter = 20
        d = xb.shape[1]
        verbose = True
        kmeans = faiss.Kmeans(d, ncentroids, niter, verbose)
        logging.info("training...")
        t0 = time.time()
        kmeans.train(xb)
        logging.info("trained %.2f s", time.time() - t0)

        D, I = self.faiss_index.search(kmeans.centroids, 1)
        print("centroid ids")
        print(I)
        return kmeans.index

    def _new_trained_index(self):
        def path_to_id(filepath):
            pos = filepath.rindex('/') + 1
            return int(filepath[pos:-4])

        logging.info("File loading...")
        t0 = time.time()
        all_filepaths = glob.glob('embeddings/*/*.emb')
        total_count = len(all_filepaths)
        logging.info("%d files %.3f s", total_count, time.time() - t0)

        train_count = min(total_count, self.max_train_count)
        if train_count <= 0:
            return self._new_index()

        random.shuffle(all_filepaths)

        filepaths = all_filepaths[:train_count]
        t0 = time.time()
        xb = self._path_to_xb(filepaths)
        ids = np.array(list(map(path_to_id, filepaths)), dtype=np.int64)
        logging.info("%d embeddings loaded %.3f s", xb.shape[0], time.time() - t0)

        if train_count < 10000:
            d = self.embedding_service.dim()
            faiss_index = FaissIndex(d)
            faiss_index.add(xb, ids)
            return faiss_index

        nlist = min(self._max_nlist, int(train_count / 39))
        faiss_index = self._new_index(nlist=nlist)

        logging.info("Training...")
        t0 = time.time()
        faiss_index.train(xb)
        logging.info("trained %.3f s", time.time() - t0)

        step = 100000
        for i in range(0, train_count, step):
            t0 = time.time()
            faiss_index.add(xb[i:i+step], ids[i:i+step])
            logging.info("added %.3f s", time.time() - t0)

        if total_count > train_count:
            for filepaths in chunks(all_filepaths[train_count:], 20000):
                t0 = time.time()
                xb = self._path_to_xb(filepaths)
                ids = np.array(list(map(path_to_id, filepaths)), dtype=np.int64)
                faiss_index.add(xb, ids)
                logging.info("%d embeddings added %.3f s", xb.shape[0], time.time() - t0)
            logging.info("Total %d embeddings added", faiss_index.ntotal())
        return faiss_index

    def Save(self, request, context):
        self.save()
        return pb2.SimpleReponse(message='Saved')

    def save(self):
        t0 = time.time()
        self.faiss_index.save(self.save_filepath)
        logging.info("index saved to %s, %.3f s", self.save_filepath, time.time() - t0)
        if self.remote_save_info:
            bucket_name, key = self.remote_save_info
            self._client().upload_file(self.save_filepath, bucket_name, key)
            logging.info('index file uploaded to s3://%s/%s' % (bucket_name, key))

    def Add(self, request, context):
        logging.debug('add - id: %d', request.id)
        if self._more_recent_emb_file_exists(request):
            return pb2.SimpleReponse(message='Already added, %s!' % request.id)

        embedding = self.fetch_embedding(request)
        if embedding is None:
            return pb2.SimpleReponse(message='No embedding, id: %d, url: %s' % (request.id, request.url))

        embedding = np.expand_dims(embedding, 0)
        ids = np.array([request.id], dtype=np.int64)
        self.faiss_index.replace(embedding, ids)

        return pb2.SimpleReponse(message='Added, %s!' % request.id)

    def Import(self, request, context):
        def get_mtime(filepath):
            if file_io.file_exists(filepath):
                return file_io.stat(filepath).mtime_nsec
            return None

        def is_new_emb(id, filepath):
            origin_mtime = get_mtime(self._get_filepath(id))
            if origin_mtime is None:
                return True
            new_mtime = get_mtime(filepath)
            return origin_mtime < new_mtime

        logging.info("Importing..")
        all_filepaths = list(glob.iglob('%s/*.emb' % request.path))

        total_count = len(all_filepaths)
        if total_count <= 0:
            logging.info("No files for importing!")
            return pb2.SimpleReponse(message='No files for importing!')

        logging.info("Importing files count: %d" % total_count)

        pos = len(request.path) + 1
        def path_to_id(filepath):
            return int(filepath[pos:-4])

        for filepaths in chunks(all_filepaths, 10000):
            t0 = time.time()

            ids = map(path_to_id, filepaths)
            ids_filepaths = [(id, filepath) for id, filepath in zip(ids, filepaths) if is_new_emb(id, filepath)]

            xb = self._path_to_xb([filepath for _, filepath in ids_filepaths])
            ids = np.array([id for id, _ in ids_filepaths], dtype=np.int64)
            self.faiss_index.replace(xb, ids)

            for id, filepath in ids_filepaths:
                file_io.rename(filepath, self._get_filepath(id, mkdir=True), overwrite=True)

            logging.info("%d embeddings added %.3f s", xb.shape[0], time.time() - t0)
        return pb2.SimpleReponse(message='Imported, %d!' % total_count)

    def _more_recent_emb_file_exists(self, request):
        filepath = self._get_filepath(request.id)
        file_ts = self._get_file_ts(filepath)
        if file_ts:
            return file_ts >= request.created_at_ts
        return False

    def _get_file_ts(self, filepath):
        if self.remote_embedding_info:
            bucket_name, key = self.remote_embedding_info
            key = "%s/%s" % (key, filepath)
            try:
                res = self._client().head_object(Bucket=bucket_name, Key=key)
                return res['LastModified'].timestamp()
            except botocore.exceptions.ClientError as e:
                error_code = int(e.response['Error']['Code'])
                if error_code != 404:
                    raise e
        elif file_io.file_exists(filepath):
            return file_io.stat(filepath).mtime_nsec / 1000000000
        return None

    def fetch_embedding(self, request):
        t0 = time.time()
        embedding = self.embedding_service.get_embedding(request.url)
        if embedding is None:
            return
        logging.debug("embedding fetched %d, %.3f s", request.id, time.time() - t0)
        filepath = self._get_filepath(request.id, mkdir=True)

        if self.remote_embedding_info:
            bucket_name, key = self.remote_embedding_info
            key = "%s/%s" % (key, filepath)
            self._client().put_object(Bucket=bucket_name, Key=key, Body=embedding.tostring())
        else:
            file_io.write_string_to_file(filepath, embedding.tostring())

        return embedding

    def Fetch(self, request, context):
        total_count = len(request.items)
        fetched_count = 0

        results = []

        pool = ThreadPool(12)
        for item in request.items:
            result = pool.spawn(self.fetch_embedding, item)
            results.append(result)
        gevent.wait()

        for result in results:
            if result.get() is not None:
                fetched_count += 1

        return pb2.SimpleReponse(message='Fetched, %d of %d!' % (fetched_count, total_count))

    def _get_filepath(self, id, mkdir=False):
        if self.remote_embedding_info:
            path = '%d' % int(id / 10000)
        else:
            path = 'embeddings/%d' % int(id / 10000)
        if mkdir and not file_io.file_exists(path):
            file_io.create_dir(path)
        return '%s/%d.emb' % (path, id)

    def Search(self, request, context):
        filepath = self._get_filepath(request.id)
        if self.remote_embedding_info:
            embedding = self._remote_path_to_embedding(filepath)
            if embedding is None:
                return pb2.SearchReponse()
        else:
            if not file_io.file_exists(filepath):
                return pb2.SearchReponse()
            embedding = path_to_embedding(filepath)
        embedding = np.expand_dims(embedding, 0)
        D, I = self.faiss_index.search(embedding, request.count)
        return pb2.SearchReponse(ids=I[0], scores=D[0])

    def SearchByEmbedding(self, request, context):
        emb = np.array(request.embedding, dtype=np.float32)
        emb = np.expand_dims(emb, axis=0)
        D, I = self.faiss_index.search(emb, request.count)
        return pb2.SearchResponse(ids=I[0], scores=D[0])

    def Info(self, request, context):
        message = 'host: %s, total: %s, dim: %d' % (os.environ['HOSTNAME'], self.faiss_index.ntotal(), self.faiss_index.dim())
        if type(self.embedding_service) is RemoteImageEmbeddingService:
            embedding_info = self.embedding_service.info()
            message = '%s, embedding_info: %s' % (message, embedding_info)
        return pb2.SimpleReponse(message=message)

    def Remove(self, request, context):
        logging.debug('remove - id: %d', request.id)
        ids = np.array([request.id], dtype=np.int64)
        self.faiss_index.remove_ids(ids)

        filepath = self._get_filepath(request.id) 

        if self.remote_embedding_info:
            bucket_name, key = self.remote_embedding_info
            key = "%s/%s" % (key, filepath)
            try:
                client = self._client()
                client.head_object(Bucket=bucket_name, Key=key)
                client.delete_object(Bucket=bucket_name, Key=key)
                return pb2.SimpleReponse(message='Removed, %s!' % request.id)
            except botocore.exceptions.ClientError as e:
                error_code = int(e.response['Error']['Code'])
                if error_code != 404:
                    raise e
                logging.warn('no key: %s' % key)
        elif file_io.file_exists(filepath):
            file_io.delete_file(filepath)
            return pb2.SimpleReponse(message='Removed, %s!' % request.id)

        return pb2.SimpleReponse(message='Not existed, %s!' % request.id)

    def Similarity(self, request, context):
        filepaths = np.array([self._get_filepath(id) for id in request.ids])

        if self.remote_embedding_info:
            client = self._client()
            p = Pool(min(8, len(filepaths)))
            def _path_to_embedding(filepaths):
                return self._remote_path_to_embedding(filepaths, client)
            embs = [emb for emb in p.imap(_path_to_embedding, filepaths) if emb is not None]
            if len(embs) < 1:
                return pb2.SimilarityReponse(similarity=0.0, count=0)
            xb = np.array(embs, dtype=np.float32)
        else:
            exists = np.array([file_io.file_exists(x) for x in filepaths])
            filepaths = filepaths[exists]
            if len(filepaths) < 1:
                return pb2.SimilarityReponse(similarity=0.0, count=0)
            xb = self._path_to_xb(filepaths)

        count, features = xb.shape
        center = np.average(xb, 0)
        value = np.mean(cosine_similarity(center.reshape(-1, features), xb))
        return pb2.SimilarityReponse(similarity=value, count=count)

    def _remote_path_to_embedding(self, filepath, client=None):
        bucket_name, key = self.remote_embedding_info
        key = "%s/%s" % (key, filepath)
        client = client or self._client()
        try:
            res = client.get_object(Bucket=bucket_name, Key=key)
            return np.frombuffer(res['Body'].read(), dtype=np.float32)
        except client.exceptions.NoSuchKey:
            logging.warn('no key: %s' % key)
            return None

    def ClusterId(self, request, context):
        filepaths = np.array([self._get_filepath(id) for id in request.ids])
        exists = np.array([file_io.file_exists(x) for x in filepaths])
        filepaths = filepaths[exists]
        count = len(filepaths)
        if count < 1:
            return pb2.ClusterIdReponse(ids=[], cluster_ids=[])
        ids = np.array(request.ids)[exists]

        xb = self._path_to_xb(filepaths)
        D, I = self._kmeans_index.search(xb, 1)
        cluster_ids = list(I.flatten())
        return pb2.ClusterIdReponse(ids=ids, cluster_ids=cluster_ids)

    def TrainCluster(self, request, context):
        t0 = time.time()
        logging.info("kmeans_index training..")
        self._kmeans_index = self._train_kmeans(request.ncentroids)
        faiss.write_index(self._kmeans_index, request.save_filepath)
        logging.info("kmeans_index loaded %.2f s", time.time() - t0)
        return pb2.SimpleReponse(message='clustered')
