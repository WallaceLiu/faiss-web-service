# -*- coding: utf-8 -*-
import time
import logging
import sys
import argparse
import signal
from concurrent import futures

import grpc

import faissimageindex_pb2_grpc as pb2_grpc
from faiss_image_index import FaissImageIndex

_ONE_DAY_IN_SECONDS = 60 * 60

def serve(args):
    logging.info('server loading')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    faiss_image_index = FaissImageIndex(args)
    pb2_grpc.add_ImageIndexServicer_to_server(faiss_image_index, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info('server started')

    # for docker heath check
    with open('/tmp/status', 'w') as f:
        f.write('started')

    def stop_serve(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop_serve)
    signal.signal(signal.SIGTERM, stop_serve)

    faiss_image_index.Migrate(None, None)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
            if not args.no_save:
                faiss_image_index.save()
    except KeyboardInterrupt:
        server.stop(0)
        if not args.no_save:
            faiss_image_index.save()
        logging.info('server stopped')

def train(args):
    faiss_image_index = FaissImageIndex(args)
    faiss_image_index.save()
    faiss_image_index.export_ids()

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Faiss Image Server')
    parser.add_argument('--log', help='log filepath')
    parser.add_argument('--debug', dest='debug', action='store_true', help='debug')
    parser.add_argument('--train_count', type=int, default=100000, help='max train count')
    parser.add_argument('--max_nlist', type=int, default=4096, help='max_nlist')
    parser.add_argument('--save_filepath', default='models/image.index', help='save file path')
    parser.add_argument('--model', default='inception_v4', help='Inception model version')
    parser.add_argument('--train_only', dest='train_only', action='store_true', help='train only flag')
    parser.add_argument('--kmeans_filepath', help='save file path')
    parser.add_argument('--ncentroids', type=int, default=100, help='n centroids')
    parser.add_argument('--max_workers', type=int, default=10, help='Max workers count')
    parser.add_argument('--remote_embedding_host', type=str, help='Remote image embedding server host')
    parser.add_argument('--lb', action='store_true', help='Use load balancing')
    parser.add_argument('--remote_embedding_timeout', type=int, default=10, help='Timeout to request remote embeddings')
    parser.add_argument('--remote_embedding_path', help='remote embedding path')
    parser.add_argument('--no_save', dest='no_save', action='store_true', help='no save')
    parser.add_argument('--nprobe', type=int, default=1, help='nprobe for the search performance')
    parser.set_defaults(debug=False)
    parser.set_defaults(train_only=False)
    parser.set_defaults(lb=False)
    parser.set_defaults(no_save=False)
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    if args.log:
        handler = logging.FileHandler(filename=args.log)
    else:
        handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s - %(message)s')
    handler.setFormatter(formatter)
    root = logging.getLogger()
    level = args.debug and logging.DEBUG or logging.INFO
    root.setLevel(level)
    root.addHandler(handler)
    if args.debug:
        logging.debug('debug mode enabled')

    if args.train_only:
        train(args)
    else:
        serve(args)
