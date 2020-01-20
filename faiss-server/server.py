# -*- coding: utf-8 -*-
import time
import logging
import sys
import signal
from concurrent import futures

import grpc
import click

import faissindex_pb2_grpc as pb2_grpc
from faiss_server import FaissServer

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

@click.command()
@click.argument('dim', type=int)
@click.option('--save-path', default='faiss_server.index', help='index save path')
@click.option('--keys-path', help='keys file path')
@click.option('--log', help='log filepath')
@click.option('--debug', is_flag=True, help='debug')
@click.option('--no-save', is_flag=True, help='no save when stop service')
@click.option('--max-workers', default=1, help='workers count')
@click.option('--nprobe', default=1, help='nprobe for the search quality')
def main(dim, save_path, keys_path, log, debug, no_save, max_workers, nprobe):
    if log:
        handler = logging.FileHandler(filename=log)
    else:
        handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s - %(message)s')
    handler.setFormatter(formatter)
    root = logging.getLogger()
    level = debug and logging.DEBUG or logging.INFO
    root.setLevel(level)
    root.addHandler(handler)

    logging.info('server loading...')
    logging.info('max workers: %d', max_workers)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = FaissServer(dim, save_path, keys_path, nprobe)
    pb2_grpc.add_ServerServicer_to_server(servicer, server)
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

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        if not no_save:
            servicer.save()
        logging.info('server stopped')


if __name__ == '__main__':
    main()
