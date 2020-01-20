import logging
import grpc
import numpy as np
from time import time

import image_embedding_pb2 as pb2
import image_embedding_pb2_grpc as pb2_grpc

class ImageEmbeddingService:
    KEEP_STUB_SEC = 60

    def __init__(self, host, load_balancing=False, timeout=10):
        self.host = host
        self.load_balancing = load_balancing
        self._timeout = timeout

        with self._channel() as channel:
            stub = pb2_grpc.ImageEmbeddingStub(channel)
            response = stub.Dimension(pb2.Empty(), timeout=1)
        self._dim = response.dim

    def _channel(self):
        if self.load_balancing:
            return grpc.insecure_channel(self.host, [("grpc.lb_policy_name", "round_robin")])
        else:
            return grpc.insecure_channel(self.host)

    def dim(self):
        return self._dim

    def get_embedding(self, url, is_retry=False):
        try:
            with self._channel() as channel:
                stub = pb2_grpc.ImageEmbeddingStub(channel)
                response = stub.Embedding(pb2.EmbeddingRequest(url=url), timeout=self._timeout)
        except grpc.RpcError as e:
            msg = str(e)
            if 'Forbidden' in msg:
                return None
            if not is_retry:
                logging.warn("embedding error: %s", msg)
                return self.get_embedding(url, is_retry=True)
            raise e
        return np.array(response.embedding, dtype=np.float32)

    def info(self):
        with self._channel() as channel:
            stub = pb2_grpc.ImageEmbeddingStub(channel)
            response = stub.Info(pb2.Empty(), timeout=1)
        return response.message
