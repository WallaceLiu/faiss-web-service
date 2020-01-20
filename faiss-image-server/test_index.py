import unittest

from server import parse_args
from faiss_image_index import FaissImageIndex

import faissimageindex_pb2 as pb2

class TestIndex(unittest.TestCase):

    def test_parse_args(self):
        args = parse_args()
        self.assertFalse(args.no_save)

    def test_image_index(self):
        index_path = '/tmp/test.index'
        args = parse_args('--save_filepath {}'.format(index_path).split(' '))
        self.assertEqual(args.save_filepath, index_path)

        index = FaissImageIndex(args)
        self.assertIsNotNone(index)

        response = index.Info(pb2.Empty(), None)
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.message)
        print('info:', response.message)

        url = 'https://dnvefa72aowie.cloudfront.net/origin/profile/201901/8c632f250a456dd98a90f75eee362a959342685f6195857caaf2975b1eb8020b.png?q=82&s=80x80&t=crop'
        request = pb2.AddRequest(id=1, url=url, created_at_ts=0)
        response = index.Add(request, None)
        self.assertIsNotNone(response)
        #self.assertEqual(response.message, "Added, 1!")

        request = pb2.SearchRequest(id=1, count=10)
        response = index.Search(request, None)
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.ids)
        self.assertIsNotNone(response.scores)
        self.assertEqual(len(response.ids), 10)

        request = pb2.SearchByEmbeddingRequest(embedding=[0.0]*1536, count=10)
        response = index.SearchByEmbedding(request, None)
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.ids)
        self.assertIsNotNone(response.scores)
        self.assertEqual(len(response.ids), 10)

        request = pb2.IdRequest(id=1)
        response = index.Remove(request, None)
        self.assertIsNotNone(response)
        self.assertEqual(response.message, "Removed, 1!")

if __name__ == '__main__':
    unittest.main()
