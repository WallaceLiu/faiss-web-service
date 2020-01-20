docker run --rm -it \
  -v $(pwd):/app \
  -v /tmp/models/faiss-image-server2.index:/tmp/test.index \
  daangn/faiss-image-server:develop test_index.py
