# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import faissimageindex_pb2 as faissimageindex__pb2


class ImageIndexStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Add = channel.unary_unary(
        '/faiss.ImageIndex/Add',
        request_serializer=faissimageindex__pb2.AddRequest.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimpleReponse.FromString,
        )
    self.Remove = channel.unary_unary(
        '/faiss.ImageIndex/Remove',
        request_serializer=faissimageindex__pb2.IdRequest.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimpleReponse.FromString,
        )
    self.Search = channel.unary_unary(
        '/faiss.ImageIndex/Search',
        request_serializer=faissimageindex__pb2.SearchRequest.SerializeToString,
        response_deserializer=faissimageindex__pb2.SearchReponse.FromString,
        )
    self.SearchByEmbedding = channel.unary_unary(
        '/faiss.ImageIndex/SearchByEmbedding',
        request_serializer=faissimageindex__pb2.SearchByEmbeddingRequest.SerializeToString,
        response_deserializer=faissimageindex__pb2.SearchResponse.FromString,
        )
    self.Fetch = channel.unary_unary(
        '/faiss.ImageIndex/Fetch',
        request_serializer=faissimageindex__pb2.FetchRequest.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimpleReponse.FromString,
        )
    self.Info = channel.unary_unary(
        '/faiss.ImageIndex/Info',
        request_serializer=faissimageindex__pb2.Empty.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimpleReponse.FromString,
        )
    self.Save = channel.unary_unary(
        '/faiss.ImageIndex/Save',
        request_serializer=faissimageindex__pb2.Empty.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimpleReponse.FromString,
        )
    self.Train = channel.unary_unary(
        '/faiss.ImageIndex/Train',
        request_serializer=faissimageindex__pb2.Empty.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimpleReponse.FromString,
        )
    self.Import = channel.unary_unary(
        '/faiss.ImageIndex/Import',
        request_serializer=faissimageindex__pb2.ImportRequest.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimpleReponse.FromString,
        )
    self.Migrate = channel.unary_unary(
        '/faiss.ImageIndex/Migrate',
        request_serializer=faissimageindex__pb2.Empty.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimpleReponse.FromString,
        )
    self.Similarity = channel.unary_unary(
        '/faiss.ImageIndex/Similarity',
        request_serializer=faissimageindex__pb2.SimilarityRequest.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimilarityReponse.FromString,
        )
    self.ClusterId = channel.unary_unary(
        '/faiss.ImageIndex/ClusterId',
        request_serializer=faissimageindex__pb2.SimilarityRequest.SerializeToString,
        response_deserializer=faissimageindex__pb2.ClusterIdReponse.FromString,
        )
    self.TrainCluster = channel.unary_unary(
        '/faiss.ImageIndex/TrainCluster',
        request_serializer=faissimageindex__pb2.TrainClusterRequest.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimpleReponse.FromString,
        )
    self.Reset = channel.unary_unary(
        '/faiss.ImageIndex/Reset',
        request_serializer=faissimageindex__pb2.Empty.SerializeToString,
        response_deserializer=faissimageindex__pb2.SimpleReponse.FromString,
        )


class ImageIndexServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Add(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Remove(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Search(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def SearchByEmbedding(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Fetch(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Info(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Save(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Train(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Import(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Migrate(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Similarity(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ClusterId(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def TrainCluster(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Reset(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ImageIndexServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Add': grpc.unary_unary_rpc_method_handler(
          servicer.Add,
          request_deserializer=faissimageindex__pb2.AddRequest.FromString,
          response_serializer=faissimageindex__pb2.SimpleReponse.SerializeToString,
      ),
      'Remove': grpc.unary_unary_rpc_method_handler(
          servicer.Remove,
          request_deserializer=faissimageindex__pb2.IdRequest.FromString,
          response_serializer=faissimageindex__pb2.SimpleReponse.SerializeToString,
      ),
      'Search': grpc.unary_unary_rpc_method_handler(
          servicer.Search,
          request_deserializer=faissimageindex__pb2.SearchRequest.FromString,
          response_serializer=faissimageindex__pb2.SearchReponse.SerializeToString,
      ),
      'SearchByEmbedding': grpc.unary_unary_rpc_method_handler(
          servicer.SearchByEmbedding,
          request_deserializer=faissimageindex__pb2.SearchByEmbeddingRequest.FromString,
          response_serializer=faissimageindex__pb2.SearchResponse.SerializeToString,
      ),
      'Fetch': grpc.unary_unary_rpc_method_handler(
          servicer.Fetch,
          request_deserializer=faissimageindex__pb2.FetchRequest.FromString,
          response_serializer=faissimageindex__pb2.SimpleReponse.SerializeToString,
      ),
      'Info': grpc.unary_unary_rpc_method_handler(
          servicer.Info,
          request_deserializer=faissimageindex__pb2.Empty.FromString,
          response_serializer=faissimageindex__pb2.SimpleReponse.SerializeToString,
      ),
      'Save': grpc.unary_unary_rpc_method_handler(
          servicer.Save,
          request_deserializer=faissimageindex__pb2.Empty.FromString,
          response_serializer=faissimageindex__pb2.SimpleReponse.SerializeToString,
      ),
      'Train': grpc.unary_unary_rpc_method_handler(
          servicer.Train,
          request_deserializer=faissimageindex__pb2.Empty.FromString,
          response_serializer=faissimageindex__pb2.SimpleReponse.SerializeToString,
      ),
      'Import': grpc.unary_unary_rpc_method_handler(
          servicer.Import,
          request_deserializer=faissimageindex__pb2.ImportRequest.FromString,
          response_serializer=faissimageindex__pb2.SimpleReponse.SerializeToString,
      ),
      'Migrate': grpc.unary_unary_rpc_method_handler(
          servicer.Migrate,
          request_deserializer=faissimageindex__pb2.Empty.FromString,
          response_serializer=faissimageindex__pb2.SimpleReponse.SerializeToString,
      ),
      'Similarity': grpc.unary_unary_rpc_method_handler(
          servicer.Similarity,
          request_deserializer=faissimageindex__pb2.SimilarityRequest.FromString,
          response_serializer=faissimageindex__pb2.SimilarityReponse.SerializeToString,
      ),
      'ClusterId': grpc.unary_unary_rpc_method_handler(
          servicer.ClusterId,
          request_deserializer=faissimageindex__pb2.SimilarityRequest.FromString,
          response_serializer=faissimageindex__pb2.ClusterIdReponse.SerializeToString,
      ),
      'TrainCluster': grpc.unary_unary_rpc_method_handler(
          servicer.TrainCluster,
          request_deserializer=faissimageindex__pb2.TrainClusterRequest.FromString,
          response_serializer=faissimageindex__pb2.SimpleReponse.SerializeToString,
      ),
      'Reset': grpc.unary_unary_rpc_method_handler(
          servicer.Reset,
          request_deserializer=faissimageindex__pb2.Empty.FromString,
          response_serializer=faissimageindex__pb2.SimpleReponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'faiss.ImageIndex', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
