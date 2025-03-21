# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import muopdb_pb2 as muopdb__pb2

GRPC_GENERATED_VERSION = '1.68.1'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in muopdb_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class AggregatorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Get = channel.unary_unary(
                '/muopdb.Aggregator/Get',
                request_serializer=muopdb__pb2.GetRequest.SerializeToString,
                response_deserializer=muopdb__pb2.GetResponse.FromString,
                _registered_method=True)


class AggregatorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Get(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_AggregatorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Get': grpc.unary_unary_rpc_method_handler(
                    servicer.Get,
                    request_deserializer=muopdb__pb2.GetRequest.FromString,
                    response_serializer=muopdb__pb2.GetResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'muopdb.Aggregator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('muopdb.Aggregator', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class Aggregator(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Get(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/muopdb.Aggregator/Get',
            muopdb__pb2.GetRequest.SerializeToString,
            muopdb__pb2.GetResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class IndexServerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CreateCollection = channel.unary_unary(
                '/muopdb.IndexServer/CreateCollection',
                request_serializer=muopdb__pb2.CreateCollectionRequest.SerializeToString,
                response_deserializer=muopdb__pb2.CreateCollectionResponse.FromString,
                _registered_method=True)
        self.Search = channel.unary_unary(
                '/muopdb.IndexServer/Search',
                request_serializer=muopdb__pb2.SearchRequest.SerializeToString,
                response_deserializer=muopdb__pb2.SearchResponse.FromString,
                _registered_method=True)
        self.Insert = channel.unary_unary(
                '/muopdb.IndexServer/Insert',
                request_serializer=muopdb__pb2.InsertRequest.SerializeToString,
                response_deserializer=muopdb__pb2.InsertResponse.FromString,
                _registered_method=True)
        self.InsertPacked = channel.unary_unary(
                '/muopdb.IndexServer/InsertPacked',
                request_serializer=muopdb__pb2.InsertPackedRequest.SerializeToString,
                response_deserializer=muopdb__pb2.InsertPackedResponse.FromString,
                _registered_method=True)
        self.Flush = channel.unary_unary(
                '/muopdb.IndexServer/Flush',
                request_serializer=muopdb__pb2.FlushRequest.SerializeToString,
                response_deserializer=muopdb__pb2.FlushResponse.FromString,
                _registered_method=True)


class IndexServerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def CreateCollection(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Search(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Insert(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def InsertPacked(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Flush(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_IndexServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'CreateCollection': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateCollection,
                    request_deserializer=muopdb__pb2.CreateCollectionRequest.FromString,
                    response_serializer=muopdb__pb2.CreateCollectionResponse.SerializeToString,
            ),
            'Search': grpc.unary_unary_rpc_method_handler(
                    servicer.Search,
                    request_deserializer=muopdb__pb2.SearchRequest.FromString,
                    response_serializer=muopdb__pb2.SearchResponse.SerializeToString,
            ),
            'Insert': grpc.unary_unary_rpc_method_handler(
                    servicer.Insert,
                    request_deserializer=muopdb__pb2.InsertRequest.FromString,
                    response_serializer=muopdb__pb2.InsertResponse.SerializeToString,
            ),
            'InsertPacked': grpc.unary_unary_rpc_method_handler(
                    servicer.InsertPacked,
                    request_deserializer=muopdb__pb2.InsertPackedRequest.FromString,
                    response_serializer=muopdb__pb2.InsertPackedResponse.SerializeToString,
            ),
            'Flush': grpc.unary_unary_rpc_method_handler(
                    servicer.Flush,
                    request_deserializer=muopdb__pb2.FlushRequest.FromString,
                    response_serializer=muopdb__pb2.FlushResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'muopdb.IndexServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('muopdb.IndexServer', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class IndexServer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CreateCollection(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/muopdb.IndexServer/CreateCollection',
            muopdb__pb2.CreateCollectionRequest.SerializeToString,
            muopdb__pb2.CreateCollectionResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Search(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/muopdb.IndexServer/Search',
            muopdb__pb2.SearchRequest.SerializeToString,
            muopdb__pb2.SearchResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Insert(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/muopdb.IndexServer/Insert',
            muopdb__pb2.InsertRequest.SerializeToString,
            muopdb__pb2.InsertResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def InsertPacked(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/muopdb.IndexServer/InsertPacked',
            muopdb__pb2.InsertPackedRequest.SerializeToString,
            muopdb__pb2.InsertPackedResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Flush(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/muopdb.IndexServer/Flush',
            muopdb__pb2.FlushRequest.SerializeToString,
            muopdb__pb2.FlushResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
