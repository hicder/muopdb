import grpc
import muopdb_pb2
import muopdb_pb2_grpc

"""
Make sure to build the proto files first:
pip install grpcio protobuf grpcio-tools ollama
python3 -m grpc.tools.protoc -I=rs/proto/proto --python_out=py/ --grpc_python_out=py/ rs/proto/proto/muopdb.proto
"""

class IndexServerClient:
    def __init__(self, host="localhost", port=9002):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = muopdb_pb2_grpc.IndexServerStub(self.channel)

    def create_collection(self, collection_name: str):
        request = muopdb_pb2.CreateCollectionRequest(
            collection_name=collection_name,
        )
        response = self.stub.CreateCollection(request)
        return response

    def search(self, collection_name: str, vector: list[float], top_k: int, ef_construction: int, record_metrics: bool=False, user_ids: list[int]=[0]):
        request = muopdb_pb2.SearchRequest(
            collection_name=collection_name,
            vector=vector,
            top_k=top_k,
            ef_construction=ef_construction,
            record_metrics=record_metrics,
            user_ids=user_ids
        )
        response = self.stub.Search(request)
        return response

    def insert(self, collection_name: str, ids: list[int], vectors: list[float]):
         request = muopdb_pb2.InsertRequest(
            collection_name=collection_name,
            ids=ids,
            vectors=vectors
         )
         response = self.stub.Insert(request)
         return response
    
    def flush(self, collection_name: str):
        request = muopdb_pb2.FlushRequest(
            collection_name=collection_name
        )
        response = self.stub.Flush(request)
        return response

    def close(self):
        self.channel.close()

if __name__ == '__main__':

    # Example usage for IndexServer
    index_server_client = IndexServerClient()
    try:
        search_response = index_server_client.search(
            index_name="my_index",
            vector=[0.4, 0.5, 0.6],
            top_k=3,
            ef_construction=50,
            record_metrics=False
        )
        print("Index Server Search Response:", search_response)


        insert_response = index_server_client.insert(
            collection_name = "my_collection",
            ids = [1, 2, 3],
            vectors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        print("Index Server Insert Response:", insert_response)

        flush_response = index_server_client.flush(
            collection_name = "my_collection"
        )
        print("Index Server Flush Response:", flush_response)

    except grpc.RpcError as e:
        print(f"Index Server RPC Error: {e}")
    finally:
        index_server_client.close()
