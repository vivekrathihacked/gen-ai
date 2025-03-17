from pymilvus import MilvusClient

client = MilvusClient("milvus_rag.db")

if client.has_collection(collection_name="rag_milvus"):
    client.drop_collection(collection_name="rag_milvus")
client.create_collection(
    collection_name="rag_milvus",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)