from qdrant_client import QdrantClient,models


#client = QdrantClient("http://vectordb:6333")
#client = QdrantClient("http://localhost:6333") # Connect to existing Qdrant instance
client = QdrantClient(":memory:")
#client = QdrantClient("localhost", port=6333)
collection_name = "imagebind_data"

# Check if collection exists
if client.collection_exists(collection_name):
    pass
    #print(f"Collection '{collection_name}' already exists.")
else:
    # Create the collection
    client.create_collection(
    collection_name=collection_name,
    vectors_config={ #Named Vectors
            "audio": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "image": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "text": models.VectorParams(size=1024, distance=models.Distance.EUCLID),
        }
    )
    #print(f"Collection '{collection_name}' created.")
print("[INFO] Client created...")
