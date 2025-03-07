# import json
# import time
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import VectorParams, Distance
# import os

# # Step 1: Load the embeddings from JSON
# def load_embeddings(file_path):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             embeddings = json.load(f)
#         print(f"Loaded {len(embeddings)} embeddings from {file_path}")
#         return embeddings
#     except FileNotFoundError:
#         print(f"Error: {file_path} not found.")
#         return []
#     except json.JSONDecodeError as e:
#         print(f"Error: Invalid JSON in {file_path}: {e}")
#         return []

# # Step 2: Upload embeddings to Qdrant
# def upload_to_qdrant(embeddings_file, api_key, cluster_url, collection_name="qest"):
#     # Initialize Qdrant client with API key and cluster URL
#     try:
#         print(f"Initializing Qdrant client with cluster URL: {cluster_url}")
#         client = QdrantClient(
#             url=cluster_url,
#             api_key=api_key
#         )
#     except Exception as e:
#         print(f"Error initializing Qdrant client: {e}")
#         return

#     # Check if the collection exists, if not create it
#     print(f"Checking if collection '{collection_name}' exists...")
#     collections = client.get_collections().collections
#     collection_names = [c.name for c in collections]
#     if collection_name not in collection_names:
#         dimension = 1024  # Fixed dimension for static-retrieval-mrl-en-v1 model
#         print(f"Creating new collection '{collection_name}' with dimension {dimension}...")
#         client.create_collection(
#             collection_name=collection_name,
#             vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
#         )
#         # Wait for collection to initialize
#         print("Waiting for collection to initialize...")
#         time.sleep(2)  # Qdrant Cloud typically initializes quickly

#     # Load embeddings from the file
#     embeddings = load_embeddings(embeddings_file)
#     if not embeddings:
#         print("No embeddings loaded. Exiting.")
#         return

#     # Load chunk texts from chunks.json for payload
#     try:
#         with open('chunks.json', 'r', encoding='utf-8') as f:
#             chunks = json.load(f)
#         if len(chunks) != len(embeddings):
#             print(f"Warning: Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)}). Using first {min(len(chunks), len(embeddings))} matches.")
#             chunks = chunks[:len(embeddings)]  # Truncate to match embeddings
#     except FileNotFoundError:
#         print("Warning: chunks.json not found. Payload will be empty.")
#         chunks = ["" for _ in embeddings]  # Fallback with empty payload

#     # Prepare points for upload
#     points = []
#     for i, (embedding, chunk_text) in enumerate(zip(embeddings, chunks)):
#         points.append({
#             'id': i,
#             'vector': embedding,
#             'payload': {'text': chunk_text if chunk_text else "No text available"}
#         })

#     # Upload in batches of 100
#     batch_size = 100
#     total_points = len(points)
#     print(f"Uploading {total_points} points in batches of {batch_size}...")

#     for i in range(0, total_points, batch_size):
#         batch = points[i:min(i + batch_size, total_points)]
#         print(f"Uploading batch {i // batch_size + 1}/{(total_points + batch_size - 1) // batch_size}...")
#         client.upsert(
#             collection_name=collection_name,
#             points=batch
#         )

#     print(f"Successfully uploaded {total_points} points to Qdrant collection '{collection_name}'")

# # Main execution
# def main():
#     input_embeddings_file = 'embeddings.json'  # Hardcoded input file path
#     api_key = "Api"  # Provided API key
#     # Replace the cluster_url with the URL from your Qdrant Cloud dashboard after creating the cluster
#     cluster_url = "url"  # Placeholder, update this
#     collection_name = "qest"  # Hardcoded collection name

#     if not os.path.exists(input_embeddings_file):
#         print(f"Error: Embeddings file '{input_embeddings_file}' not found")
#         return

#     upload_to_qdrant(input_embeddings_file, api_key, cluster_url, collection_name)

# if __name__ == "__main__":
#     main()

import os
import json
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Step 1: Load the embeddings from JSON with dimension check
def load_embeddings(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        if embeddings and isinstance(embeddings[0], list):
            dimension = len(embeddings[0])
            print(f"Loaded {len(embeddings)} embeddings from {file_path} with dimension {dimension}")
        else:
            print(f"Error: Embeddings data format invalid in {file_path}")
            return [], 0
        return embeddings, dimension
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return [], 0
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return [], 0

# Step 2: Upload embeddings to Qdrant with dimension enforcement
def upload_to_qdrant(embeddings_file, api_key, cluster_url, collection_name="qest"):
    # Initialize Qdrant client with API key and cluster URL
    try:
        print(f"Initializing Qdrant client with cluster URL: {cluster_url}")
        client = QdrantClient(
            url=cluster_url,
            api_key=api_key
        )
    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")
        return

    # Load embeddings and determine dimension
    embeddings, dimension = load_embeddings(embeddings_file)
    if not embeddings:
        print("No embeddings loaded. Exiting.")
        return

    # Check if the collection exists and enforce correct dimension
    print(f"Checking if collection '{collection_name}' exists...")
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    if collection_name in collection_names:
        collection_info = client.get_collection(collection_name=collection_name)
        if collection_info.config.params.vectors.size != dimension:
            print(f"Existing collection '{collection_name}' has dimension {collection_info.config.params.vectors.size}, but data requires {dimension}. Recreating collection...")
            client.delete_collection(collection_name=collection_name)
            # Use a simple time.sleep instead of wait_for_collection
            time.sleep(2)  # Wait for deletion

    # Create collection with the correct dimension
    if collection_name not in [c.name for c in client.get_collections().collections]:
        print(f"Creating new collection '{collection_name}' with dimension {dimension}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )
        # Replace wait_for_collection with a simple delay
        print("Waiting for collection to initialize...")
        time.sleep(2)  # Wait for 2 seconds to allow collection initialization

    # Load chunk texts from chunks.json for payload
    try:
        with open('clear_chunk.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        if len(chunks) != len(embeddings):
            print(f"Warning: Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)}). Using first {min(len(chunks), len(embeddings))} matches.")
            chunks = chunks[:len(embeddings)]  # Truncate to match embeddings
    except FileNotFoundError:
        print("Warning: chunks.json not found. Payload will be empty.")
        chunks = ["" for _ in embeddings]  # Fallback with empty payload

    # Prepare points for upload
    points = []
    for i, (embedding, chunk_text) in enumerate(zip(embeddings, chunks)):
        points.append({
            'id': i,
            'vector': embedding,
            'payload': {'text': chunk_text if chunk_text else "No text available"}
        })

    # Upload in batches of 100
    batch_size = 100
    total_points = len(points)
    print(f"Uploading {total_points} points in batches of {batch_size}...")

    for i in range(0, total_points, batch_size):
        batch = points[i:min(i + batch_size, total_points)]
        print(f"Uploading batch {i // batch_size + 1}/{(total_points + batch_size - 1) // batch_size}...")
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
        except Exception as e:
            print(f"Upload failed for batch {i // batch_size + 1}: {e}")
            return

    print(f"Successfully uploaded {total_points} points to Qdrant collection '{collection_name}'")

# Main execution
def main():
    input_embeddings_file = 'embeddings.json'  # Hardcoded input file path
    api_key = os.environ.get('QDRANT_API_KEY', 'Api')  # Use environment variable for API key
    cluster_url = os.environ.get('QDRANT_CLUSTER_URL', 'url')  # Use environment variable for cluster URL
    collection_name = "qest"  # Hardcoded collection name

    if not os.path.exists(input_embeddings_file):
        print(f"Error: Embeddings file '{input_embeddings_file}' not found")
        return

    upload_to_qdrant(input_embeddings_file, api_key, cluster_url, collection_name)

if __name__ == "__main__":
    main()
