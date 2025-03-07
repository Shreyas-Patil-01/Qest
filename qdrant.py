import os
import json
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

def load_embeddings(file_path):
    """Load embeddings from JSON file with 'chunk_id', 'text', and 'embedding'."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        if not isinstance(embeddings, list) or not all('embedding' in chunk for chunk in embeddings):
            raise ValueError("Invalid JSON format: Expected a list of dictionaries with 'embedding' key.")
        print(f"Loaded {len(embeddings)} embeddings from {file_path}")
        return embeddings
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading {file_path}: {e}")
        return []

def upload_to_qdrant(embeddings_file, api_key, cluster_url, collection_name="qest"):
    """Upload embeddings with 'chunk_id', 'text', and 'embedding' to Qdrant."""
    try:
        print(f"Initializing Qdrant client with cluster URL: {cluster_url}")
        client = QdrantClient(url=cluster_url, api_key=api_key)
    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")
        return
    
    # Load embeddings
    embeddings = load_embeddings(embeddings_file)
    if not embeddings:
        print("No embeddings loaded. Exiting.")
        return
    
    # Determine vector dimension
    dimension = len(embeddings[0]['embedding'])
    
    # Check if collection exists and has the correct dimension
    print(f"Checking if collection '{collection_name}' exists...")
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name in collection_names:
        collection_info = client.get_collection(collection_name=collection_name)
        if collection_info.config.params.vectors.size != dimension:
            print(f"Collection '{collection_name}' has dimension {collection_info.config.params.vectors.size}, expected {dimension}. Recreating...")
            client.delete_collection(collection_name=collection_name)
            time.sleep(2)
    
    if collection_name not in [c.name for c in client.get_collections().collections]:
        print(f"Creating collection '{collection_name}' with dimension {dimension}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )
        time.sleep(2)
    
    # Prepare points for upload
    points = []
    for chunk in embeddings:
        points.append({
            'id': chunk['chunk_id'],
            'vector': chunk['embedding'],
            'payload': {'text': chunk['text']}
        })
    
    # Upload in batches of 100
    batch_size = 100
    total_points = len(points)
    print(f"Uploading {total_points} points in batches of {batch_size}...")
    
    for i in range(0, total_points, batch_size):
        batch = points[i:min(i + batch_size, total_points)]
        print(f"Uploading batch {i // batch_size + 1}/{(total_points + batch_size - 1) // batch_size}...")
        try:
            client.upsert(collection_name=collection_name, points=batch)
        except Exception as e:
            print(f"Upload failed for batch {i // batch_size + 1}: {e}")
            return
    
    print(f"Successfully uploaded {total_points} points to Qdrant collection '{collection_name}'")

def main():
    input_embeddings_file = 'embeddings.json'
    api_key = os.environ.get('QDRANT_API_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.UGN42H_g9n7ieQLdB-DI8Awg_LL26nLGDxkC6f1JIEM')
    cluster_url = os.environ.get('QDRANT_CLUSTER_URL', 'https://dbc8fef8-478c-4529-ae46-20e4590e845d.europe-west3-0.gcp.cloud.qdrant.io')
    collection_name = "qest"
    
    if not os.path.exists(input_embeddings_file):
        print(f"Error: Embeddings file '{input_embeddings_file}' not found")
        return
    
    upload_to_qdrant(input_embeddings_file, api_key, cluster_url, collection_name)

if __name__ == "__main__":
    main()
