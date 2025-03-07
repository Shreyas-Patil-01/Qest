import json
from sentence_transformers import SentenceTransformer
import numpy as np

def load_chunks(file_path):
    """Load JSON file containing chunks with 'chunk_id' and 'text'."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Invalid JSON format: Expected a list of dictionaries.")
        print(f"Loaded {len(data)} chunks from {file_path}")
        return data
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading {file_path}: {e}")
        return []

def generate_embeddings(chunks, model_name="sentence-transformers/static-retrieval-mrl-en-v1"):
    """Generate embeddings for the 'text' attribute of each chunk."""
    try:
        model = SentenceTransformer(model_name)
        texts = [chunk["text"] for chunk in chunks if "text" in chunk]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        # Attach embeddings back to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
        
        return chunks
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def save_embeddings(embeddings, output_file_path):
    """Save embeddings to a JSON file, including 'chunk_id', 'text', and 'embedding'."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, indent=4, ensure_ascii=False)
        print(f"Embeddings saved to {output_file_path} with {len(embeddings)} entries")
    except Exception as e:
        print(f"Error saving to {output_file_path}: {e}")

def main():
    input_chunks_file = 'final_chunk.json'  # Input file path
    output_embedding_file = 'embeddings.json'  # Output file path
    
    # Load chunks
    chunks = load_chunks(input_chunks_file)
    
    if not chunks:
        print("No valid chunks loaded. Exiting.")
        return
    
    # Generate embeddings
    embedded_chunks = generate_embeddings(chunks)
    
    if not embedded_chunks:
        print("No embeddings generated. Exiting.")
        return
    
    # Save embeddings
    save_embeddings(embedded_chunks, output_embedding_file)

if __name__ == '__main__':
    main()
