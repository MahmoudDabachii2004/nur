"""
colab_indexer.py

This script is designed to run inside Google Colab using a GPU runtime (T4/V100/A100).
It handles:
1. Installing the required libraries (FlagEmbedding, chromadb, etc.).
2. Presenting file upload prompts for the three processed chunk files:
   - quran_chunks.jsonl
   - hadith_chunks.jsonl
   - tafsir_chunks.jsonl
3. Encoding the chunks into dense and sparse embeddings using BGE-M3 on CUDA.
4. Structuring and saving the databases (ChromaDB + Sparse JSON files).
5. Zipping the results and triggering an automatic browser download.
"""

import json
import os
import sys
import time
import zipfile
import subprocess
import numpy as np

# Ensure necessary libraries are installed when running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Detected Google Colab environment. Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "FlagEmbedding", "chromadb", "tqdm"], check=True)
    print("Requirements installed successfully!")

from tqdm import tqdm
import chromadb

# Define path constants for the database structure
DATA_DIR = "./data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
SPARSE_DIR = os.path.join(DATA_DIR, "sparse")

# Ensure target directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(SPARSE_DIR, exist_ok=True)


def upload_files_colab():
    """
    Prompt the user to upload chunk files to Google Colab.
    
    Returns:
        None
    """
    from google.colab import files
    print("\n--- STEP 1: UPLOAD CHUNK FILES ---")
    print("Please upload your chunk files (quran_chunks.jsonl, hadith_chunks.jsonl, tafsir_chunks.jsonl).")
    
    uploaded = files.upload()
    for filename, content in uploaded.items():
        dest_path = os.path.join(PROCESSED_DIR, filename)
        with open(dest_path, "wb") as f:
            f.write(content)
        print(f"  Saved {filename} to {dest_path}")


def load_jsonl(path):
    """
    Load a JSONL file into a list of dictionaries.
    
    Args:
        path (str): Path to the JSONL file.
        
    Returns:
        list: List of dictionary records.
    """
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def extract_metadata(chunk):
    """
    Extract flat ChromaDB-compatible metadata from a nested dictionary chunk.
    
    Args:
        chunk (dict): The chunk dictionary containing metadata.
        
    Returns:
        dict: A flat dictionary of supported metadata values.
    """
    meta = {}
    for k, v in chunk.items():
        if k in ("id", "embedding_text", "text_ar_normalized", "tafsir_text_ar_normalized"):
            continue
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
        elif isinstance(v, dict):
            for dk, dv in v.items():
                if isinstance(dv, (str, int, float, bool)):
                    meta[f"{k}_{dk}"] = dv
    return meta


def embed_and_store_unified(collection_name, chunks, model, batch_size=128):
    """
    Encode chunks using BGEM3FlagModel and store outputs:
    - Dense vectors go to ChromaDB under '{collection_name}_dense'
    - Sparse vectors go to 'data/sparse/{collection_name}_sparse.json'
    
    Args:
        collection_name (str): Name of the collection.
        chunks (list): List of chunks to embed.
        model (BGEM3FlagModel): Loaded model instance.
        batch_size (int): Batch size to use for processing.
        
    Returns:
        int: Number of items successfully indexed.
    """
    print(f"\n[{collection_name}] Ingesting {len(chunks)} chunks using batch_size={batch_size}...")
    
    # Initialize Persistent ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Clean up any existing collections
    for suffix in ["_dense", "_sparse", ""]:
        try:
            client.delete_collection(f"{collection_name}{suffix}")
        except Exception:
            pass
            
    # Create the target dense collection
    col = client.create_collection(
        name=f"{collection_name}_dense",
        metadata={"description": f"NUR {collection_name} dense (semantic)"}
    )
    
    ids = [c["id"] for c in chunks]
    documents = [c.get("embedding_text", "") for c in chunks]
    metadatas = [extract_metadata(c) for c in chunks]
    
    total = len(documents)
    start_time = time.time()
    sparse_data = {}
    
    # Perform batched encoding on GPU
    for i in tqdm(range(0, total, batch_size), desc=f"Embedding {collection_name}"):
        batch_docs = documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        # Call model.encode to fetch dense and sparse weights simultaneously
        output = model.encode(
            batch_docs,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        
        # --- Process Dense Embeddings ---
        dense_vecs = output['dense_vecs']
        if hasattr(dense_vecs, 'cpu'):
            dense_vecs = dense_vecs.cpu().numpy()
            
        # Normalize dense vectors to ensure cosine similarity matches dot product
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense_vecs = dense_vecs / norms
        
        col.add(
            ids=batch_ids,
            embeddings=dense_vecs.tolist(),
            documents=batch_docs,
            metadatas=batch_metas
        )
        
        # --- Process Sparse Embeddings ---
        lexical_weights = output['lexical_weights']
        for j, lw in enumerate(lexical_weights):
            if lw and len(lw) > 0:
                sparse_data[batch_ids[j]] = {
                    "indices": [int(k) for k in lw.keys()],
                    "values": [float(v) for v in lw.values()]
                }
                
    # Save sparse dictionaries to JSON
    sparse_path = os.path.join(SPARSE_DIR, f"{collection_name}_sparse.json")
    with open(sparse_path, "w", encoding="utf-8") as f:
        json.dump(sparse_data, f)
        
    elapsed = time.time() - start_time
    print(f"Finished {collection_name} in {elapsed:.1f}s ({total/elapsed:.1f} chunks/s)")
    print(f"ChromaDB: {col.count()} docs | Sparse: {len(sparse_data)} entries")
    return col.count()


def zip_results(zip_filename="nur_indexed_data.zip"):
    """
    Zip the resulting chroma_db and sparse directories.
    
    Args:
        zip_filename (str): Name of the generated zip file.
        
    Returns:
        str: Absolute path of the generated zip file.
    """
    print("\n--- STEP 3: ZIPPING EMBEDDINGS ---")
    zip_path = os.path.join("./", zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Zip chroma_db directory
        for root, dirs, files_list in os.walk(CHROMA_DIR):
            for file in files_list:
                file_path = os.path.join(root, file)
                # Compute relative path to keep structure inside the zip
                rel_path = os.path.relpath(file_path, "./")
                zipf.write(file_path, rel_path)
                
        # Zip sparse directory
        for root, dirs, files_list in os.walk(SPARSE_DIR):
            for file in files_list:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, "./")
                zipf.write(file_path, rel_path)
                
    print(f"Successfully zipped files to {zip_path} (Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB)")
    return zip_path


def main():
    """
    Main orchestrator for Colab GPU-accelerated embeddings generation.
    
    Returns:
        None
    """
    # Ask for uploads if running inside Google Colab
    if IN_COLAB:
        upload_files_colab()
    else:
        print("Running in local environment. Ensure chunks are in 'data/processed/' directory.")
        
    # Check that chunk files exist
    quran_path = os.path.join(PROCESSED_DIR, "quran_chunks.jsonl")
    hadith_path = os.path.join(PROCESSED_DIR, "hadith_chunks.jsonl")
    tafsir_path = os.path.join(PROCESSED_DIR, "tafsir_chunks.jsonl")
    
    if not (os.path.exists(quran_path) and os.path.exists(hadith_path) and os.path.exists(tafsir_path)):
        print("\n❌ Error: Missing chunk files in data/processed!")
        print("Make sure you uploaded: quran_chunks.jsonl, hadith_chunks.jsonl, tafsir_chunks.jsonl")
        sys.exit(1)
        
    # Load embedding model
    print("\n--- STEP 2: EMBEDDING GENERATION ON GPU ---")
    print("Loading BGEM3FlagModel on CUDA device...")
    from FlagEmbedding import BGEM3FlagModel
    
    # We enforce 'cuda' because this script is designed for Colab GPU runtimes
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="cuda")
    print("Model loaded successfully on GPU!")
    
    results = {}
    
    # Process Quran
    quran_chunks = load_jsonl(quran_path)
    results["quran"] = embed_and_store_unified("quran", quran_chunks, model, batch_size=128)
    
    # Process Hadith
    hadith_chunks = load_jsonl(hadith_path)
    results["hadith"] = embed_and_store_unified("hadith", hadith_chunks, model, batch_size=128)
    
    # Process Tafsir
    tafsir_chunks = load_jsonl(tafsir_path)
    tafsir_ar = [c for c in tafsir_chunks if c.get("language") == "ar"]
    tafsir_en = [c for c in tafsir_chunks if c.get("language") == "en"]
    results["tafsir_ar"] = embed_and_store_unified("tafsir_ar", tafsir_ar, model, batch_size=128)
    results["tafsir_en"] = embed_and_store_unified("tafsir_en", tafsir_en, model, batch_size=128)
    
    # Zip results
    zip_file = zip_results()
    
    # Trigger download if running in Colab
    if IN_COLAB:
        from google.colab import files
        print("\n--- STEP 4: DOWNLOAD INDEX FILES ---")
        print("Downloading the zipped database to your computer. Please wait...")
        files.download(zip_file)
        print("Download initiated!")
    else:
        print(f"\nDone! Zip archive is located at: {os.path.abspath(zip_file)}")


if __name__ == "__main__":
    main()
