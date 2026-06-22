"""
benchmark_dense.py

A comprehensive test suite to evaluate the DenseRetriever across multiple 
languages (Arabic, French, English) and collections. 
Forces offline mode to prevent Hugging Face network checks.
"""

import os
import sys
import torch

# Force Hugging Face to use ONLY local files (no network checks)
os.environ["HF_HUB_OFFLINE"] = "1"

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FlagEmbedding import BGEM3FlagModel
from src.nur.retriever.dense import DenseRetriever

def run_test(model, retriever, query, source, lang):
    """Executes a single search query and prints the formatted results."""
    print(f"\n{'='*60}")
    print(f"🗣️  Language: {lang} | 📚 Collection: {source}")
    print(f"❓ Query: \"{query}\"")
    print(f"{'='*60}")
    
    # Encode the query
    q_output = model.encode(
        [query],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    
    q_dense = q_output['dense_vecs'][0]
    if hasattr(q_dense, 'cpu'):
        q_dense = q_dense.cpu().numpy()
        
    q_dense_list = q_dense.tolist()
    
    # Search the database
    results = retriever.search(query_vector=q_dense_list, source=source, top_k=3)
    
    # Print Top 3 results
    for i, res in enumerate(results, 1):
        print(f"\nRank {i} | ID: {res['id']} | Similarity: {res['similarity']:.4f}")
        preview = res['document'][:250].replace('\n', ' ')
        print(f"Preview: {preview}...")

def main():
    print("--- INITIALIZING BENCHMARK SUITE ---")
    print("Loading BGE-M3 model on MPS (Offline Mode)...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="mps")
    
    print("Connecting to ChromaDB...")
    retriever = DenseRetriever(db_path="./data/chroma_db")
    
    # --- TEST BATTERY ---
    # 1. English Queries
    run_test(model, retriever, "What does the Quran say about charity and zakat?", "quran", "English")
    run_test(model, retriever, "Patience and reward in trials", "hadith", "English")
    
    # 2. French Queries (The critical ones that failed before)
    run_test(model, retriever, "Quel est le statut de l'usure Riba?", "quran", "French")
    run_test(model, retriever, "L'importance de la recherche de la science", "hadith", "French")
    
    # 3. Arabic Queries
    run_test(model, retriever, "ما حكم صلاة الجماعة", "hadith", "Arabic")
    run_test(model, retriever, "أحاديث عن الصبر عند المصيبة", "hadith", "Arabic")

    print(f"\n{'='*60}")
    print("✅ BENCHMARK COMPLETE")

if __name__ == "__main__":
    main()