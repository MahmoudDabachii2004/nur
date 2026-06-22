"""
test_dense_search.py

This script tests the DenseRetriever module. It loads the BGE-M3 model locally,
encodes an English query about charity, and searches the Quran collection to verify
that the semantic search returns relevant verses.
"""

from __future__ import annotations

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from FlagEmbedding import BGEM3FlagModel
from src.nur.retriever.dense import DenseRetriever


def main() -> None:
    """Loads the BGE-M3 model, initializes DenseRetriever, and searches the Quran collection."""
    print("--- Initializing Local Test ---")

    # 1. Initialize BGE-M3 on Apple Silicon (MPS)
    print("Loading BGE-M3 model on MPS...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="mps")

    # 2. Initialize Dense Retriever
    retriever = DenseRetriever(db_path="./data/chroma_db")

    # 3. Define test query and encode it
    query = "What does the Quran say about charity and zakat?"
    print(f"\nEncoding query: '{query}'")

    q_output = model.encode(
        [query], return_dense=True, return_sparse=False, return_colbert_vecs=False
    )

    q_dense = q_output["dense_vecs"][0].tolist()

    # 4. Search the Quran collection
    print("Searching 'quran_dense' collection...")
    results = retriever.search(query_vector=q_dense, source="quran", top_k=3)

    # 5. Print results
    print("\n--- Top 3 Results ---")
    for i, res in enumerate(results, 1):
        print(f"{i}. ID: {res['id']} | Similarity: {res['similarity']:.4f}")
        # Print first 150 characters of the document
        preview = res["document"][:150].replace("\n", " ")
        print(f"   Preview: {preview}...")

    print("\nTest complete.")


if __name__ == "__main__":
    main()
