"""
test_sparse_search.py

This script tests the SparseRetriever module WITHOUT requiring the BGE-M3 model.
Instead of encoding a live query string (which needs the 2.3GB model download),
it pulls an existing chunk's sparse vector from the JSON index and uses it as a
synthetic query. This verifies the retriever's math in isolation.

The fundamental property being tested: SELF-SIMILARITY. When a chunk's own sparse
vector is used as the query, that chunk MUST rank #1 — because the dot product of
a vector with itself equals the sum of its squared weights, which is the maximum
possible score any chunk can achieve against that query.

The live BGE-M3 query encoding (with a real user question string) is tested
separately in benchmark_sparse.py, which mirrors benchmark_dense.py and is meant
to be run on the user's Mac where the BGE-M3 model is already cached.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add the src directory to the Python path so we can import the nur package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nur.retriever.sparse import SparseRetriever


def load_chunk_vector(sparse_path: Path, source: str, chunk_id: str) -> dict[int, float]:
    """Load a single chunk's sparse vector from the JSON index.

    Args:
        sparse_path: Directory containing {source}_sparse.json files.
        source: One of 'quran', 'hadith', 'tafsir_ar', 'tafsir_en'.
        chunk_id: The chunk ID to load (e.g. 'quran_1_1').

    Returns:
        The chunk's sparse vector as {token_id: weight} dict, ready to be
        passed to SparseRetriever.search() as a query.
    """
    file_path = sparse_path / f"{source}_sparse.json"
    with file_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    chunk = raw[chunk_id]
    # JSON stores indices as a list of ints and values as a list of floats.
    # Reconstruct the {token_id: weight} dict.
    return {int(t): float(v) for t, v in zip(chunk["indices"], chunk["values"])}


def run_self_similarity_test(
    retriever: SparseRetriever,
    sparse_path: Path,
    source: str,
    chunk_id: str,
) -> bool:
    """Verify that a chunk ranks #1 when its own vector is the query.

    Args:
        retriever: The SparseRetriever instance.
        sparse_path: Directory containing the sparse JSON files.
        source: Collection name ('quran', 'hadith', etc.).
        chunk_id: The chunk to test.

    Returns:
        True if the chunk is rank #1, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"📚 Source: {source} | 🧪 Test chunk: {chunk_id}")
    print(f"{'='*60}")

    query_vector = load_chunk_vector(sparse_path, source, chunk_id)
    print(f"Query vector: {len(query_vector)} non-zero tokens (from {chunk_id})")

    results = retriever.search(query_sparse=query_vector, source=source, top_k=5)

    if not results:
        print("❌ FAIL: No results returned.")
        return False

    print(f"\nTop 5 results:")
    for i, res in enumerate(results, 1):
        marker = " ✅ SELF" if res["id"] == chunk_id else ""
        print(f"  Rank {i}: {res['id']:30s}  score={res['score']:.6f}{marker}")

    if results[0]["id"] != chunk_id:
        print(f"\n❌ FAIL: {chunk_id} is NOT rank 1. Got {results[0]['id']}.")
        return False

    print(f"\n✅ PASS: {chunk_id} is rank 1 for its own vector.")
    return True


def main() -> None:
    """Run the self-similarity test suite across multiple collections."""
    print("--- SparseRetriever Test Suite (model-free) ---")

    sparse_path = Path("./data/sparse")
    retriever = SparseRetriever(sparse_path=str(sparse_path))

    # Test cases: (source, chunk_id)
    # quran_1_1   = Al-Fatihah 1 (Bismillah)
    # quran_2_255 = Ayat al-Kursi (Throne Verse)
    # hadith_tirmidhi_1 = First hadith in Tirmidhi (purification/salat)
    test_cases = [
        ("quran", "quran_1_1"),
        ("quran", "quran_2_255"),
        ("hadith", "hadith_tirmidhi_1"),
    ]

    all_passed = True
    for source, chunk_id in test_cases:
        if not run_self_similarity_test(retriever, sparse_path, source, chunk_id):
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED — SparseRetriever math is correct.")
    else:
        print("❌ SOME TESTS FAILED — see output above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
