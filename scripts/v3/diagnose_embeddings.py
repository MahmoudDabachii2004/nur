"""Diagnostic: check if V3 embeddings are working correctly."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import chromadb
import numpy as np
from FlagEmbedding import BGEM3FlagModel

print("=" * 60)
print("V3 Embedding Diagnostic")
print("=" * 60)

# 1. Check collection metadata (distance metric)
client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma_db_v3"))
quran_col = client.get_collection("quran_v3_dense")
print(f"\n[1] Collection metadata: {quran_col.metadata}")

# 2. Load BGE-M3
print("\n[2] Loading BGE-M3...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("    Loaded")

# 3. Test 1: Query with EXACT verse text — should return itself as #1
print("\n[3] Test 1: Self-retrieval (query = exact verse text)")
print("-" * 60)

test_cases = [
    ("SRC-QURAN-2-255", "Ayat al-Kursi"),
    ("SRC-QURAN-2-43", "Establish prayer"),
    ("SRC-QURAN-1-1", "Bismillah"),
]

for expected_id, label in test_cases:
    # Get the actual chunk
    chunk_data = quran_col.get(ids=[expected_id], include=["documents", "metadatas"])
    if not chunk_data["ids"]:
        print(f"  [{label}] {expected_id} NOT FOUND in collection!")
        continue

    doc = chunk_data["documents"][0]
    meta = chunk_data["metadatas"][0]
    text_ar = meta.get("text_ar", "")
    text_en = meta.get("text_en", "")

    # Query with the EXACT Arabic text
    q_emb = model.encode([text_ar], return_dense=True, return_sparse=False)
    q_dense = q_emb["dense_vecs"]
    if hasattr(q_dense, "cpu"):
        q_dense = q_dense.cpu().numpy()
    norms = np.linalg.norm(q_dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q_dense = q_dense / norms

    results = quran_col.query(
        query_embeddings=q_dense.tolist(),
        n_results=5,
        include=["distances"],
    )

    retrieved = results["ids"][0]
    distances = results["distances"][0]

    print(f"\n  [{label}] {expected_id}")
    print(f"    Query (AR): {text_ar[:80]}...")
    print(f"    Top-5 results:")
    for i, (rid, dist) in enumerate(zip(retrieved, distances)):
        marker = " ← EXPECTED" if rid == expected_id else ""
        print(f"      #{i+1} {rid} (L2 dist={dist:.4f}){marker}")

    # Check if self is in top-1
    if retrieved[0] == expected_id:
        print(f"    ✅ Self-retrieval PASSED (rank 1)")
    elif expected_id in retrieved:
        rank = retrieved.index(expected_id) + 1
        print(f"    ⚠️ Self-retrieval at rank {rank} (not 1)")
    else:
        print(f"    ❌ Self-retrieval FAILED — verse not in top-5!")

# 4. Test 2: Check embedding norms (should all be ~1.0 if normalized)
print("\n[4] Test 2: Check embedding normalization")
print("-" * 60)

# Get a few embeddings
sample_ids = ["SRC-QURAN-2-255", "SRC-QURAN-114-1", "SRC-QURAN-1-1"]
for sid in sample_ids:
    result = quran_col.get(ids=[sid], include=["embeddings"])
    if result["embeddings"]:
        emb = np.array(result["embeddings"][0])
        norm = np.linalg.norm(emb)
        print(f"  {sid}: norm={norm:.6f} (should be ~1.0)")

# 5. Test 3: Check if distance metric is cosine or L2
print("\n[5] Test 3: Distance metric analysis")
print("-" * 60)

# Get embeddings for 2 verses
ids = ["SRC-QURAN-2-255", "SRC-QURAN-114-1"]
embs = []
for sid in ids:
    result = quran_col.get(ids=[sid], include=["embeddings"])
    if result["embeddings"]:
        embs.append(np.array(result["embeddings"][0]))

if len(embs) == 2:
    # Compute L2 distance manually
    l2_dist = np.linalg.norm(embs[0] - embs[1])
    # Compute cosine similarity manually
    cos_sim = np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]))
    print(f"  Between {ids[0]} and {ids[1]}:")
    print(f"    Manual L2 distance: {l2_dist:.6f}")
    print(f"    Manual cosine similarity: {cos_sim:.6f}")

    # Now query with embedding of 2:255 and see what ChromaDB returns as distance
    q_emb = embs[0].reshape(1, -1).tolist()
    results = quran_col.query(
        query_embeddings=q_emb,
        n_results=3,
        include=["distances"],
    )
    print(f"  ChromaDB query with 2:255 embedding:")
    for rid, dist in zip(results["ids"][0], results["distances"][0]):
        print(f"    {rid}: ChromaDB distance={dist:.6f}")

# 6. Test 4: Query "Ayat al-Kursi" in English
print("\n[6] Test 4: Query 'Ayat al-Kursi throne of Allah'")
print("-" * 60)

queries = [
    "Ayat al-Kursi throne of Allah",
    "كرسي الله",
    "Allah there is no deity except Him the Ever-Living",
    "establish prayer and give zakah",
]

for q in queries:
    q_emb = model.encode([q], return_dense=True, return_sparse=False)
    q_dense = q_emb["dense_vecs"]
    if hasattr(q_dense, "cpu"):
        q_dense = q_dense.cpu().numpy()
    norms = np.linalg.norm(q_dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q_dense = q_dense / norms

    results = quran_col.query(
        query_embeddings=q_dense.tolist(),
        n_results=3,
        include=["distances"],
    )

    print(f"\n  Query: '{q}'")
    for i, (rid, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
        print(f"    #{i+1} {rid} (dist={dist:.4f})")

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)
