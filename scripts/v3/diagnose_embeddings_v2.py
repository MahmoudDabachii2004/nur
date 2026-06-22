"""Diagnostic v2: check chunk lengths and what BGE-M3 actually sees."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import chromadb
import numpy as np
from FlagEmbedding import BGEM3FlagModel

print("=" * 60)
print("V3 Embedding Diagnostic v2 — Chunk length analysis")
print("=" * 60)

# 1. Check chunk lengths in the stored documents
client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma_db_v3"))
quran_col = client.get_collection("quran_v3_dense")

# Get a few chunks to see their lengths
print("\n[1] Chunk document lengths (stored in ChromaDB)")
print("-" * 60)

sample_ids = ["SRC-QURAN-2-255", "SRC-QURAN-2-43", "SRC-QURAN-1-1",
              "SRC-QURAN-112-1", "SRC-QURAN-114-1", "SRC-QURAN-108-2"]

for sid in sample_ids:
    result = quran_col.get(ids=[sid], include=["documents"])
    if result["documents"]:
        doc = result["documents"][0]
        print(f"  {sid}: {len(doc)} chars")
        print(f"    First 200 chars: {doc[:200]!r}")
        print()

# 2. Compare: query with FULL embedding_text vs query with just verse text
print("\n[2] Compare queries: full embedding_text vs verse-only")
print("-" * 60)

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# Get the full embedding_text for 2:255
result = quran_col.get(ids=["SRC-QURAN-2-255"], include=["documents", "metadatas"])
if result["documents"]:
    full_doc = result["documents"][0]
    meta = result["metadatas"][0]
    verse_ar = meta.get("text_ar", "")
    verse_en = meta.get("text_en", "")

    print(f"  Full embedding_text length: {len(full_doc)} chars")
    print(f"  Verse AR length: {len(verse_ar)} chars")
    print(f"  Verse EN length: {len(verse_en)} chars")
    print()

    # Query 1: with full embedding_text (what was used for indexing)
    print(f"  Query 1: FULL embedding_text (should return 2:255 as #1)")
    q_emb = model.encode([full_doc], return_dense=True, return_sparse=False)
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
    for i, (rid, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
        marker = " ← EXPECTED" if rid == "SRC-QURAN-2-255" else ""
        print(f"    #{i+1} {rid} (dist={dist:.4f}){marker}")

    # Query 2: with just the verse AR text
    print(f"\n  Query 2: verse AR only")
    q_emb = model.encode([verse_ar], return_dense=True, return_sparse=False)
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
    for i, (rid, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
        marker = " ← EXPECTED" if rid == "SRC-QURAN-2-255" else ""
        print(f"    #{i+1} {rid} (dist={dist:.4f}){marker}")

    # Query 3: with just the verse EN text
    print(f"\n  Query 3: verse EN only")
    q_emb = model.encode([verse_en], return_dense=True, return_sparse=False)
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
    for i, (rid, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
        marker = " ← EXPECTED" if rid == "SRC-QURAN-2-255" else ""
        print(f"    #{i+1} {rid} (dist={dist:.4f}){marker}")

# 3. Check token counts
print("\n[3] Token count estimation")
print("-" * 60)
if result and result["documents"]:
    full_doc = result["documents"][0]
    # Rough estimate: 1 token ~ 4 chars for EN, ~2 chars for AR
    # BGE-M3 max = 8192 tokens
    en_tokens = len(full_doc) / 4
    ar_tokens = len(full_doc) / 2
    print(f"  Full doc: {len(full_doc)} chars")
    print(f"  Estimated tokens (EN ratio): {en_tokens:.0f}")
    print(f"  Estimated tokens (AR ratio): {ar_tokens:.0f}")
    print(f"  BGE-M3 max tokens: 8192")
    if ar_tokens > 8192:
        print(f"  ⚠️  DOC IS TOO LONG — BGE-M3 truncates at 8192 tokens!")
        print(f"      Truncation ratio: {8192/ar_tokens*100:.1f}% of doc is embedded")
        print(f"      Lost: {100 - 8192/ar_tokens*100:.1f}% of the document")

# 4. Sample chunks from short surahs to compare
print("\n[4] Short surah chunk lengths (the ones being returned)")
print("-" * 60)
short_ids = ["SRC-QURAN-112-1", "SRC-QURAN-112-2", "SRC-QURAN-112-4",
             "SRC-QURAN-108-2", "SRC-QURAN-114-1"]
for sid in short_ids:
    r = quran_col.get(ids=[sid], include=["documents"])
    if r["documents"]:
        doc = r["documents"][0]
        print(f"  {sid}: {len(doc)} chars")

print("\n" + "=" * 60)
print("Diagnostic v2 complete")
print("=" * 60)
