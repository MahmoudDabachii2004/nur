"""
NUR V3 — Step 8: Verify pipeline against 5 ground-truth examples

Loads the 5 examples from docs/v3/08_EXAMPLES.md and verifies that the
V3 retrieval pipeline returns the expected verses/hadiths for each.

This is a SMOKE TEST — not a full benchmark. It verifies:
  1. ChromaDB collections exist and have correct counts
  2. Dense retrieval works for FR/EN/AR queries
  3. The expected verses are in top-10 for each example
  4. Cross-refs are populated for some chunks

Usage (local, after downloading the indexed DB from Lightning AI):
  GROQ_API_KEY=... python scripts/v3/08_verify_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import DATA_DIR  # noqa: E402

CHROMA_DIR = DATA_DIR / "chroma_db_v3"

# 5 ground-truth examples from docs/v3/08_EXAMPLES.md
GROUND_TRUTH = [
    {
        "case": 1,
        "language": "fr",
        "question": "Pourquoi la prière est obligatoire ?",
        "expected_quran_ids": ["SRC-QURAN-2-43", "SRC-QURAN-2-3", "SRC-QURAN-4-103",
                               "SRC-QURAN-20-14", "SRC-QURAN-24-56"],
        "expected_phase_a": "STRONG",
    },
    {
        "case": 2,
        "language": "ar",
        "question": "معلومات عن كرسي الله",
        "expected_quran_ids": ["SRC-QURAN-2-255"],
        "expected_phase_a": "STRONG",
    },
    {
        "case": 3,
        "language": "fr",
        "question": "Est-ce que fumer est haram ?",
        "expected_quran_ids": ["SRC-QURAN-2-195", "SRC-QURAN-4-29", "SRC-QURAN-17-27",
                               "SRC-QURAN-5-90", "SRC-QURAN-7-157"],
        "expected_phase_a": "STRONG",
    },
    {
        "case": 4,
        "language": "fr",
        "question": "Le wudu peut-il être fait avec l'eau du puits ?",
        "expected_quran_ids": ["SRC-QURAN-5-6", "SRC-QURAN-2-222"],
        "expected_phase_a": "STRONG",
    },
    {
        "case": 5,
        "language": "fr",
        "question": "L'IA a-t-elle une âme ?",
        "expected_quran_ids": ["SRC-QURAN-17-85", "SRC-QURAN-15-29"],
        "expected_phase_a": "WEAK",
    },
]


def verify_collections() -> bool:
    """Verify ChromaDB collections exist with correct counts."""
    print("\n[1/3] Verifying ChromaDB collections...")
    try:
        import chromadb
    except ImportError:
        print("  [FAIL] chromadb not installed. Run: pip install chromadb")
        return False

    if not CHROMA_DIR.exists():
        print(f"  [FAIL] {CHROMA_DIR} not found. Download from Lightning AI first.")
        return False

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    expected = {
        "quran_v3_dense": 6236,
        "hadith_v3_dense": 33738,
    }
    all_ok = True
    for col_name, expected_count in expected.items():
        try:
            col = client.get_collection(col_name)
            actual = col.count()
            status = "✓" if actual == expected_count else "⚠️"
            print(f"  {col_name}: {actual:,} / {expected_count:,} {status}")
            if actual != expected_count:
                all_ok = False
        except Exception as e:
            print(f"  {col_name}: NOT FOUND ({e})")
            all_ok = False
    return all_ok


def verify_retrieval() -> bool:
    """Verify dense retrieval returns expected verses for each example."""
    print("\n[2/3] Verifying retrieval for 5 ground-truth examples...")
    try:
        import chromadb
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as e:
        print(f"  [FAIL] Missing dep: {e}")
        return False

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    quran_col = client.get_collection("quran_v3_dense")

    print("  Loading BGE-M3...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("  BGE-M3 loaded")

    all_ok = True
    for ex in GROUND_TRUTH:
        print(f"\n  --- Case {ex['case']} ({ex['language']}): {ex['question'][:50]}... ---")
        # Encode query
        q_emb = model.encode([ex["question"]], return_dense=True, return_sparse=False)
        q_dense = q_emb["dense_vecs"]
        if hasattr(q_dense, "cpu"):
            q_dense = q_dense.cpu().numpy()
        import numpy as np
        norms = np.linalg.norm(q_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q_dense = q_dense / norms

        # Query top 10
        results = quran_col.query(
            query_embeddings=q_dense.tolist(),
            n_results=10,
            include=["metadatas", "distances"],
        )

        retrieved_ids = results["ids"][0]
        retrieved_distances = results["distances"][0]

        # Show top-5
        for i, (rid, dist) in enumerate(zip(retrieved_ids[:5], retrieved_distances[:5])):
            print(f"    #{i+1} {rid} (distance={dist:.3f})")

        # Check if any expected is in top-10
        expected_set = set(ex["expected_quran_ids"])
        retrieved_set = set(retrieved_ids[:10])
        hits = expected_set & retrieved_set
        hit_pct = len(hits) / len(expected_set) * 100

        status = "✓" if hit_pct >= 40 else "⚠️"
        print(f"    Expected hits: {len(hits)}/{len(expected_set)} ({hit_pct:.0f}%) {status}")
        if hit_pct < 40:
            all_ok = False

    return all_ok


def verify_cross_refs() -> bool:
    """Verify that some Quran chunks have hadith cross-refs."""
    print("\n[3/3] Verifying cross-refs in quran_v3.jsonl...")
    quran_path = DATA_DIR / "processed" / "quran_v3.jsonl"
    if not quran_path.exists():
        print(f"  [FAIL] {quran_path} not found")
        return False

    total = 0
    with_refs = 0
    sample_refs = []
    with quran_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                total += 1
                refs = obj.get("hadith_cross_refs", {}).get("high_confidence", [])
                if refs:
                    with_refs += 1
                    if len(sample_refs) < 3:
                        sample_refs.append((obj["id"], refs[:3]))
            except json.JSONDecodeError:
                continue

    pct = with_refs / max(total, 1) * 100
    print(f"  Chunks with cross-refs: {with_refs:,} / {total:,} ({pct:.1f}%)")
    if sample_refs:
        print(f"  Samples:")
        for cid, refs in sample_refs:
            print(f"    {cid} → {refs}")

    # We expect at least 5% of chunks to have cross-refs (conservative)
    return pct >= 5.0


def main() -> int:
    print("=" * 60)
    print("NUR V3 — Step 8: Verify pipeline (5 ground-truth examples)")
    print("=" * 60)

    ok1 = verify_collections()
    ok2 = verify_retrieval()
    ok3 = verify_cross_refs()

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Collections exist:    {'✓' if ok1 else '✗'}")
    print(f"  Retrieval works:      {'✓' if ok2 else '✗'}")
    print(f"  Cross-refs populated: {'✓' if ok3 else '✗'}")

    if ok1 and ok2 and ok3:
        print("\n  ✅ V3 PIPELINE VERIFIED")
        return 0
    else:
        print("\n  ⚠️  Some checks failed — review above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
