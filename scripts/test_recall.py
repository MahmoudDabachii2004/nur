"""
test_recall.py — Recall audit for the NUR retriever.

Prints the top 100 chunks retrieved by RRF for a given query, as a flat list.
No panels, no colors, no formatting — just the raw data so it's easy to
copy-paste back to the agent for analysis.

USAGE:
  python3 scripts/test_recall.py
  python3 scripts/test_recall.py --query "Is prayer obligatory"
  python3 scripts/test_recall.py --query "..." --top-k 200

OUTPUT FORMAT (one chunk per line):
  rank | chunk_id | source | rrf_score | dense_rank | sparse_rank | text_preview

At the end, prints a RECALL CHECK section showing whether specific key chunks
(known-correct answers) appeared in the results.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nur.pipeline import NURPipeline
from src.nur.config import settings


# ============================================================
# Known-correct chunks to check recall against.
# Add entries here as we discover recall gaps in testing.
# ============================================================

RECALL_CHECKS: dict[str, dict] = {
    "Is prayer obligatory": {
        "query": "Is prayer obligatory",
        "expected": [
            ("quran_2_43", "Quran 2:43 — 'establish prayer' (foundational command)"),
            ("quran_2_3", "Quran 2:3 — believers establish prayer"),
            ("quran_4_103", "Quran 4:103 — 'prayer decreed upon believers'"),
            ("quran_2_110", "Quran 2:110 — establish prayer + give zakat"),
            ("quran_2_238", "Quran 2:238 — 'guard prayers'"),
            ("hadith_bukhari_8", "Bukhari #8 — Islam built on 5 pillars"),
            ("hadith_muslim_8", "Muslim #8 — 5 pillars hadith"),
            ("hadith_bukhari_1", "Bukhari #1 — actions by intention"),
        ],
    },
}


def main() -> None:
    """Run the recall audit and print results as a flat list."""
    parser = argparse.ArgumentParser(description="NUR retriever recall audit.")
    parser.add_argument(
        "--query", "-q",
        default="Is prayer obligatory",
        help="The query to test (default: 'Is prayer obligatory').",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=100,
        help="Number of chunks to retrieve (default: 100).",
    )
    args = parser.parse_args()

    # Pre-flight check
    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY not set (needed for the Architect to decompose).")
        sys.exit(1)

    print(f"# NUR Recall Audit")
    print(f"# Query: {args.query}")
    print(f"# Top-K: {args.top_k}")
    print(f"# Models: Architect={settings.llm_architect}, BGE-M3 on auto-detected device")
    print()

    # Initialize pipeline (loads BGE-M3 on first query)
    print("# Initializing pipeline...", flush=True)
    pipeline = NURPipeline()
    print(f"# Pipeline ready. BGE-M3 device: {pipeline._device}")
    print()

    # Step 1: Architect decomposes the query
    print("# Step 1: Architect decomposing query...", flush=True)
    sub_questions = pipeline.generator.decompose_query(args.query)
    all_queries = [args.query] + sub_questions
    print(f"# Sub-questions ({len(sub_questions)}):")
    for i, sq in enumerate(sub_questions, 1):
        print(f"#   {i}. {sq}")
    print()

    # Step 2: Retrieve top-K chunks per (query, source) — NO truncation to 30
    # We patch the _retrieve call to use top_k=args.top_k instead of the default 30.
    print(f"# Step 2: Retrieving top-{args.top_k} per (query, source)...", flush=True)
    retrieved = pipeline._retrieve(all_queries, top_k=args.top_k)
    print(f"# Total unique chunks retrieved: {len(retrieved)}")
    print()

    # Print the flat list — one chunk per line, easy to copy-paste
    print(f"# ===== TOP {len(retrieved)} CHUNKS (ranked by RRF score) =====")
    print(f"# rank | chunk_id | source | rrf_score | dense_rank | sparse_rank | text_preview")
    print()

    # Build a lookup of chunk metadata for previews (batch fetch)
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.chroma_path if hasattr(settings, 'chroma_path') else './data/chroma_db'))
    
    # Group chunk IDs by source for batch fetching
    by_source: dict[str, list[str]] = {}
    for chunk in retrieved:
        by_source.setdefault(chunk["source"], []).append(chunk["id"])
    
    # Fetch metadata for previews
    metadata_map: dict[str, str] = {}
    for source, chunk_ids in by_source.items():
        try:
            col = client.get_collection(f"{source}_dense")
            res = col.get(ids=chunk_ids, include=["documents"])
            for i, cid in enumerate(res["ids"]):
                doc = res["documents"][i] if i < len(res["documents"]) else ""
                # Extract just the first 80 chars of the actual text (after the context prefix)
                metadata_map[cid] = doc[:80].replace("\n", " ").replace("\r", " ")
        except Exception as e:
            print(f"# WARNING: could not fetch metadata for {source}: {e}", file=sys.stderr)

    for rank, chunk in enumerate(retrieved, 1):
        cid = chunk["id"]
        source = chunk["source"]
        rrf = chunk["rrf_score"]
        d_rank = str(chunk["dense_rank"]) if chunk["dense_rank"] else "-"
        s_rank = str(chunk["sparse_rank"]) if chunk["sparse_rank"] else "-"
        preview = metadata_map.get(cid, "(no preview)")
        print(f"{rank:3d} | {cid:30s} | {source:10s} | {rrf:.6f} | {d_rank:>9s} | {s_rank:>10s} | {preview}")

    # Step 3: Recall check — did the key chunks appear?
    print()
    print("# ===== RECALL CHECK =====")
    check = RECALL_CHECKS.get(args.query)
    if check:
        print(f"# Expected chunks for '{args.query}':")
        retrieved_ids = {c["id"] for c in retrieved}
        found = 0
        total = 0
        for chunk_id, description in check["expected"]:
            total += 1
            if chunk_id in retrieved_ids:
                rank = next(i for i, c in enumerate(retrieved, 1) if c["id"] == chunk_id)
                print(f"#   [FOUND rank {rank:3d}] {chunk_id:30s} — {description}")
                found += 1
            else:
                print(f"#   [MISS       ] {chunk_id:30s} — {description}")
        print()
        print(f"# Recall: {found}/{total} = {found/total*100:.0f}%")
        if found < total:
            print("# ⚠️  Some key chunks are missing — see MISS entries above.")
            print("#     Consider: improving Architect prompt, increasing --top-k, or adding query expansion.")
        else:
            print("# ✅ All key chunks found — recall is good for this query.")
    else:
        print(f"# No recall check defined for '{args.query}'.")
        print("# Add an entry to RECALL_CHECKS in this script to track expected chunks.")

    print()
    print("# Done.")


if __name__ == "__main__":
    main()
