"""
test_recall.py — Comparative recall audit for the NUR retriever.

Tests retrieval recall across multiple pool sizes (100, 200, 300, 400, 500)
to find the sweet spot between recall and reranker latency.

The script runs the retriever ONCE per pool size, then checks how many of
the expected key verses/hadiths were found. It prints a comparative table
at the end so you can see the recall progression.

NOTE: This script does NOT run the reranker — it only measures the RETRIEVER
recall (what enters the reranker pool). The reranker's job is to pick the
best 10 from this pool; if a key verse isn't in the pool, the reranker
cannot recover it.

USAGE:
  python3 scripts/test_recall.py
  python3 scripts/test_recall.py --query "Is prayer obligatory"
  python3 scripts/test_recall.py --pool-sizes 100 200 300

OUTPUT:
  Flat list of chunks per pool size (for manual inspection) + a comparative
  recall table at the end.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nur.pipeline import NURPipeline
from src.nur.config import settings


# ============================================================
# Known-correct chunks to check recall against.
# Verified against alquran.cloud search API (DEC-030).
# ============================================================

RECALL_CHECKS: dict[str, dict] = {
    "Is prayer obligatory": {
        "query": "Is prayer obligatory",
        "expected_quran": [
            (2, 10, "Standard 2:3 — 'establish prayer' (foundational)"),
            (2, 50, "Standard 2:43 — 'establish prayer and give zakah'"),
            (2, 90, "Standard 2:83 — covenant of Children of Israel"),
            (2, 117, "Standard 2:110 — 'establish prayer and give zakah'"),
            (2, 284, "Standard 2:277 — 'establish prayer and give zakah'"),
            (4, 570, "Standard 4:77 — 'restrain hands and establish prayer'"),
            (8, 1163, "Standard 8:3 — 'the ones who establish prayer'"),
            (9, 1246, "Standard 9:11 — 'repent, establish prayer, give zakah'"),
            (9, 1253, "Standard 9:18 — 'maintain mosques, establish prayer'"),
            (11, 114, "Standard 11:114 — 'establish prayer at two ends of day'"),
            (4, 596, "Standard 4:103 — 'prayer decreed, remember Allah'"),
        ],
        "expected_hadith": [
            ("bukhari", 8, "Bukhari #8 — Islam built on 5 pillars"),
            ("muslim", 8, "Muslim #8 — 5 pillars hadith"),
            ("bukhari", 1, "Bukhari #1 — actions by intention"),
        ],
    },
}


def check_recall(
    retrieved: list[dict],
    metadata_map: dict[str, dict],
    check: dict,
) -> tuple[int, int, list[str]]:
    """Check how many expected chunks were found in the retrieved pool.

    Returns (found_count, total_count, detail_lines).
    """
    found = 0
    total = 0
    details = []

    # Check Quran verses across all source types (quran + tafsir)
    for surah_num, ayah_num, description in check.get("expected_quran", []):
        total += 1
        match_rank = None
        match_source = None
        for i, chunk in enumerate(retrieved, 1):
            meta = metadata_map.get(chunk["id"], {})
            if (
                meta.get("surah_num") == surah_num
                and meta.get("ayah_num") == ayah_num
            ):
                match_rank = i
                match_source = chunk["source"]
                break
        if match_rank:
            details.append(f"  [FOUND rank {match_rank:3d}] {match_source:10s} {surah_num}:{ayah_num} — {description}")
            found += 1
        else:
            details.append(f"  [MISS       ] quran/tafsir  {surah_num}:{ayah_num} — {description}")

    # Check Hadiths by chunk ID
    details.append("")
    for collection_slug, hadith_number, description in check.get("expected_hadith", []):
        total += 1
        expected_id = f"hadith_{collection_slug}_{hadith_number}"
        match_rank = None
        for i, chunk in enumerate(retrieved, 1):
            if chunk["id"] == expected_id:
                match_rank = i
                break
        if match_rank:
            details.append(f"  [FOUND rank {match_rank:3d}] {expected_id:30s} — {description}")
            found += 1
        else:
            details.append(f"  [MISS       ] {expected_id:30s} — {description}")

    return found, total, details


def run_single_test(
    pipeline: NURPipeline,
    query: str,
    sub_questions: list[str],
    search_keywords: list[str],
    pool_size: int,
    check: dict,
    use_keywords: bool,
) -> dict:
    """Run a single recall test with the given pool size.

    Returns a dict with: pool_size, found, total, recall_pct, elapsed_s, details.
    """
    all_queries = [query] + sub_questions
    if use_keywords:
        all_queries = all_queries + search_keywords

    print(f"\n{'='*70}")
    label = "WITH keywords" if use_keywords else "WITHOUT keywords"
    print(f"  Pool size: {pool_size} | {label}")
    print(f"  Queries: {len(all_queries)} ({'raw + sub-q + keywords' if use_keywords else 'raw + sub-q'})")
    print(f"{'='*70}")

    start = time.time()
    retrieved = pipeline._retrieve(all_queries, top_k=pool_size)
    elapsed = time.time() - start

    print(f"  Retrieved {len(retrieved)} unique chunks in {elapsed:.1f}s")

    # Fetch metadata for all retrieved chunks
    import chromadb
    client = chromadb.PersistentClient(path='./data/chroma_db')
    by_source: dict[str, list[str]] = {}
    for chunk in retrieved:
        by_source.setdefault(chunk["source"], []).append(chunk["id"])
    metadata_map: dict[str, dict] = {}
    for source, chunk_ids in by_source.items():
        try:
            col = client.get_collection(f"{source}_dense")
            res = col.get(ids=chunk_ids, include=["metadatas", "documents"])
            for i, cid in enumerate(res["ids"]):
                meta = res["metadatas"][i] if i < len(res["metadatas"]) else {}
                doc = res["documents"][i] if i < len(res["documents"]) else ""
                metadata_map[cid] = {
                    "preview": doc[:80].replace("\n", " "),
                    "surah_num": meta.get("surah_num"),
                    "ayah_num": meta.get("ayah_num"),
                    "hadith_number": meta.get("hadith_number"),
                    "collection": meta.get("collection"),
                }
        except Exception as e:
            print(f"  WARNING: metadata fetch failed for {source}: {e}", file=sys.stderr)

    # Check recall
    found, total, details = check_recall(retrieved, metadata_map, check)
    recall_pct = found / total * 100 if total > 0 else 0

    print(f"\n  RECALL CHECK:")
    for line in details:
        print(f"  {line}")
    print(f"\n  Recall: {found}/{total} = {recall_pct:.0f}%")

    return {
        "pool_size": pool_size,
        "use_keywords": use_keywords,
        "found": found,
        "total": total,
        "recall_pct": recall_pct,
        "elapsed_s": elapsed,
        "num_chunks": len(retrieved),
        "details": details,
    }


def main() -> None:
    """Run the comparative recall audit."""
    parser = argparse.ArgumentParser(description="NUR retriever recall audit (comparative).")
    parser.add_argument(
        "--query", "-q",
        default="Is prayer obligatory",
        help="The query to test (default: 'Is prayer obligatory').",
    )
    parser.add_argument(
        "--pool-sizes", "-k",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400, 500],
        help="Pool sizes to test (default: 100 200 300 400 500).",
    )
    parser.add_argument(
        "--no-keywords",
        action="store_true",
        help="Also test WITHOUT keywords (to measure keyword impact).",
    )
    args = parser.parse_args()

    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    check = RECALL_CHECKS.get(args.query)
    if not check:
        print(f"ERROR: No recall check defined for '{args.query}'.")
        print(f"Available: {list(RECALL_CHECKS.keys())}")
        sys.exit(1)

    print(f"# NUR Recall Audit (Comparative)")
    print(f"# Query: {args.query}")
    print(f"# Pool sizes: {args.pool_sizes}")
    print(f"# Test keywords impact: {not args.no_keywords is False or True}")  # always test both if --no-keywords

    # Initialize pipeline ONCE (loads BGE-M3 once, reuse for all tests)
    print(f"\n# Initializing pipeline...", flush=True)
    pipeline = NURPipeline()
    print(f"# Pipeline ready. BGE-M3 device: {pipeline._device}")

    # Step 1: Architect (run ONCE, reuse for all pool sizes)
    print(f"\n# Step 1: Architect decomposing query + extracting keywords...", flush=True)
    sub_questions, search_keywords = pipeline.generator.decompose_query(args.query)
    print(f"# Sub-questions ({len(sub_questions)}):")
    for i, sq in enumerate(sub_questions, 1):
        print(f"#   {i}. {sq}")
    print(f"# Search keywords ({len(search_keywords)}):")
    print(f"#   {', '.join(search_keywords)}")

    # Run tests — ALWAYS test both configs (WITH and WITHOUT keywords)
    # so we can measure the keyword impact at every pool size.
    results = []
    configs = [
        (False, "WITHOUT keywords"),
        (True, "WITH keywords"),
    ]

    for pool_size in args.pool_sizes:
        for use_kw, label in configs:
            result = run_single_test(
                pipeline=pipeline,
                query=args.query,
                sub_questions=sub_questions,
                search_keywords=search_keywords,
                pool_size=pool_size,
                check=check,
                use_keywords=use_kw,
            )
            results.append(result)

    # Print comparative table
    print(f"\n\n{'='*70}")
    print(f"  COMPARATIVE RECALL TABLE")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'Pool':<6} {'Found':<7} {'Total':<7} {'Recall':<8} {'Time':<8} {'Chunks'}")
    print(f"  {'-'*70}")
    for r in results:
        label = "WITH keywords" if r["use_keywords"] else "WITHOUT keywords"
        print(f"  {label:<25} {r['pool_size']:<6} {r['found']:<7} {r['total']:<7} "
              f"{r['recall_pct']:>5.0f}%   {r['elapsed_s']:>5.1f}s  {r['num_chunks']}")
    print(f"{'='*70}")

    # Find best config
    best = max(results, key=lambda r: r["recall_pct"])
    print(f"\n  Best: {'WITH keywords' if best['use_keywords'] else 'WITHOUT keywords'} "
          f"pool={best['pool_size']} → {best['recall_pct']:.0f}% recall "
          f"({best['found']}/{best['total']}) in {best['elapsed_s']:.1f}s")

    print(f"\n# Done.")


if __name__ == "__main__":
    main()
