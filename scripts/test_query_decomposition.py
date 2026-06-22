"""
test_query_decomposition.py — Test 3 query decomposition strategies.

The user observed that 6 sub-questions might be REDUNDANT or DEVIATING:
  - "What does the Quran say about charity?" (sub-q 1) = same as raw query
  - "What is the ruling on charity in Islam?" (sub-q 5) = deviates to hadiths/fatwas

This test isolates the impact of sub-questions vs keywords:

  Config 1 (current): raw + ALL sub-questions + ALL keywords = 13 queries
  Config 2 (keywords only): raw + ALL keywords = 7 queries (no sub-questions)
  Config 3 (minimal): raw + first 3 sub-questions + ALL keywords = ~10 queries

If sub-questions dilute the pool, Config 2 should give better precision.

USAGE:
  python3 scripts/test_query_decomposition.py
  python3 scripts/test_query_decomposition.py --query "Is prayer obligatory?"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nur.config import settings
from src.nur.pipeline import NURPipeline


TEST_QUESTIONS = [
    "What does the Quran say about charity and zakat?",
    "Is prayer obligatory?",
    "How to perform wudu (ablution)?",
    "What does the Quran say about patience in trials?",
]


def run_config(
    pipeline: NURPipeline,
    query: str,
    sub_questions: list[str],
    search_keywords: list[str],
    config_name: str,
    include_sub_questions: bool,
    max_sub_questions: int | None,
) -> dict:
    """Run retrieval + reranker with a specific query composition.

    Args:
        include_sub_questions: If True, include sub-questions in the query pool.
        max_sub_questions: If set, limit the number of sub-questions.
    """
    all_queries = [query]

    if include_sub_questions:
        subs = sub_questions[:max_sub_questions] if max_sub_questions else sub_questions
        all_queries = all_queries + subs

    all_queries = all_queries + search_keywords

    label_parts = [f"{len(all_queries)} queries"]
    if include_sub_questions:
        n_subs = len(subs)
        label_parts.append(f"{n_subs} sub-q")
    else:
        label_parts.append("0 sub-q")
    label_parts.append(f"{len(search_keywords)} keywords")
    label = " + ".join(label_parts)

    print(f"\n  Config: {config_name} ({label})")

    start = time.time()

    # Retrieve
    retrieved = pipeline._retrieve(all_queries, top_k=settings.top_k_initial)
    pool_dist = Counter(c["source"] for c in retrieved)

    # Rerank
    chunks_with_docs = pipeline._fetch_chunk_documents(retrieved)
    all_source_refs = pipeline._chunks_to_source_refs(retrieved)

    reranked = pipeline.reranker.rerank(
        query=query,
        chunks=chunks_with_docs,
        source_refs=all_source_refs,
        top_k=settings.top_k_rerank,
        apply_authenticity_weight=True,
        normalize=True,
    )

    elapsed = time.time() - start
    top10_dist = Counter(c["source"] for c in reranked)
    top1_score = reranked[0]["reranker_score"] if reranked else 0.0

    print(f"    Pool: {len(retrieved)} chunks | Top-10: "
          f"Q={top10_dist.get('quran',0)} H={top10_dist.get('hadith',0)} "
          f"TAR={top10_dist.get('tafsir_ar',0)} TEN={top10_dist.get('tafsir_en',0)} "
          f"| Top1={top1_score:.4f} | {elapsed:.1f}s")

    return {
        "config": config_name,
        "label": label,
        "n_queries": len(all_queries),
        "pool_size": len(retrieved),
        "top10_dist": dict(top10_dist),
        "top1_score": top1_score,
        "elapsed_s": elapsed,
        "top_10_ids": [c["id"] for c in reranked],
        "top_10_sources": [c["source"] for c in reranked],
        "top_10_scores": [c["reranker_score"] for c in reranked],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Single query to test. If omitted, tests all 4 questions.",
    )
    args = parser.parse_args()

    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    queries = [args.query] if args.query else TEST_QUESTIONS

    print(f"# NUR Query Decomposition Comparison")
    print(f"# Questions: {len(queries)}")
    print(f"# Configs: current (all sub-q), keywords only (no sub-q), minimal (3 sub-q)")

    print(f"\n# Initializing pipeline...", flush=True)
    pipeline = NURPipeline()
    print(f"# Pipeline ready. BGE-M3 device: {pipeline._device}")

    all_results = {}

    for query in queries:
        print(f"\n{'='*70}")
        print(f"  Query: '{query}'")
        print(f"{'='*70}")

        # Step 1: Architect (run ONCE per question)
        sub_questions, search_keywords = pipeline.generator.decompose_query(query)
        print(f"  Sub-questions ({len(sub_questions)}):")
        for i, sq in enumerate(sub_questions, 1):
            print(f"    {i}. {sq}")
        print(f"  Keywords ({len(search_keywords)}): {', '.join(search_keywords)}")

        results = []

        # Config 1: raw + ALL sub-questions + ALL keywords
        r1 = run_config(
            pipeline, query, sub_questions, search_keywords,
            "Config 1 (current)", include_sub_questions=True, max_sub_questions=None,
        )
        results.append(r1)

        # Config 2: raw + ALL keywords (NO sub-questions)
        r2 = run_config(
            pipeline, query, sub_questions, search_keywords,
            "Config 2 (keywords only)", include_sub_questions=False, max_sub_questions=None,
        )
        results.append(r2)

        # Config 3: raw + first 3 sub-questions + ALL keywords
        r3 = run_config(
            pipeline, query, sub_questions, search_keywords,
            "Config 3 (minimal 3 sub-q)", include_sub_questions=True, max_sub_questions=3,
        )
        results.append(r3)

        all_results[query] = results

    # Print comparative table
    print(f"\n\n{'='*90}")
    print(f"  COMPARATIVE QUERY DECOMPOSITION TABLE")
    print(f"{'='*90}")
    for query, results in all_results.items():
        print(f"\n  Query: '{query[:60]}'")
        print(f"  {'Config':<35} {'Queries':<10} {'Q':<3} {'H':<3} {'TAR':<4} {'TEN':<4} {'Top1':<7} {'Time'}")
        print(f"  {'-'*80}")
        for r in results:
            d = r["top10_dist"]
            print(f"  {r['config']:<35} {r['n_queries']:<10} {d.get('quran',0):<3} "
                  f"{d.get('hadith',0):<3} {d.get('tafsir_ar',0):<4} {d.get('tafsir_en',0):<4} "
                  f"{r['top1_score']:.4f}  {r['elapsed_s']:.0f}s")

    # Print detailed top-10 for each
    for query, results in all_results.items():
        for r in results:
            print(f"\n  {r['config']} — '{query[:50]}'")
            for i, (cid, src, score) in enumerate(
                zip(r["top_10_ids"], r["top_10_sources"], r["top_10_scores"]), 1
            ):
                print(f"    {i:2d}. {cid:30s} ({src:10s}) score={score:.4f}")

    print(f"\n# Done.")


if __name__ == "__main__":
    main()
