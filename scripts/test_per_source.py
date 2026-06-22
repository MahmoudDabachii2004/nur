"""
test_per_source.py — Test per-source retrieval strategies.

Tests 4 approaches to building the reranker pool:
  1. 20 per source (equal, 80 total)
  2. 100 per source (equal, 400 total)
  3. 20 total with ratio proportional to collection size
  4. 100 total with ratio proportional to collection size

The user's idea: instead of retrieving 400 chunks from all 52,446 mixed together
(where hadiths dominate because they're 33,738 of 52,446), retrieve SEPARATELY
from each collection. This guarantees each source type has fair representation
in the reranker pool.

Collection sizes:
  - Quran:      6,236 chunks
  - Hadith:    33,738 chunks
  - Tafsir AR:  6,236 chunks
  - Tafsir EN:  6,236 chunks
  - Total:    52,446 chunks

Ratio (proportional to collection size):
  - Quran:      11.9%
  - Hadith:     64.3%
  - Tafsir AR:  11.9%
  - Tafsir EN:  11.9%

USAGE:
  python3 scripts/test_per_source.py
  python3 scripts/test_per_source.py --query "What does the Quran say about charity?"
  python3 scripts/test_per_source.py --mmr-lambda 0.7
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


# ============================================================
# Collection sizes for ratio calculation
# ============================================================

COLLECTION_SIZES = {
    "quran": 6236,
    "hadith": 33738,
    "tafsir_ar": 6236,
    "tafsir_en": 6236,
}
TOTAL_CHUNKS = sum(COLLECTION_SIZES.values())  # 52,446


def compute_ratio_allocation(total_pool: int) -> dict[str, int]:
    """Compute per-source allocation proportional to collection size.

    Args:
        total_pool: Total number of chunks to distribute.

    Returns:
        Dict mapping source name to number of chunks to retrieve from it.
    """
    allocation = {}
    remaining = total_pool
    sources_sorted = sorted(COLLECTION_SIZES.keys(), key=lambda s: -COLLECTION_SIZES[s])

    for i, source in enumerate(sources_sorted):
        if i == len(sources_sorted) - 1:
            # Last source gets the remainder to avoid rounding gaps
            allocation[source] = remaining
        else:
            share = round(total_pool * COLLECTION_SIZES[source] / TOTAL_CHUNKS)
            allocation[source] = share
            remaining -= share

    return allocation


def retrieve_per_source(
    pipeline: NURPipeline,
    all_queries: list[str],
    per_source_k: dict[str, int],
) -> list[dict]:
    """Retrieve chunks separately from each source with a specific top-K per source.

    This is the user's idea: instead of retrieving from all sources mixed together,
    retrieve separately from each collection. This guarantees each source type has
    fair representation in the pool.

    Args:
        pipeline: The NURPipeline instance.
        all_queries: List of query strings (raw + sub-questions + keywords).
        per_source_k: Dict mapping source name to top-K for that source.

    Returns:
        A deduplicated pool of chunks, each with: id, rrf_score, source,
        dense_rank, sparse_rank.
    """
    pool: dict[str, dict] = {}

    for source, k in per_source_k.items():
        for query_idx, query in enumerate(all_queries):
            dense_vec, sparse_vec = pipeline._encode_query(query)
            query_label = "raw" if query_idx == 0 else f"sub{query_idx}"

            dense_results = pipeline.dense_retriever.search(
                query_vector=dense_vec, source=source, top_k=k
            )
            sparse_results = pipeline.sparse_retriever.search(
                query_sparse=sparse_vec, source=source, top_k=k
            )
            fused = pipeline.fuser.fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=k,
            )

            for item in fused:
                cid = item["id"]
                if cid not in pool or item["rrf_score"] > pool[cid]["rrf_score"]:
                    pool[cid] = {
                        "id": cid,
                        "rrf_score": item["rrf_score"],
                        "source": source,
                        "dense_rank": item["dense_rank"],
                        "sparse_rank": item["sparse_rank"],
                    }

    return sorted(pool.values(), key=lambda x: -x["rrf_score"])


def run_test(
    pipeline: NURPipeline,
    query: str,
    sub_questions: list[str],
    search_keywords: list[str],
    per_source_k: dict[str, int],
    label: str,
    mmr_lambda: float = 0.7,
    use_reranker: bool = True,
) -> dict:
    """Run a single per-source retrieval test.

    Args:
        use_reranker: If True, score with cross-encoder + MMR. If False,
                      take top-10 by RRF score only (instant, no model).
    """
    all_queries = [query] + sub_questions + search_keywords
    total_pool = sum(per_source_k.values())

    mode = "WITH reranker + MMR" if use_reranker else "WITHOUT reranker (RRF only)"
    print(f"\n{'='*70}")
    print(f"  Test: {label} [{mode}]")
    print(f"  Per-source K: {per_source_k}")
    print(f"  Total pool: {total_pool}")
    print(f"{'='*70}")

    start = time.time()

    # Step 2: Retrieve per source
    retrieved = retrieve_per_source(pipeline, all_queries, per_source_k)
    print(f"  Retrieved {len(retrieved)} unique chunks")

    if use_reranker:
        # Step 3: Rerank with MMR
        chunks_with_docs = pipeline._fetch_chunk_documents(retrieved)
        all_source_refs = pipeline._chunks_to_source_refs(retrieved)

        reranked = pipeline.reranker.rerank(
            query=query,
            chunks=chunks_with_docs,
            source_refs=all_source_refs,
            top_k=settings.top_k_rerank,
            apply_authenticity_weight=True,
            normalize=True,
            mmr_lambda=mmr_lambda,
        )
    else:
        # No reranker — just take top-10 by RRF score
        reranked = []
        for c in retrieved[:10]:
            reranked.append({
                "id": c["id"],
                "source": c["source"],
                "reranker_score": 0.0,  # no reranker score
                "final_score": c["rrf_score"],
                "rrf_score": c["rrf_score"],
            })

    elapsed = time.time() - start

    # Analyze
    pool_dist = Counter(c["source"] for c in retrieved)
    top10_dist = Counter(c["source"] for c in reranked)

    print(f"  Pool source distribution:")
    for source in ["quran", "hadith", "tafsir_ar", "tafsir_en"]:
        print(f"    {source:12s}: {pool_dist.get(source, 0)}")
    print(f"  Top-10 source distribution:")
    for source in ["quran", "hadith", "tafsir_ar", "tafsir_en"]:
        print(f"    {source:12s}: {top10_dist.get(source, 0)}")
    print(f"  Top-10 chunks:")
    for i, c in enumerate(reranked, 1):
        score = c.get("reranker_score", 0.0)
        final = c.get("final_score", 0.0)
        if use_reranker:
            print(f"    {i:2d}. {c['id']:30s} ({c['source']:10s}) reranker={score:.4f} final={final:.4f}")
        else:
            print(f"    {i:2d}. {c['id']:30s} ({c['source']:10s}) rrf={final:.6f}")

    return {
        "label": f"{label} [{'reranker' if use_reranker else 'RRF only'}]",
        "per_source_k": per_source_k,
        "use_reranker": use_reranker,
        "pool_size": len(retrieved),
        "pool_dist": dict(pool_dist),
        "top10_dist": dict(top10_dist),
        "top_10_ids": [c["id"] for c in reranked],
        "top_10_sources": [c["source"] for c in reranked],
        "top_10_scores": [c.get("reranker_score", 0.0) for c in reranked],
        "elapsed_s": elapsed,
    }


def main() -> None:
    """Run the per-source retrieval comparison test."""
    parser = argparse.ArgumentParser(description="Test per-source retrieval strategies.")
    parser.add_argument(
        "--query", "-q",
        default="What does the Quran say about charity and zakat?",
        help="The query to test.",
    )
    parser.add_argument(
        "--mmr-lambda",
        type=float,
        default=0.7,
        help="MMR lambda for diversity (default: 0.7).",
    )
    args = parser.parse_args()

    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    print(f"# NUR Per-Source Retrieval Comparison Test")
    print(f"# Query: {args.query}")
    print(f"# MMR lambda: {args.mmr_lambda}")
    print(f"# Collection sizes: {COLLECTION_SIZES}")
    print(f"# Total chunks: {TOTAL_CHUNKS}")

    # Compute ratio allocations
    ratio_20 = compute_ratio_allocation(20)
    ratio_100 = compute_ratio_allocation(100)

    print(f"# Ratio for 20: {ratio_20}")
    print(f"# Ratio for 100: {ratio_100}")

    # Initialize pipeline ONCE
    print(f"\n# Initializing pipeline...", flush=True)
    pipeline = NURPipeline()
    print(f"# Pipeline ready. BGE-M3 device: {pipeline._device}")

    # Step 1: Architect (run ONCE, reuse for all tests)
    print(f"\n# Step 1: Architect...", flush=True)
    sub_questions, search_keywords = pipeline.generator.decompose_query(args.query)
    print(f"# Sub-questions: {len(sub_questions)}, Keywords: {len(search_keywords)}")

    # Define 4 test configs
    configs = [
        (
            {"quran": 20, "hadith": 20, "tafsir_ar": 20, "tafsir_en": 20},
            "20 per source (equal, 80 total)",
        ),
        (
            {"quran": 100, "hadith": 100, "tafsir_ar": 100, "tafsir_en": 100},
            "100 per source (equal, 400 total)",
        ),
        (
            ratio_20,
            f"20 total with ratio ({ratio_20})",
        ),
        (
            ratio_100,
            f"100 total with ratio ({ratio_100})",
        ),
    ]

    # Run 8 tests: 4 configs × with/without reranker
    results = []
    for per_source_k, label in configs:
        # WITHOUT reranker first (instant — just RRF top-10)
        result_no_reranker = run_test(
            pipeline=pipeline,
            query=args.query,
            sub_questions=sub_questions,
            search_keywords=search_keywords,
            per_source_k=per_source_k,
            label=label,
            mmr_lambda=args.mmr_lambda,
            use_reranker=False,
        )
        results.append(result_no_reranker)

        # WITH reranker + MMR (~1 min)
        result_reranker = run_test(
            pipeline=pipeline,
            query=args.query,
            sub_questions=sub_questions,
            search_keywords=search_keywords,
            per_source_k=per_source_k,
            label=label,
            mmr_lambda=args.mmr_lambda,
            use_reranker=True,
        )
        results.append(result_reranker)

    # Print comparative table
    print(f"\n\n{'='*90}")
    print(f"  COMPARATIVE PER-SOURCE RETRIEVAL TABLE")
    print(f"  (MMR λ={args.mmr_lambda})")
    print(f"{'='*90}")
    print(f"  {'Config':<45} {'Pool':<6} {'Quran':<7} {'Hadith':<7} {'T_AR':<5} {'T_EN':<5} {'Time'}")
    print(f"  {'-'*85}")
    for r in results:
        d = r["top10_dist"]
        print(f"  {r['label']:<45} {r['pool_size']:<6} {d.get('quran',0):<7} {d.get('hadith',0):<7} "
              f"{d.get('tafsir_ar',0):<5} {d.get('tafsir_en',0):<5} {r['elapsed_s']:.1f}s")
    print(f"{'='*90}")

    # Print detailed top-10 for each
    for r in results:
        print(f"\n  {r['label']}:")
        for i, (cid, src, score) in enumerate(
            zip(r["top_10_ids"], r["top_10_sources"], r["top_10_scores"]), 1
        ):
            print(f"    {i:2d}. {cid:30s} ({src:10s}) score={score:.4f}")

    print(f"\n# Done.")


if __name__ == "__main__":
    main()
