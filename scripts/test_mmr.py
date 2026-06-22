"""
test_mmr.py — Test Maximum Marginal Relevance (MMR) with 3 lambda values.

Runs the full pipeline 3 times with different MMR lambda values (0.5, 0.7, 0.9)
and compares the source distribution in the top-10 chunks.

MMR balances pertinence and diversity:
  λ=0.9 → 90% pertinence, 10% diversity (close to pure relevance)
  λ=0.7 → 70% pertinence, 30% diversity (recommended balance)
  λ=0.5 → 50% pertinence, 50% diversity (maximum diversity)

The test also includes a baseline (no MMR, pure relevance) for comparison.

USAGE:
  python3 scripts/test_mmr.py
  python3 scripts/test_mmr.py --query "What does the Quran say about charity and zakat?"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nur.pipeline import NURPipeline
from src.nur.config import settings


def run_test(
    pipeline: NURPipeline,
    query: str,
    mmr_lambda: float | None,
    label: str,
) -> dict:
    """Run the pipeline with a specific MMR lambda and return results.

    Args:
        pipeline: The NURPipeline instance (reused across tests).
        query: The user question.
        mmr_lambda: MMR lambda value (0.0-1.0) or None for no MMR.
        label: Human-readable label for this test.

    Returns:
        A dict with: label, mmr_lambda, source_distribution, top_10_ids,
        top_10_sources, elapsed_s, abstained, error.
    """
    print(f"\n{'='*70}")
    print(f"  Test: {label}")
    print(f"  MMR lambda: {mmr_lambda if mmr_lambda is not None else 'None (pure relevance)'}")
    print(f"{'='*70}")

    start = time.time()

    # We need to monkey-patch the pipeline's query() to pass mmr_lambda.
    # Instead of modifying query(), we replicate the relevant steps manually.
    # This avoids loading BGE-M3 and the reranker multiple times.

    # Step 1: Architect (reuse if already done — we pass sub_questions from outside)
    # Actually, we'll call the full query() but intercept the reranker call.
    # The simplest approach: modify the reranker call in pipeline.query() to
    # accept mmr_lambda. But that requires changing the pipeline code.
    #
    # Alternative: we run the pipeline steps manually here, reusing the
    # already-loaded BGE-M3 and reranker from the pipeline instance.

    from src.nur.sources import SourceRef, render_sources_for_prompt

    # Step 1: Architect
    sub_questions, search_keywords = pipeline.generator.decompose_query(query)
    all_queries = [query] + sub_questions + search_keywords
    print(f"  Sub-questions: {len(sub_questions)}, Keywords: {len(search_keywords)}")

    # Step 2: Retrieve
    retrieved = pipeline._retrieve(all_queries, top_k=settings.top_k_initial)
    print(f"  Retrieved {len(retrieved)} chunks")

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

    elapsed = time.time() - start

    # Analyze source distribution
    source_dist = Counter(c["source"] for c in reranked)

    print(f"  Top-10 source distribution:")
    for source, count in sorted(source_dist.items()):
        print(f"    {source:12s}: {count}")
    print(f"  Top-10 chunks:")
    for i, c in enumerate(reranked, 1):
        print(f"    {i:2d}. {c['id']:30s} ({c['source']:10s}) score={c['reranker_score']:.4f} final={c['final_score']:.4f}")

    # Check abstention
    abstained = pipeline.reranker.should_abstain(reranked, threshold=0.35)

    return {
        "label": label,
        "mmr_lambda": mmr_lambda,
        "source_dist": dict(source_dist),
        "top_10_ids": [c["id"] for c in reranked],
        "top_10_sources": [c["source"] for c in reranked],
        "top_10_scores": [c["reranker_score"] for c in reranked],
        "elapsed_s": elapsed,
        "abstained": abstained,
    }


def main() -> None:
    """Run the MMR comparison test."""
    parser = argparse.ArgumentParser(description="Test MMR with 3 lambda values.")
    parser.add_argument(
        "--query", "-q",
        default="What does the Quran say about charity and zakat?",
        help="The query to test.",
    )
    args = parser.parse_args()

    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    print(f"# NUR MMR Lambda Comparison Test")
    print(f"# Query: {args.query}")
    print(f"# Testing: no MMR (baseline), λ=0.5, λ=0.7, λ=0.9")

    # Initialize pipeline ONCE
    print(f"\n# Initializing pipeline...", flush=True)
    pipeline = NURPipeline()
    print(f"# Pipeline ready. BGE-M3 device: {pipeline._device}")

    # Run 4 tests: baseline + 3 lambda values
    tests = [
        (None, "No MMR (pure relevance)"),
        (0.5, "MMR λ=0.5 (max diversity)"),
        (0.7, "MMR λ=0.7 (balanced)"),
        (0.9, "MMR λ=0.9 (high relevance)"),
    ]

    results = []
    for mmr_lambda, label in tests:
        result = run_test(pipeline, args.query, mmr_lambda, label)
        results.append(result)

    # Print comparative table
    print(f"\n\n{'='*70}")
    print(f"  COMPARATIVE MMR TABLE")
    print(f"{'='*70}")
    print(f"  {'Config':<30} {'Quran':<7} {'Hadith':<7} {'Tafsir':<7} {'AR':<7} {'EN':<7} {'Time'}")
    print(f"  {'-'*70}")
    for r in results:
        dist = r["source_dist"]
        quran = dist.get("quran", 0)
        hadith = dist.get("hadith", 0)
        tafsir_ar = dist.get("tafsir_ar", 0)
        tafsir_en = dist.get("tafsir_en", 0)
        print(f"  {r['label']:<30} {quran:<7} {hadith:<7} {tafsir_ar:<7} {tafsir_en:<7} {r['elapsed_s']:.1f}s")
    print(f"{'='*70}")

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
