"""
test_multi_query.py — Validate the new Phase 3 architecture across 5 diverse questions.

Tests the full pipeline (per-source retrieval + reranker + MMR + Quran weight)
on 5 different question types to verify the architecture works universally:

  1. "What does the Quran say about charity and zakat?"  → Quran-focused
  2. "Is prayer obligatory?"                              → Conceptual/broad
  3. "What is the ruling on usury (Riba)?"                → Ruling (halal/haram)
  4. "How to perform wudu (ablution)?"                    → Practical (hadith-focused)
  5. "What does the Quran say about patience in trials?"  → Conceptual (Quran-focused)

For each question, we measure:
  - Source distribution in top-10 (Quran/Hadith/Tafsir)
  - Top-1 reranker score (quality indicator)
  - Whether the pipeline abstained (score < 0.35)
  - Time taken

USAGE:
  python3 scripts/test_multi_query.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nur.config import settings
from src.nur.pipeline import NURPipeline


# ============================================================
# Test questions — diverse types to validate the architecture
# ============================================================

TEST_QUESTIONS = [
    {
        "query": "What does the Quran say about charity and zakat?",
        "type": "Quran-focused (explicit 'Quran' in query)",
        "expect": "Quran should be prominent",
    },
    {
        "query": "Is prayer obligatory?",
        "type": "Conceptual/broad",
        "expect": "Mix of Quran ('establish prayer') + Hadith (details)",
    },
    {
        "query": "What is the ruling on usury (Riba)?",
        "type": "Ruling (halal/haram)",
        "expect": "Quran (Riba verses) + Hadith (Riba hadiths)",
    },
    {
        "query": "How to perform wudu (ablution)?",
        "type": "Practical (how-to)",
        "expect": "Hadith should dominate (practical details)",
    },
    {
        "query": "What does the Quran say about patience in trials?",
        "type": "Conceptual (Quran-focused)",
        "expect": "Quran should be prominent",
    },
]


def run_single_query(
    pipeline: NURPipeline,
    query: str,
    question_type: str,
    expect: str,
) -> dict:
    """Run the full pipeline on a single query and return metrics.

    Uses the new architecture: per-source retrieval (20 per source) +
    reranker + MMR λ=0.7 + Quran weight 1.3.
    """
    print(f"\n{'='*70}")
    print(f"  Query: '{query}'")
    print(f"  Type: {question_type}")
    print(f"  Expected: {expect}")
    print(f"{'='*70}")

    start = time.time()

    # Step 1: Architect
    sub_questions, search_keywords = pipeline.generator.decompose_query(query)
    all_queries = [query] + sub_questions + search_keywords
    print(f"  Sub-questions: {len(sub_questions)}, Keywords: {len(search_keywords)}")

    # Step 2: Per-source retrieval (20 per source — the user's idea)
    per_source_k = {"quran": 20, "hadith": 20, "tafsir_ar": 20, "tafsir_en": 20}
    # Use the per-source retrieval function from test_per_source.py
    from scripts.test_per_source import retrieve_per_source
    retrieved = retrieve_per_source(pipeline, all_queries, per_source_k)
    print(f"  Retrieved {len(retrieved)} unique chunks (per-source: 20 each)")

    # Step 3: Rerank with MMR λ=0.7
    chunks_with_docs = pipeline._fetch_chunk_documents(retrieved)
    all_source_refs = pipeline._chunks_to_source_refs(retrieved)

    reranked = pipeline.reranker.rerank(
        query=query,
        chunks=chunks_with_docs,
        source_refs=all_source_refs,
        top_k=settings.top_k_rerank,
        apply_authenticity_weight=True,
        normalize=True,
        mmr_lambda=0.7,
    )

    elapsed = time.time() - start

    # Analyze
    top10_dist = Counter(c["source"] for c in reranked)
    top1_score = reranked[0]["reranker_score"] if reranked else 0.0
    abstained = pipeline.reranker.should_abstain(reranked, threshold=0.35)

    print(f"  Top-10 source distribution:")
    for source in ["quran", "hadith", "tafsir_ar", "tafsir_en"]:
        count = top10_dist.get(source, 0)
        print(f"    {source:12s}: {count}")
    print(f"  Top-1 reranker score: {top1_score:.4f}")
    print(f"  Abstained: {abstained}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Top-10 chunks:")
    for i, c in enumerate(reranked, 1):
        print(f"    {i:2d}. {c['id']:30s} ({c['source']:10s}) score={c['reranker_score']:.4f}")

    return {
        "query": query,
        "type": question_type,
        "expect": expect,
        "top10_dist": dict(top10_dist),
        "top1_score": top1_score,
        "abstained": abstained,
        "elapsed_s": elapsed,
        "top_10": [(c["id"], c["source"], c["reranker_score"]) for c in reranked],
    }


def main() -> None:
    """Run the multi-query validation test."""
    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    print(f"# NUR Multi-Query Architecture Validation")
    print(f"# Architecture: per-source (20 each) + reranker + MMR λ=0.7 + Quran weight 1.3")
    print(f"# Questions: {len(TEST_QUESTIONS)}")

    # Initialize pipeline ONCE
    print(f"\n# Initializing pipeline...", flush=True)
    pipeline = NURPipeline()
    print(f"# Pipeline ready. BGE-M3 device: {pipeline._device}")

    # Run all 5 questions (with error handling — skip failed, continue)
    results = []
    for q in TEST_QUESTIONS:
        try:
            result = run_single_query(
                pipeline=pipeline,
                query=q["query"],
                question_type=q["type"],
                expect=q["expect"],
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ FAILED: {type(e).__name__}: {str(e)[:200]}")
            results.append({
                "query": q["query"],
                "type": q["type"],
                "expect": q["expect"],
                "top10_dist": {},
                "top1_score": 0.0,
                "abstained": True,
                "elapsed_s": 0.0,
                "top_10": [],
                "error": str(e)[:200],
            })
            print(f"  Continuing to next question...\n")

    # Print comparative summary
    print(f"\n\n{'='*90}")
    print(f"  MULTI-QUERY VALIDATION SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Query':<50} {'Q':<3} {'H':<3} {'TAR':<4} {'TEN':<4} {'Top1':<6} {'Time'}")
    print(f"  {'-'*85}")
    for r in results:
        d = r["top10_dist"]
        q_short = r["query"][:48]
        print(f"  {q_short:<50} {d.get('quran',0):<3} {d.get('hadith',0):<3} "
              f"{d.get('tafsir_ar',0):<4} {d.get('tafsir_en',0):<4} "
              f"{r['top1_score']:.2f}  {r['elapsed_s']:.0f}s")
    print(f"{'='*90}")

    # Print detailed results
    for r in results:
        print(f"\n  '{r['query']}'")
        print(f"  Type: {r['type']}")
        print(f"  Expected: {r['expect']}")
        print(f"  Top-10:")
        for i, (cid, src, score) in enumerate(r["top_10"], 1):
            print(f"    {i:2d}. {cid:30s} ({src:10s}) score={score:.4f}")

    print(f"\n# Done.")


if __name__ == "__main__":
    main()
