"""
test_ground_truth.py — Evaluate pipeline against authoritative ground truth.

For each question, we know EXACTLY which verses/hadiths should be found.
This test runs the pipeline and measures:
  - RECALL: % of expected verses found in the top-10
  - PRECISION: % of top-10 chunks that are in the expected list
  - Top-1 score: quality indicator

This is the DEFINITIVE test. If recall < 50%, the pipeline has a real problem.

USAGE:
  python3 scripts/test_ground_truth.py
  python3 scripts/test_ground_truth.py --query "What does the Quran say about charity and zakat?"
  python3 scripts/test_ground_truth.py --config all     # all sub-q + keywords
  python3 scripts/test_ground_truth.py --config keywords # keywords only
  python3 scripts/test_ground_truth.py --config minimal  # 3 sub-q + keywords
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
from scripts.ground_truth import (
    GROUND_TRUTH,
    get_expected_global_ayahs,
    get_expected_hadith_ids,
)


def run_evaluation(
    pipeline: NURPipeline,
    query: str,
    sub_questions: list[str],
    search_keywords: list[str],
    config_name: str,
    include_sub_questions: bool,
    max_sub_questions: int | None,
) -> dict:
    """Run pipeline on one query and evaluate against ground truth.

    Returns a dict with: query, config, recall_quran, recall_hadith,
    precision, top10_ids, expected_found, expected_missing.
    """
    all_queries = [query]
    if include_sub_questions:
        subs = sub_questions[:max_sub_questions] if max_sub_questions else sub_questions
        all_queries = all_queries + subs
    all_queries = all_queries + search_keywords

    # Get ground truth
    expected_quran = get_expected_global_ayahs(query)
    expected_hadith = get_expected_hadith_ids(query)

    # Convert expected to chunk IDs for matching
    expected_quran_ids = set()
    for surah, global_ayah, desc in expected_quran:
        expected_quran_ids.add(f"quran_{surah}_{global_ayah}")
        # Also add tafsir versions (if the verse is found as tafsir, it counts)
        expected_quran_ids.add(f"tafsir_ar_{surah}_{global_ayah}")
        expected_quran_ids.add(f"tafsir_en_{surah}_{global_ayah}")

    expected_hadith_ids = set()
    for collection, number, desc in expected_hadith:
        expected_hadith_ids.add(f"hadith_{collection}_{number}")

    all_expected = expected_quran_ids | expected_hadith_ids

    start = time.time()

    # Retrieve + rerank
    retrieved = pipeline._retrieve(all_queries, top_k=settings.top_k_initial)
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

    # Evaluate
    top10_ids = [c["id"] for c in reranked]
    top10_set = set(top10_ids)

    found = all_expected & top10_set
    missing = all_expected - top10_set

    # Separate Quran vs Hadith recall
    quran_expected = {eid for eid in expected_quran_ids if eid.startswith("quran_")}
    quran_found = quran_expected & top10_set
    hadith_expected = expected_hadith_ids
    hadith_found = hadith_expected & top10_set

    # Also check if expected verses appear as tafsir
    tafsir_expected = {eid for eid in expected_quran_ids if eid.startswith("tafsir_")}
    tafsir_found = tafsir_expected & top10_set

    quran_recall = len(quran_found) / len(quran_expected) * 100 if quran_expected else 0
    hadith_recall = len(hadith_found) / len(hadith_expected) * 100 if hadith_expected else 0

    # Precision: how many of top-10 are in expected?
    precision = len(found) / len(top10_set) * 100 if top10_set else 0

    top10_dist = Counter(c["source"] for c in reranked)
    top1_score = reranked[0]["reranker_score"] if reranked else 0.0

    label_parts = []
    if include_sub_questions:
        n_subs = len(subs)
        label_parts.append(f"{n_subs} sub-q")
    else:
        label_parts.append("0 sub-q")
    label_parts.append(f"{len(search_keywords)} kw")
    label = " + ".join(label_parts)

    print(f"\n  [{config_name}] ({label}, {len(all_queries)} queries)")
    print(f"    Pool: {len(retrieved)} | Top-10: Q={top10_dist.get('quran',0)} H={top10_dist.get('hadith',0)} "
          f"TAR={top10_dist.get('tafsir_ar',0)} TEN={top10_dist.get('tafsir_en',0)}")
    print(f"    Quran recall: {len(quran_found)}/{len(quran_expected)} = {quran_recall:.0f}%")
    print(f"    Hadith recall: {len(hadith_found)}/{len(hadith_expected)} = {hadith_recall:.0f}%")
    print(f"    Tafsir of expected verses found: {len(tafsir_found)}")
    print(f"    Precision: {len(found)}/{len(top10_set)} = {precision:.0f}%")
    print(f"    Top1={top1_score:.4f} | {elapsed:.1f}s")

    if missing:
        print(f"    MISSING expected:")
        for mid in sorted(missing):
            # Find description
            desc = ""
            for surah, ga, d in expected_quran:
                if f"quran_{surah}_{ga}" == mid or f"tafsir_ar_{surah}_{ga}" == mid or f"tafsir_en_{surah}_{ga}" == mid:
                    desc = d
                    break
            for col, num, d in expected_hadith:
                if f"hadith_{col}_{num}" == mid:
                    desc = d
                    break
            print(f"      ❌ {mid:30s} — {desc}")

    if found:
        print(f"    FOUND expected:")
        for fid in sorted(found):
            print(f"      ✅ {fid}")

    return {
        "query": query,
        "config": config_name,
        "label": label,
        "n_queries": len(all_queries),
        "quran_recall": quran_recall,
        "quran_found": len(quran_found),
        "quran_expected": len(quran_expected),
        "hadith_recall": hadith_recall,
        "hadith_found": len(hadith_found),
        "hadith_expected": len(hadith_expected),
        "tafsir_found": len(tafsir_found),
        "precision": precision,
        "top1_score": top1_score,
        "elapsed_s": elapsed,
        "top_10_ids": top10_ids,
        "found": sorted(found),
        "missing": sorted(missing),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", default=None)
    parser.add_argument(
        "--config", "-c",
        choices=["all", "keywords", "minimal"],
        default="all",
        help="Query decomposition config: all (6 sub-q), keywords (0 sub-q), minimal (3 sub-q)",
    )
    args = parser.parse_args()

    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    # Determine which questions to test
    if args.query:
        queries = [args.query]
    else:
        queries = list(GROUND_TRUTH.keys())

    print(f"# NUR Ground Truth Evaluation")
    print(f"# Questions: {len(queries)}")
    print(f"# Config: {args.config}")

    print(f"\n# Initializing pipeline...", flush=True)
    pipeline = NURPipeline()
    print(f"# Pipeline ready. BGE-M3 device: {pipeline._device}")

    # Config settings
    if args.config == "all":
        include_subs, max_subs = True, None
        config_name = "Config 1 (all sub-q)"
    elif args.config == "keywords":
        include_subs, max_subs = False, None
        config_name = "Config 2 (keywords only)"
    else:  # minimal
        include_subs, max_subs = True, 3
        config_name = "Config 3 (minimal 3 sub-q)"

    all_results = []

    for query in queries:
        print(f"\n{'='*70}")
        print(f"  Query: '{query}'")
        gt = GROUND_TRUTH.get(query, {})
        print(f"  Ground truth: {len(gt.get('expected_quran_standard',[]))} Quran verses, "
              f"{len(gt.get('expected_hadith',[]))} hadiths")
        print(f"{'='*70}")

        # Step 1: Architect
        sub_questions, search_keywords = pipeline.generator.decompose_query(query)
        print(f"  Sub-questions: {len(sub_questions)}, Keywords: {len(search_keywords)}")

        result = run_evaluation(
            pipeline=pipeline,
            query=query,
            sub_questions=sub_questions,
            search_keywords=search_keywords,
            config_name=config_name,
            include_sub_questions=include_subs,
            max_sub_questions=max_subs,
        )
        all_results.append(result)

    # Print summary table
    print(f"\n\n{'='*100}")
    print(f"  GROUND TRUTH EVALUATION SUMMARY")
    print(f"  Config: {config_name}")
    print(f"{'='*100}")
    print(f"  {'Query':<45} {'Q_Recall':<10} {'H_Recall':<10} {'Tafsir':<8} {'Precision':<10} {'Top1':<7} {'Time'}")
    print(f"  {'-'*95}")
    for r in all_results:
        q_short = r["query"][:43]
        print(f"  {q_short:<45} {r['quran_found']}/{r['quran_expected']} ({r['quran_recall']:.0f}%)   "
              f"{r['hadith_found']}/{r['hadith_expected']} ({r['hadith_recall']:.0f}%)   "
              f"{r['tafsir_found']:<8} {r['precision']:.0f}%       "
              f"{r['top1_score']:.2f}  {r['elapsed_s']:.0f}s")
    print(f"{'='*100}")

    # Overall stats
    avg_q_recall = sum(r["quran_recall"] for r in all_results) / len(all_results) if all_results else 0
    avg_h_recall = sum(r["hadith_recall"] for r in all_results) / len(all_results) if all_results else 0
    avg_precision = sum(r["precision"] for r in all_results) / len(all_results) if all_results else 0
    print(f"\n  AVERAGES:")
    print(f"    Quran recall:   {avg_q_recall:.0f}%")
    print(f"    Hadith recall:  {avg_h_recall:.0f}%")
    print(f"    Precision:      {avg_precision:.0f}%")

    print(f"\n# Done.")


if __name__ == "__main__":
    main()
