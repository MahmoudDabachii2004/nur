"""
test_pipeline.py

Full integration test of the NUR pipeline. Runs a real user question through
all 5 steps (Architect → Retrieval → Reporter) and prints every intermediate
result for transparency.

WHY THIS MUST RUN ON THE USER'S MACHINE (Rule 3 — No Silent Test Skips):
  This test needs:
    1. BGE-M3 model (2.3GB) to encode the user query into dense + sparse vectors
    2. The local ChromaDB database (536MB) to retrieve chunks
    3. A valid GROQ_API_KEY to call the Architect and Reporter LLMs
  The agent server cannot run this test because it has neither the BGE-M3 model
  weights nor a working Groq API key (returns 403). The user MUST run this on
  their Mac where all three are available.

PREREQUISITES:
  - BGE-M3 model cached locally (the first run of test_dense_search.py already
    downloaded it)
  - ChromaDB database extracted to data/chroma_db/
  - GROQ_API_KEY in .env
  - groq, instructor, FlagEmbedding, torch, chromadb packages installed

USAGE:
  python scripts/test_pipeline.py
  python scripts/test_pipeline.py --query "Votre question en français"
  python scripts/test_pipeline.py --force-reasoning  # use 70B model
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add the src directory to the Python path so we can import the nur package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nur.pipeline import NURPipeline, PipelineResult
from src.nur.config import settings


def print_separator(title: str) -> None:
    """Print a visual separator with a title."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_sub_questions(result: PipelineResult) -> None:
    """Print the Architect's sub-questions (Step 1 output)."""
    print_separator("STEP 1 — ARCHITECT: Query Decomposition")
    print(f"  User query: '{result.user_query}'")
    print(f"  Model: {settings.llm_architect}")
    print(f"\n  Generated {len(result.sub_questions)} sub-questions:")
    for i, sq in enumerate(result.sub_questions, 1):
        print(f"    {i}. {sq}")


def print_retrieved_chunks(result: PipelineResult) -> None:
    """Print the retrieved chunks (Step 2 output)."""
    print_separator("STEP 2 — MULTI-QUERY HYBRID RETRIEVAL")
    print(f"  Retrieved {len(result.retrieved_chunks)} unique chunks (top {settings.top_k_initial}).")
    print(f"  Sources searched: {', '.join(NURPipeline.SOURCES)}")
    print(f"\n  {'Rank':<6}{'Chunk ID':<30}{'Source':<12}{'RRF Score':<12}{'D-Rank':<8}{'S-Rank'}")
    print(f"  {'-'*80}")
    for i, chunk in enumerate(result.retrieved_chunks[:15], 1):  # Show top 15
        d_rank = str(chunk["dense_rank"]) if chunk["dense_rank"] else "—"
        s_rank = str(chunk["sparse_rank"]) if chunk["sparse_rank"] else "—"
        print(f"  {i:<6}{chunk['id']:<30}{chunk['source']:<12}{chunk['rrf_score']:<12.6f}{d_rank:<8}{s_rank}")
    if len(result.retrieved_chunks) > 15:
        print(f"  ... and {len(result.retrieved_chunks) - 15} more")


def print_top_chunks(result: PipelineResult) -> None:
    """Print the top 10 chunks sent to the Reporter (Step 3/4 input)."""
    print_separator(f"STEP 3/4 — TOP {len(result.top_chunks)} CHUNKS SENT TO REPORTER")
    for i, ref in enumerate(result.top_chunks, 1):
        print(f"\n  [S{i}] {ref.display_label}")
        print(f"       Source ID: {ref.source_id}")
        print(f"       URL: {ref.url}")
        if ref.grade:
            print(f"       Grade: {ref.grade} (weight: {ref.grade_weight})")
        if ref.text_ar:
            preview = ref.text_ar[:100].replace("\n", " ")
            print(f"       Arabic: {preview}...")
        if ref.text_en:
            preview = ref.text_en[:100].replace("\n", " ")
            print(f"       English: {preview}...")


def print_report(result: PipelineResult) -> None:
    """Print the Reporter's structured output (Step 4 output) + grade warnings."""
    print_separator("STEP 4 — REPORTER: Structured Generation")

    if result.abstained:
        print("  🚫 PIPELINE ABSTAINED — no report generated.")
        print("     Top reranker score < 0.35 → insufficient reliable sources.")
        return

    if not result.report:
        print("  ❌ No report generated (pipeline error).")
        return

    print(f"  Model: {settings.llm_primary}")

    # Print answer-level warning FIRST (if any) — this is the most important
    # information for the user to see before reading the answer.
    if result.answer_warning:
        print(f"\n  {'─'*60}")
        print(f"  ⚠️  ANSWER WARNING:")
        print(f"  {result.answer_warning}")
        print(f"  {'─'*60}")

    print(f"\n  ── conflict_detection ──")
    print(f"  {result.report.conflict_detection}")

    print(f"\n  ── direct_reports ({len(result.report.direct_reports)} entries) ──")
    for report in result.report.direct_reports:
        print(f"\n  [{report.source_id}] ({report.source_type})")
        if report.arabic_text:
            preview = report.arabic_text[:120].replace("\n", " ")
            print(f"    Arabic: {preview}...")
        print(f"    Report: {report.report}")

        # Show grade explanation for hadith sources (Pillar 3 education layer)
        if report.source_id in result.grade_explanations:
            explanation = result.grade_explanations[report.source_id]
            print(f"    📚 Grade: {explanation[:200]}")

    print(f"\n  ── synthesis ──")
    print(f"  {result.report.synthesis}")


def validate_report(result: PipelineResult) -> bool:
    """Run validation checks on the pipeline output.

    Returns True if all checks pass, False otherwise.
    """
    print_separator("VALIDATION")
    all_pass = True

    if result.error:
        print(f"  ❌ Pipeline error: {result.error}")
        return False

    # Check 0: Abstention (Pillar 4)
    if result.abstained:
        print("  🚫 Pipeline ABSTAINED — top reranker score < 0.35 threshold.")
        print("     This is correct behavior (Pillar 4: when in doubt, abstain).")
        print("     The system did not find sufficient reliable sources.")
        return True  # abstention is a valid outcome, not a failure

    # Check 1: Architect returned sub-questions
    if not result.sub_questions:
        print("  ❌ Architect returned no sub-questions")
        all_pass = False
    elif len(result.sub_questions) > 6:
        print(f"  ❌ Architect returned {len(result.sub_questions)} sub-questions (max is 6)")
        all_pass = False
    else:
        print(f"  ✅ Architect returned {len(result.sub_questions)} sub-questions")

    # Check 2: Retrieval returned chunks
    if not result.retrieved_chunks:
        print("  ❌ Retrieval returned no chunks")
        all_pass = False
    else:
        print(f"  ✅ Retrieval returned {len(result.retrieved_chunks)} chunks")

    # Check 3: Top chunks were converted to SourceRefs
    if not result.top_chunks:
        print("  ❌ No SourceRef objects built for the Reporter")
        all_pass = False
    else:
        print(f"  ✅ {len(result.top_chunks)} SourceRef objects built")

    # Check 4: Reporter produced a valid report
    if not result.report:
        print("  ❌ Reporter produced no report")
        all_pass = False
    else:
        report = result.report

        # Check 4a: direct_reports non-empty
        if not report.direct_reports:
            print("  ❌ direct_reports is empty")
            all_pass = False
        else:
            print(f"  ✅ direct_reports has {len(report.direct_reports)} entries")

            # Check 4b: all source IDs are valid
            valid_ids = {f"S{i}" for i in range(1, len(result.top_chunks) + 1)}
            for dr in report.direct_reports:
                if dr.source_id not in valid_ids:
                    print(f"  ❌ Invalid source_id '{dr.source_id}' (valid: {valid_ids})")
                    all_pass = False
            if all_pass:
                print(f"  ✅ All source IDs are valid: {[dr.source_id for dr in report.direct_reports]}")

        # Check 4c: synthesis cites valid source IDs
        # Accept BOTH [SX] and (SX) — LLMs use parentheses naturally
        import re
        cited = set(re.findall(r"[\[\(]S(\d+)[\]\)]", report.synthesis))
        cited_ids = {f"S{n}" for n in cited}
        if not cited_ids:
            print("  ❌ Synthesis does not cite any [SX] source IDs")
            all_pass = False
        elif not cited_ids.issubset(valid_ids):
            print(f"  ❌ Synthesis cites invalid IDs: {cited_ids - valid_ids}")
            all_pass = False
        else:
            print(f"  ✅ Synthesis cites valid IDs: {cited_ids}")

    return all_pass


def main() -> None:
    """Run the full pipeline test."""
    import argparse

    parser = argparse.ArgumentParser(description="Test the NUR pipeline end-to-end.")
    parser.add_argument(
        "--query", "-q",
        default="What does the Quran say about charity and zakat?",
        help="The question to ask NUR.",
    )
    parser.add_argument(
        "--force-reasoning",
        action="store_true",
        help="Use the Llama 3.3 70B reasoning model instead of Scout.",
    )
    args = parser.parse_args()

    print_separator("NUR PIPELINE — FULL INTEGRATION TEST")
    print(f"  This test makes REAL API calls to Groq and loads BGE-M3.")
    print(f"  Estimated cost: ~6,000-8,000 tokens (retrieval + 2 LLM calls).")
    print(f"  Estimated time: 30-60 seconds.")

    # Pre-flight check
    if not settings.groq_api_key:
        print("\n❌ ERROR: GROQ_API_KEY is not set. Add it to .env and retry.")
        sys.exit(1)

    print(f"\n  GROQ_API_KEY: {settings.groq_api_key[:10]}...{settings.groq_api_key[-4:]}")
    print(f"  Architect:   {settings.llm_architect}")
    print(f"  Reporter:    {settings.llm_primary}")
    if args.force_reasoning:
        print(f"  Reasoning:   {settings.llm_reasoning} (FORCED)")

    # Initialize the pipeline
    print_separator("INITIALIZING PIPELINE")
    start_time = time.time()
    pipeline = NURPipeline()
    init_time = time.time() - start_time
    print(f"  Pipeline initialized in {init_time:.1f}s")
    print(f"  BGE-M3 device: {pipeline._device}")
    print(f"  (BGE-M3 will load on first query — expect ~10s delay)")

    # Run the query
    print_separator("RUNNING QUERY")
    print(f"  Query: '{args.query}'")
    start_time = time.time()
    result = pipeline.query(args.query, force_reasoning=args.force_reasoning)
    elapsed = time.time() - start_time
    print(f"\n  Pipeline completed in {elapsed:.1f}s")

    # Print all intermediate results
    print_sub_questions(result)
    print_retrieved_chunks(result)
    print_top_chunks(result)
    print_report(result)

    # Validate
    all_pass = validate_report(result)

    # Final summary
    print_separator("SUMMARY")
    if all_pass and not result.error:
        print(f"  ✅ ALL CHECKS PASSED — Pipeline works end-to-end.")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Sub-questions: {len(result.sub_questions)}")
        print(f"  Retrieved chunks: {len(result.retrieved_chunks)}")
        print(f"  Chunks sent to Reporter: {len(result.top_chunks)}")
        print(f"  Direct reports: {len(result.report.direct_reports)}")
    else:
        print(f"  ❌ SOME CHECKS FAILED — see output above.")
        if result.error:
            print(f"  Error: {result.error}")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()
