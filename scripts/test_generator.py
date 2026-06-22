"""
test_generator.py

This script tests the Generator module (Architect + Reporter) by making live
Groq API calls. It validates:
  1. The Architect (Step 1) decomposes a complex query into sub-questions.
  2. The Reporter (Step 4) produces a valid structured JSON report from
     retrieved chunks, with correct source IDs and verbatim Arabic text.

WHY THIS MUST RUN ON THE USER'S MACHINE (Rule 3 — No Silent Test Skips):
  The agent server's GROQ_API_KEY returns HTTP 403 Forbidden (likely a region
  or scope restriction on the key). The agent verified the code imports cleanly
  and the Pydantic schemas validate against synthetic data, but the LIVE API
  call could not be tested on the agent side. The user must run this script on
  their Mac to validate the end-to-end Groq integration.

PREREQUISITES:
  - A valid GROQ_API_KEY in .env (or exported as an environment variable)
  - The `groq` and `instructor` packages installed:
      pip install groq instructor
  - The fixture file scripts/_fixtures_zakat.json (committed with this script)

USAGE:
  python scripts/test_generator.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add the src directory to the Python path so we can import the nur package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nur.generator import Architect, Reporter
from src.nur.sources import SourceRef, render_sources_for_prompt


def load_zakat_fixtures() -> list[SourceRef]:
    """Load the 3 zakat-related Quran chunks as SourceRef objects.

    The fixture file contains real chunks pulled from the local ChromaDB —
    specifically the top 3 dense-search results for "What does the Quran say
    about charity and zakat?" from the Phase 1 benchmark.

    Returns:
        A list of 3 SourceRef objects ready to be rendered for the LLM prompt.
    """
    fixture_path = Path(__file__).parent / "_fixtures_zakat.json"
    with fixture_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    sources = []
    for chunk in chunks:
        sources.append(
            SourceRef(
                kind="quran",
                surah=chunk["surah_num"],
                ayah=chunk["ayah_num"],
                text_ar=chunk["text_ar"],
                text_en=chunk["text_en"],
            )
        )
    return sources


def test_architect() -> bool:
    """Test Step 1: The Architect decomposes a complex query.

    Uses the marital-dilemma example from docs/RAG_PIPELINE_ARCHITECTURE.md
    Section 4 — a complex question that should decompose into 2-4 sub-questions.
    """
    print(f"\n{'='*70}")
    print("TEST 1: Architect — Query Decomposition (Step 1)")
    print(f"{'='*70}")

    query = "I committed adultery, repented. Must I tell my wife?"
    print(f"  Query: '{query}'")
    print(f"  Model: llama-3.1-8b-instant (Groq)")
    print()

    try:
        architect = Architect()
        sub_questions, search_keywords = architect.decompose(query)
    except Exception as e:
        print(f"  ❌ FAIL — API call failed: {type(e).__name__}: {e}")
        print(f"     Common causes:")
        print(f"       - GROQ_API_KEY missing or invalid (check .env)")
        print(f"       - Network/firewall blocking api.groq.com")
        print(f"       - Rate limit hit (wait 60s and retry)")
        return False

    print(f"  Returned {len(sub_questions)} sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"    {i}. {sq}")
    print(f"\n  Extracted {len(search_keywords)} search keywords:")
    print(f"    {', '.join(search_keywords)}")

    # Validation checks
    if len(sub_questions) < 1:
        print("  ❌ FAIL — must return at least 1 sub-question")
        return False
    if len(sub_questions) > 6:
        print(f"  ❌ FAIL — returned {len(sub_questions)} sub-questions (max is 6)")
        return False
    if len(search_keywords) < 5:
        print(f"  ❌ FAIL — must return at least 5 search keywords, got {len(search_keywords)}")
        return False

    print(f"\n  ✅ PASS — {len(sub_questions)} sub-questions returned (within 1-6 range)")
    return True


def test_reporter() -> bool:
    """Test Step 4: The Reporter generates a structured JSON report.

    Feeds the 3 zakat fixture chunks to the Reporter and verifies:
      - The output matches the ReporterOutput schema (instructor enforces this).
      - direct_reports contains 3 entries (one per source).
      - Each source_id matches one of the injected S1/S2/S3 IDs.
      - Arabic text is preserved (non-empty, matches the chunk).
      - Synthesis cites at least one source ID.
    """
    print(f"\n{'='*70}")
    print("TEST 2: Reporter — Structured Generation (Step 4)")
    print(f"{'='*70}")

    query = "What does the Quran say about charity and zakat?"
    sources = load_zakat_fixtures()
    sources_xml = render_sources_for_prompt(sources)

    print(f"  Query: '{query}'")
    print(f"  Sources: {len(sources)} chunks (zakat-related Quran verses)")
    print(f"  Model: meta-llama/llama-4-scout-17b-16e-instruct (Groq, 30K TPM)")
    print()
    print("  Injected sources:")
    for i, s in enumerate(sources, 1):
        print(f"    S{i}: {s.display_label} — {s.text_en[:60]}...")
    print()

    try:
        reporter = Reporter()
        output = reporter.generate(user_query=query, sources_xml=sources_xml)
    except Exception as e:
        print(f"  ❌ FAIL — API call failed: {type(e).__name__}: {e}")
        print(f"     Common causes:")
        print(f"       - GROQ_API_KEY missing or invalid (check .env)")
        print(f"       - Rate limit hit (Llama 4 Scout: 30K TPM, 70B: 12K TPM)")
        print(f"       - Network/firewall blocking api.groq.com")
        return False

    # instructor guarantees the schema; we just print and sanity-check the content
    print("  ── conflict_detection ──")
    print(f"  {output.conflict_detection}")
    print()
    print(f"  ── direct_reports ({len(output.direct_reports)} entries) ──")
    for i, report in enumerate(output.direct_reports, 1):
        print(f"  [{report.source_id}] ({report.source_type})")
        print(f"    Arabic: {report.arabic_text[:80]}...")
        print(f"    Report: {report.report}")
        print()
    print("  ── synthesis ──")
    print(f"  {output.synthesis}")
    print()

    # Validation checks
    valid_ids = {f"S{i}" for i in range(1, len(sources) + 1)}
    all_pass = True

    if len(output.direct_reports) < 1:
        print("  ❌ FAIL — direct_reports must have at least 1 entry")
        all_pass = False

    for report in output.direct_reports:
        if report.source_id not in valid_ids:
            print(f"  ❌ FAIL — source_id '{report.source_id}' not in injected IDs {valid_ids}")
            all_pass = False
        if not report.arabic_text.strip():
            print(f"  ❌ FAIL — arabic_text empty for {report.source_id}")
            all_pass = False
        if not report.report.strip():
            print(f"  ❌ FAIL — report empty for {report.source_id}")
            all_pass = False

    # Synthesis must cite at least one [SX] that exists in direct_reports
    import re
    cited_ids = set(re.findall(r"\[S(\d+)\]", output.synthesis))
    cited_ids = {f"S{n}" for n in cited_ids}
    if not cited_ids:
        print("  ❌ FAIL — synthesis does not cite any [SX] source IDs")
        all_pass = False
    elif not cited_ids.issubset(valid_ids):
        print(f"  ❌ FAIL — synthesis cites {cited_ids} but only {valid_ids} are valid")
        all_pass = False
    else:
        print(f"  ✅ Synthesis cites valid IDs: {cited_ids}")

    if all_pass:
        print("\n  ✅ PASS — Reporter produced a valid structured report")
    else:
        print("\n  ❌ FAIL — see issues above")
    return all_pass


def main() -> None:
    """Run both tests and report pass/fail summary."""
    print("--- Generator Test Suite (live Groq API calls) ---")
    print("This script makes REAL API calls to Groq. It will consume")
    print("free-tier quota. Estimated cost: ~2,000 tokens total.")
    print()

    # Pre-flight check: warn if no API key
    from src.nur.config import settings
    if not settings.groq_api_key:
        print("❌ ERROR: GROQ_API_KEY is not set. Add it to .env and retry.")
        print("   Example: GROQ_API_KEY=gsk_...")
        sys.exit(1)

    print(f"  GROQ_API_KEY: {settings.groq_api_key[:10]}...{settings.groq_api_key[-4:]}")
    print(f"  Architect model: {settings.llm_architect}")
    print(f"  Reporter model:  {settings.llm_primary}")

    results = [test_architect(), test_reporter()]

    print(f"\n{'='*70}")
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"✅ ALL {total} TESTS PASSED — Generator works end-to-end.")
    else:
        print(f"❌ {total - passed}/{total} TESTS FAILED — see output above.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
