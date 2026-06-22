"""
NUR V3 Runtime — Pipeline orchestrator

Wires together:
  Step 1: Architect (query decomposition)
  Step 2: Phase A (Quran retrieval → top 5 + confidence A)
  Step 3: Cross-refs auto-pull
  Step 4: Phase B (Hadith retrieval → top 5 + confidence B)
  Step 5: Reporter (Groq structured JSON)
  Step 6: Verification (NLI + Quran char + source ID + tafsir labeling)

Per docs/v3/04_RETRIEVAL_PIPELINE.md.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from .architect import decompose_query  # noqa: E402
from .retriever_quran import retrieve_quran  # noqa: E402
from .retriever_hadith import retrieve_hadith  # noqa: E402
from .cross_refs import auto_pull_hadiths  # noqa: E402
from .reporter import call_reporter  # noqa: E402
from .verifier import verify_response  # noqa: E402


@dataclass
class PipelineResult:
    """Result of a full V3 pipeline run."""
    user_question: str
    detected_lang: str

    # Step 1
    sub_questions: list[str] = field(default_factory=list)

    # Step 2 (Phase A)
    quran_chunks: list[dict] = field(default_factory=list)
    phase_a_status: str = "EMPTY"
    phase_a_confidence: float = 0.0

    # Step 3 (Cross-refs)
    auto_pulled_hadiths: list[dict] = field(default_factory=list)

    # Step 4 (Phase B)
    hadith_chunks: list[dict] = field(default_factory=list)
    phase_b_status: str = "EMPTY"
    phase_b_confidence: float = 0.0

    # Step 5 (Reporter)
    response: dict = field(default_factory=dict)

    # Step 6 (Verification)
    is_valid: bool = False
    verification_errors: list[dict] = field(default_factory=list)

    # Timing
    total_seconds: float = 0.0
    step_timings: dict[str, float] = field(default_factory=dict)


def _detect_language(text: str) -> str:
    """Simple language detection: AR if has Arabic chars, FR if has French accents, else EN."""
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    if arabic_chars > 3:
        return "ar"
    fr_indicators = ["é", "è", "à", "est-ce", "pourquoi", "comment", "quel", "quelle", "le ", "la ", "les "]
    fr_count = sum(1 for ind in fr_indicators if ind in text.lower())
    if fr_count >= 2:
        return "fr"
    return "en"


def run_pipeline(user_question: str, max_retries: int = 1) -> PipelineResult:
    """Run the full V3 pipeline on a user question.

    Returns PipelineResult with all intermediate steps.
    """
    start = time.time()
    result = PipelineResult(
        user_question=user_question,
        detected_lang=_detect_language(user_question),
    )

    # ============== Step 1: Architect ==============
    print("\n" + "=" * 60)
    print("[Step 1] Architect — query decomposition")
    print("=" * 60)
    t0 = time.time()
    result.sub_questions = decompose_query(user_question)
    result.step_timings["architect"] = time.time() - t0
    print(f"  → {len(result.sub_questions)} queries:")
    for i, q in enumerate(result.sub_questions, 1):
        print(f"    {i}. {q}")

    # ============== Step 2: Phase A (Quran) ==============
    print("\n" + "=" * 60)
    print("[Step 2] Phase A — Quran+Tafsir retrieval")
    print("=" * 60)
    t0 = time.time()
    result.quran_chunks, result.phase_a_status, result.phase_a_confidence = retrieve_quran(
        queries=result.sub_questions,
        top_k_initial=400,
        top_k_rerank=5,
    )
    result.step_timings["phase_a"] = time.time() - t0
    print(f"  Top-5 Quran chunks:")
    for i, c in enumerate(result.quran_chunks, 1):
        print(f"    S{i} {c['id']} (rerank={c.get('rerank_score', 0):.3f})")

    # ============== Step 3: Cross-refs auto-pull ==============
    print("\n" + "=" * 60)
    print("[Step 3] Auto-pull hadiths via cross-refs")
    print("=" * 60)
    t0 = time.time()
    result.auto_pulled_hadiths = auto_pull_hadiths(
        quran_chunks=result.quran_chunks,
        max_per_chunk=3,
        max_total=5,
    )
    result.step_timings["cross_refs"] = time.time() - t0

    # ============== Step 4: Phase B (Hadith) ==============
    print("\n" + "=" * 60)
    print("[Step 4] Phase B — Hadith retrieval")
    print("=" * 60)
    t0 = time.time()
    result.hadith_chunks, result.phase_b_status, result.phase_b_confidence = retrieve_hadith(
        queries=result.sub_questions,
        top_k_initial=400,
        top_k_rerank=5,
    )
    result.step_timings["phase_b"] = time.time() - t0
    print(f"  Top-5 Hadith chunks:")
    for i, c in enumerate(result.hadith_chunks, 1):
        print(f"    S{i+5} {c['id']} (rerank={c.get('rerank_score', 0):.3f}, final={c.get('final_score', 0):.3f})")

    # ============== Step 5: Reporter ==============
    print("\n" + "=" * 60)
    print("[Step 5] Reporter — structured JSON generation")
    print("=" * 60)
    t0 = time.time()
    result.response = call_reporter(
        user_question=user_question,
        detected_lang=result.detected_lang,
        phase_a_status=result.phase_a_status,
        phase_a_confidence=result.phase_a_confidence,
        phase_b_status=result.phase_b_status,
        phase_b_confidence=result.phase_b_confidence,
        quran_chunks=result.quran_chunks,
        hadith_chunks=result.hadith_chunks,
        auto_pulled_hadiths=result.auto_pulled_hadiths,
        max_retries=max_retries,
    )
    result.step_timings["reporter"] = time.time() - t0

    # ============== Step 6: Verification ==============
    print("\n" + "=" * 60)
    print("[Step 6] Verification (Pillar 4)")
    print("=" * 60)
    t0 = time.time()
    all_provided_sources = (
        result.quran_chunks
        + result.hadith_chunks
        + result.auto_pulled_hadiths
    )
    result.is_valid, result.verification_errors = verify_response(
        parsed=result.response,
        provided_sources=all_provided_sources,
    )
    result.step_timings["verification"] = time.time() - t0

    if result.verification_errors:
        print(f"  ⚠️ {len(result.verification_errors)} verification issues:")
        for e in result.verification_errors:
            print(f"    - {e}")
    else:
        print("  ✅ All checks passed")

    result.total_seconds = time.time() - start
    print(f"\n[Total] {result.total_seconds:.1f}s")
    for step, sec in result.step_timings.items():
        print(f"  {step}: {sec:.1f}s")

    return result


if __name__ == "__main__":
    import os
    q = " ".join(sys.argv[1:]) or "Est-ce que fumer est haram ?"
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY env var required")
        sys.exit(1)
    res = run_pipeline(q)
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(res.response.get("answer", "(no answer)"))
