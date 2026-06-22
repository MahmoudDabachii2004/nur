"""
NUR V3 Runtime — Step 6: Verification (Pillar 4)

Post-generation verification:
  a. NLI check (each sentence entailed by a source)
  b. Quran char-by-char verification (cited AR text must match original)
  c. Source ID validation ([Sn] in answer must exist in provided sources)
  d. Tafsir labeling check

Per docs/v3/06_GENERATION_VERIFICATION.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from nur.arabic import normalize_arabic  # noqa: E402
from nur.config import settings, DATA_DIR  # noqa: E402

# QURAN_DB_PATH for char-by-char verification
QURAN_DB_PATH = DATA_DIR / "processed" / "quran_v3.jsonl"
HADITH_DB_PATH = DATA_DIR / "processed" / "hadith_v3.jsonl"

# Lazy-loaded singletons
_nli_model = None
_quran_lookup = None
_hadith_lookup = None


def _load_nli_model():
    """Lazy-load NLI cross-encoder."""
    global _nli_model
    if _nli_model is not None:
        return _nli_model
    try:
        from sentence_transformers import CrossEncoder
        print(f"[Verifier] Loading NLI model: {settings.nli_model}")
        _nli_model = CrossEncoder(settings.nli_model)
        print("[Verifier] NLI model loaded.")
    except ImportError:
        print("[Verifier] WARN: sentence-transformers not installed, NLI check disabled")
        _nli_model = False  # mark as unavailable
    return _nli_model


def _load_quran_lookup():
    """Load Quran text lookup: source_id → original Arabic text."""
    global _quran_lookup
    if _quran_lookup is not None:
        return _quran_lookup
    _quran_lookup = {}
    if not QURAN_DB_PATH.exists():
        print(f"[Verifier] WARN: {QURAN_DB_PATH} not found")
        return _quran_lookup
    import json
    with QURAN_DB_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                sid = obj.get("id")
                if sid and "text_ar" in obj:
                    _quran_lookup[sid] = obj["text_ar"]
            except json.JSONDecodeError:
                continue
    return _quran_lookup


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences (simple regex-based)."""
    # Match sentences ending in . ! ? or strong clause separators (—, ;, etc.)
    parts = re.split(r'(?<=[.!?])\s+|(?<=—)\s+', text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 20]


def _is_substantive(sentence: str) -> bool:
    """Filter out transitional phrases ('Therefore,', 'In conclusion,') that don't need NLI check."""
    trivial_starts = ("therefore", "thus", "in conclusion", "however", "moreover", "furthermore",
                       "in summary", "en effet", "par conséquent", "cependant", "de plus", "أيضا")
    s = sentence.lower().lstrip()
    for ts in trivial_starts:
        if s.startswith(ts) and len(s) < 80:
            return False
    return True


def verify_nli(answer: str, sources: list[dict]) -> list[dict]:
    """Check each substantive sentence is entailed by at least one source.

    Returns list of error dicts (empty = all sentences pass).
    """
    nli_model = _load_nli_model()
    if not nli_model:  # False means unavailable
        return [{"issue": "nli_unavailable"}]

    sentences = [s for s in _split_into_sentences(answer) if _is_substantive(s)]
    errors = []

    for sent in sentences:
        max_entailment = 0.0
        for source in sources:
            source_text = source.get("embedding_text", "")[:1000]  # cap for NLI speed
            if not source_text:
                continue
            try:
                scores = nli_model.predict([(source_text, sent)])
                # scores = [entailment, neutral, contradiction]
                entail = float(scores[0][0])
                if entail > max_entailment:
                    max_entailment = entail
            except Exception:
                continue

        if max_entailment < settings.nli_threshold:
            errors.append({
                "sentence": sent,
                "max_nli": max_entailment,
                "issue": "not_entailed",
            })

    return errors


def verify_quran_text(citations: list[dict]) -> list[dict]:
    """Verify each Quran citation matches the original Arabic char-by-char.

    After normalization (tashkeel stripping, alef normalization, etc.),
    cited text must EXACTLY equal the original.

    Returns list of error dicts (empty = all match).
    """
    quran_lookup = _load_quran_lookup()
    errors = []

    for cit in citations:
        if cit.get("type") != "quran":
            continue
        sid = cit.get("source_id", "")
        if sid not in quran_lookup:
            errors.append({
                "citation": sid,
                "issue": "source_not_found",
            })
            continue

        original_ar = quran_lookup[sid]
        cited_ar = cit.get("arabic", "")

        norm_original = normalize_arabic(original_ar)
        norm_cited = normalize_arabic(cited_ar)

        if norm_original != norm_cited:
            errors.append({
                "citation": sid,
                "issue": "quran_text_mismatch",
                "expected_chars": len(norm_original),
                "got_chars": len(norm_cited),
            })

    return errors


def verify_source_ids(answer: str, citations: list[dict], provided_sources: list[dict]) -> list[dict]:
    """Verify all [Sn] in answer + all source_ids in citations exist in provided sources."""
    errors = []

    cited_in_answer = set(re.findall(r"\[S(\d+)\]", answer))
    valid_indices = {str(i + 1) for i in range(len(provided_sources))}
    invalid = cited_in_answer - valid_indices
    if invalid:
        errors.append({
            "issue": "invalid_inline_citations",
            "invalid": sorted(invalid),
        })

    valid_source_ids = {s["id"] for s in provided_sources}
    for cit in citations:
        if cit.get("source_id") not in valid_source_ids:
            errors.append({
                "issue": "invalid_source_id_in_citation",
                "invalid_id": cit.get("source_id"),
            })

    return errors


def verify_tafsir_labeling(answer: str, citations: list[dict]) -> list[dict]:
    """Verify tafsir content is properly labeled in the answer."""
    errors = []
    for cit in citations:
        if cit.get("type") == "quran" and cit.get("tafsir_used"):
            tafsir_name = cit["tafsir_used"]
            patterns = [
                f"Tafsir {tafsir_name}",
                f"{tafsir_name} says",
                f"{tafsir_name} explains",
                f"selon {tafsir_name}",
                f"according to {tafsir_name}",
                f"يقول {tafsir_name}",
            ]
            if not any(p.lower() in answer.lower() for p in patterns):
                errors.append({
                    "citation": cit.get("source_id"),
                    "issue": "tafsir_not_labeled",
                    "expected_pattern": patterns[0],
                })
    return errors


def verify_response(
    parsed: dict,
    provided_sources: list[dict],
) -> tuple[bool, list[dict]]:
    """Run all verification checks on the Reporter response.

    Returns (is_valid, all_errors).
    is_valid = True if no critical errors (Quran mismatch / invalid source IDs).
    """
    all_errors: list[dict] = []

    # 1. Source ID validation (cheap, do first)
    all_errors.extend(verify_source_ids(parsed["answer"], parsed["citations"], provided_sources))

    # 2. Quran char-by-char (most critical)
    all_errors.extend(verify_quran_text(parsed["citations"]))

    # 3. Tafsir labeling
    all_errors.extend(verify_tafsir_labeling(parsed["answer"], parsed["citations"]))

    # 4. NLI (expensive, do last)
    nli_errors = verify_nli(parsed["answer"], provided_sources)
    all_errors.extend(nli_errors)

    # Determine severity
    critical_issues = {"quran_text_mismatch", "invalid_source_id_in_citation",
                       "invalid_inline_citations", "source_not_found"}
    has_critical = any(e.get("issue") in critical_issues for e in all_errors)

    return (not has_critical, all_errors)
