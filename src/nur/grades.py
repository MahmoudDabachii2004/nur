"""
grades.py

This file defines the hadith grading system for NUR — the education layer that
helps users understand what Sahih, Hasan, Da'if, and Mawdu' mean, and computes
the appropriate warning level for each answer.

WHY THIS EXISTS (docs/PILLARS.md Pillar 3 + Pillar 4 + Pillar 8):
  Hadith grades are not just metadata — they are a theological safety mechanism.
  A layperson may not know the difference between a Sahih (rigorously verified)
  hadith and a Mawdu' (fabricated) one. Presenting them identically is
  irresponsible and potentially harmful: a believer might act on a fabricated
  hadith thinking it is authentic.

  This module provides:
    1. Short, plain-language explanations of each grade level (for users who
       don't know the Arabic terminology).
    2. A two-tier warning system:
       - Per-hadith warning: shown on each individual Da'if or Mawdu' hadith.
       - Answer-level warning: shown at the top of the answer when the overall
         source quality is concerning (e.g., no Sahih or Hasan hadiths found).

THE TWO-TIER WARNING LOGIC (agreed with the user, 2026-06-22):
  The user's instruction was: "if there is no sahih or hasan it should be a big
  watchout and if there is it should be small warning additional context to
  warn max people."

  Translated to logic:
    - If the answer contains ONLY Da'if hadiths (no Sahih, no Hasan):
      → BIG WARNING: "⚠️ No authentic (Sahih) or good (Hasan) hadiths were
        found for this question. The following narrations are weak (Da'if) and
        should not be used as a basis for rulings without scholarly guidance."
    - If the answer contains Da'if hadiths ALONGSIDE Sahih/Hasan:
      → SMALL WARNING: "ℹ️ Some additional context below comes from weak
        (Da'if) narrations. The authentic (Sahih) sources remain the primary
        basis for the answer."
    - If the answer contains any Mawdu' (fabricated) hadith:
      → CRITICAL WARNING: "🚫 WARNING: One or more fabricated (Mawdu')
        narrations were found for this topic. They are shown for awareness
        only — never cite them or act upon them."
    - If the answer contains only Sahih and/or Hasan:
      → NO WARNING.

  The CRITICAL (Mawdu') warning takes precedence over BIG and SMALL.

GRADE LEVELS:
  The hadith dataset (meeAtif/hadith_datasets, see docs/DATA_SOURCES.md)
  stores grades as strings like "Sahih (Darussalam)", "Hasan (Darussalam)",
  "Da'if (Darussalam)", "Sahih (by consensus)", etc. The grader name varies
  (Darussalam, Al-Albani, by consensus) because different scholars grade
  differently — this is itself important information (Pillar 9: Ikhtilaf).

  We normalize these to 5 levels: sahih, hasan, daif, mawdu, unknown.
  The normalization is conservative: if we cannot determine the grade, we
  treat it as "unknown" (neutral weight, no warning) rather than guessing.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any


# ============================================================
# Warning Level Enum — ordered by severity
# ============================================================


class WarningLevel(IntEnum):
    """Severity levels for grade warnings, ordered from least to most severe.

    Used both for per-hadith warnings and answer-level warnings.
    The IntEnum allows comparison: CRITICAL > BIG > SMALL > NONE.
    """

    NONE = 0
    """No warning. Used for Sahih and Hasan hadiths, and for answers that
    contain only Sahih/Hasan sources."""

    SMALL = 1
    """Small informational warning. Used for a Da'if hadith when Sahih/Hasan
    hadiths are also present in the same answer — the Da'if is 'additional
    context', not the primary basis."""

    BIG = 2
    """Big warning. Used when an answer contains ONLY Da'if hadiths (no Sahih,
    no Hasan). The user must be alerted that no authentic sources were found."""

    CRITICAL = 3
    """Critical warning. Used for any Mawdu' (fabricated) hadith. These are
    shown for awareness only — never to be cited or acted upon."""


# ============================================================
# Grade Education Map — what each grade means in plain language
# ============================================================


GRADE_INFO: dict[str, dict[str, str]] = {
    "sahih": {
        "label": "Sahih (Authentic)",
        "explanation": (
            "Rigorously verified: the chain of narration (isnad) is complete, "
            "all narrators are known for their honesty and precise memory, and "
            "the hadith has no hidden defect (illah). This is the highest grade "
            "of authenticity. It can be used as a basis for Islamic rulings."
        ),
        "warning_level": WarningLevel.NONE,
        "warning_text": None,
    },
    "hasan": {
        "label": "Hasan (Good)",
        "explanation": (
            "Reliable but with a minor concern: the chain is complete and "
            "narrators are generally trustworthy, but one narrator's memory "
            "was slightly less precise than the Sahih standard. Hasan hadiths "
            "are acceptable as evidence for rulings, just below Sahih in rank."
        ),
        "warning_level": WarningLevel.NONE,
        "warning_text": None,
    },
    "daif": {
        "label": "Da'if (Weak)",
        "explanation": (
            "The chain has a serious defect: a weak narrator, a break in the "
            "chain, or a hidden flaw. Da'if hadiths cannot be used alone as a "
            "basis for rulings (ahkam). They may be cited for encouragement "
            "(virtuous deeds) only when no stronger source exists, and only "
            "with clear attribution of their weakness."
        ),
        "warning_level": WarningLevel.SMALL,
        "warning_text": (
            "⚠️ This hadith is weak (Da'if). Do not base rulings on it without "
            "scholarly guidance."
        ),
    },
    "mawdu": {
        "label": "Mawdu' (Fabricated)",
        "explanation": (
            "Forged: someone fabricated this text and falsely attributed it to "
            "the Prophet ﷺ. It is NOT a real hadith. It is recorded only so "
            "scholars can identify and warn against it. Never cite it, never "
            "act upon it, and never attribute it to the Prophet ﷺ."
        ),
        "warning_level": WarningLevel.CRITICAL,
        "warning_text": (
            "🚫 WARNING: This narration is fabricated (Mawdu'). It is shown for "
            "awareness only — never cite it or act upon it."
        ),
    },
    "unknown": {
        "label": "Grade Unknown",
        "explanation": (
            "The grade of this hadith could not be determined from the "
            "available data. Treat with caution and consult a scholar."
        ),
        "warning_level": WarningLevel.NONE,
        "warning_text": None,
    },
}
"""Static reference map for each grade level. The `warning_level` field is
the PER-HADITH warning; the answer-level warning is computed separately by
`get_answer_warning()` because it depends on the overall mix of grades."""


# ============================================================
# Grade Normalization — parse raw grade strings into levels
# ============================================================


def normalize_grade_level(grade_string: str | None) -> str:
    """Parse a raw grade string from the hadith dataset into a normalized level.

    The hadith dataset stores grades as strings like:
      - "Sahih (Darussalam)"
      - "Sahih (by consensus)"
      - "Hasan (Darussalam)"
      - "Da'if (Darussalam)"
      - "Da'if (Al-Albani)"
      - "Ungraded"
      - "" (empty)

    We normalize these to one of: "sahih", "hasan", "daif", "mawdu", "unknown".

    The matching is conservative — we look for keywords (case-insensitive).
    If no keyword matches, we return "unknown" rather than guessing.

    Args:
        grade_string: The raw grade string from metadata, e.g. "Sahih (Darussalam)".

    Returns:
        One of: "sahih", "hasan", "daif", "mawdu", "unknown".
    """
    if not grade_string or not grade_string.strip():
        return "unknown"

    g = grade_string.lower().strip()

    # Check Mawdu' first — it's the most critical and must not be misdetected.
    # "mawdu", "mawdu'", "fabricated", "forged" all indicate fabrication.
    if "mawdu" in g or "fabricated" in g or "forged" in g:
        return "mawdu"

    # Munkar and rejected narrations are treated as very weak — closer to Mawdu
    # than to Da'if in severity. We classify them as "mawdu" for safety.
    if "munkar" in g or "rejected" in g:
        return "mawdu"

    # Da'if — check before Sahih/Hasan because "Da'if" is unambiguous.
    # "da'if", "daif", "da'eef", "weak" all indicate weakness.
    if "da'if" in g or "daif" in g or "da'eef" in g or "weak" in g:
        return "daif"

    # Hasan — "hasan" or "good".
    if "hasan" in g or "good" in g:
        return "hasan"

    # Sahih — "sahih" or "authentic".
    if "sahih" in g or "authentic" in g:
        return "sahih"

    # "Ungraded" or anything else — we don't know, so be neutral.
    return "unknown"


def get_grade_info(grade_string: str | None) -> dict[str, Any]:
    """Get the full grade info (label, explanation, warning) for a grade string.

    Args:
        grade_string: The raw grade string from metadata, e.g. "Sahih (Darussalam)".

    Returns:
        A dict with keys: level, label, explanation, warning_level, warning_text.
    """
    level = normalize_grade_level(grade_string)
    info = GRADE_INFO[level]
    return {
        "level": level,
        "label": info["label"],
        "explanation": info["explanation"],
        "warning_level": info["warning_level"],
        "warning_text": info["warning_text"],
    }


# ============================================================
# Answer-Level Warning — computed from the mix of grades in the answer
# ============================================================


def get_answer_warning(grade_levels: list[str]) -> tuple[WarningLevel, str | None]:
    """Compute the answer-level warning from the grades of all cited hadiths.

    This implements the two-tier logic agreed with the user:
      - If any Mawdu' is present → CRITICAL (fabricated hadith warning).
      - Else if NO Sahih and NO Hasan → BIG (no authentic sources warning).
      - Else if Da'if is present alongside Sahih/Hasan → SMALL (additional
        context warning).
      - Else (only Sahih/Hasan) → NONE.

    Quran and Tafsir sources do NOT count toward this calculation — they are
    the Word of Allah and classical exegesis respectively, not graded hadiths.
    Only hadith grades are relevant here.

    Args:
        grade_levels: A list of normalized grade levels for each hadith cited
                      in the answer. Quran/Tafsir sources should be excluded
                      (the caller filters by source_type == "hadith").

    Returns:
        A tuple of (WarningLevel, warning_text_or_None).
    """
    if not grade_levels:
        # No hadiths in the answer at all — no warning needed.
        return WarningLevel.NONE, None

    has_mawdu = "mawdu" in grade_levels
    has_sahih = "sahih" in grade_levels
    has_hasan = "hasan" in grade_levels
    has_daif = "daif" in grade_levels
    has_authentic = has_sahih or has_hasan

    if has_mawdu:
        return WarningLevel.CRITICAL, (
            "🚫 WARNING: One or more fabricated (Mawdu') narrations were found "
            "for this topic. They are shown for awareness only — never cite "
            "them or act upon them."
        )

    if not has_authentic and has_daif:
        return WarningLevel.BIG, (
            "⚠️ IMPORTANT: No authentic (Sahih) or good (Hasan) hadiths were "
            "found for this question in our database. The following narrations "
            "are weak (Da'if) and should NOT be used as a basis for rulings. "
            "Please consult a qualified scholar before acting on this information."
        )

    if has_daif:
        return WarningLevel.SMALL, (
            "ℹ️ Note: Some additional context below comes from weak (Da'if) "
            "narrations. The authentic (Sahih/Hasan) sources remain the "
            "primary basis for this answer."
        )

    return WarningLevel.NONE, None
