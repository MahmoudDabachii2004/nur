"""
Arabic text normalization for NUR.

WHY THIS MATTERS:
  Arabic text has many surface forms that refer to the same word:
    - Diacritics (tashkeel):  كِتَابٌ  vs  كتاب  (same word, different rendering)
    - Alef variants:          أ  إ  آ  vs  ا  (all normalize to ا)
    - Ya vs Alif Maksura:     ى  vs  ي  (e.g. عيسى vs عيسي)
    - Ta Marbuta:             ة  vs  ه  (e.g. سنة vs سنه)
    - Tatweel (elongation):   كتــاب  vs  كتاب

  Without normalization, the same word gets different embeddings, destroying
  retrieval precision. This is one of the "7 deadly sins" of existing Islamic
  RAG projects (see docs/ARCHITECTURE.md Section 4, Sin #3).

USAGE:
  from nur.arabic import normalize_arabic, strip_tashkeel, normalize_alef

  text = "أَوْلَٰئِكَ عَلَيْهِمْ صَلَوَاتٌ مِّن رَّبِّهِمْ"
  cleaned = normalize_arabic(text)
"""

from __future__ import annotations

import re
from typing import Literal

# ----- Diacritics (tashkeel) -----
# Arabic diacritical marks — Unicode range U+064B to U+0652 plus supplementary marks
_TASHKEEL_PATTERN = re.compile(
    "["
    "\u0610-\u061a"   # Quranic annotation signs
    "\u064b-\u065f"   # Standard tashkeel (fatḥa, ḍamma, kasra, shadda, sukun, etc.)
    "\u0670"          # Superscript alef
    "\u06d6-\u06dc"   # Quranic annotation signs (extended)
    "\u06df-\u06e8"   # More Quranic marks
    "\u06ea-\u06ed"   # More Quranic marks
    "]"
)

# Tatweel (kashida) — used for visual elongation, no semantic value
_TATWEEL = "\u0640"
_TATWEEL_PATTERN = re.compile(_TATWEEL)

# ----- Letter normalizations -----
# All alef variants → bare alef (ا)
_ALEF_VARIANTS = "أإآٱ"  # not ا itself — that's the target
_ALEF_PATTERN = re.compile(f"[{_ALEF_VARIANTS}]")

# Alif Maqsura (ى) → Ya (ي)
# Common in words like عيسى, موسى, حتى
_ALIF_MAQSURA_PATTERN = re.compile("\u0649")  # ى

# Ta Marbuta (ة) → Ha (ه)
# Note: this is a LOSSY normalization — some argue against it.
# We apply it because retrieval precision matters more than morphological
# correctness for our use case.
_TA_MARBUTA_PATTERN = re.compile("\u0629")  # ة

# Hamza on ya (ئ) → bare ya (ي) — sometimes used, optional
_HAMZA_ON_YA_PATTERN = re.compile("\u0626")  # ئ

# Hamza on waw (ؤ) → bare waw (و) — sometimes used, optional
_HAMZA_ON_WAW_PATTERN = re.compile("\u0624")  # ؤ

# ----- Punctuation / whitespace -----
# Multiple spaces → single space
_MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Leading/trailing whitespace
# (handled by .strip())


def strip_tashkeel(text: str) -> str:
    """Remove all Arabic diacritics (tashkeel / harakat).

    كِتَابٌ → كتاب
    أَوْلَٰئِكَ → اولئك
    """
    return _TASHKEEL_PATTERN.sub("", text)


def strip_tatweel(text: str) -> str:
    """Remove tatweel (kashida) — visual elongation with no semantic value.

    كتــاب → كتاب
    """
    return _TATWEEL_PATTERN.sub("", text)


def normalize_alef(text: str) -> str:
    """Normalize all alef variants to bare alef (ا).

    أ → ا, إ → ا, آ → ا, ٱ → ا
    """
    return _ALEF_PATTERN.sub("ا", text)


def normalize_ya(text: str) -> str:
    """Normalize alif maqsura (ى) to ya (ي).

    عيسى → عيسي, موسى → موسي, حتى → حتي
    """
    return _ALIF_MAQSURA_PATTERN.sub("ي", text)


def normalize_ta_marbuta(text: str) -> str:
    """Normalize ta marbuta (ة) to ha (ه).

    سنة → سنه, رحمة → رحمه

    Note: This is lossy but useful for retrieval.
    """
    return _TA_MARBUTA_PATTERN.sub("ه", text)


def normalize_hamza(text: str) -> str:
    """Normalize hamza-bearing forms to bare carriers.

    ئ → ي, ؤ → و
    (وَاحِدَة and وَاحِدَه should match after this)
    """
    text = _HAMZA_ON_YA_PATTERN.sub("ي", text)
    text = _HAMZA_ON_WAW_PATTERN.sub("و", text)
    return text


def normalize_arabic(
    text: str,
    *,
    strip_diacritics: bool = True,
    strip_tatweel_: bool = True,
    normalize_alef_: bool = True,
    normalize_ya_: bool = True,
    normalize_ta_: bool = True,
    normalize_hamza_: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    """Full Arabic normalization pipeline.

    Default order:
      1. Strip tashkeel (diacritics)
      2. Strip tatweel
      3. Normalize alef variants → ا
      4. Normalize alif maqsura → ي
      5. Normalize ta marbuta → ه
      6. Normalize hamza carriers
      7. Collapse multiple whitespace → single space
      8. Strip leading/trailing whitespace

    Args:
        text: Input Arabic text (may contain tashkeel, alef variants, etc.)
        strip_diacritics: Remove harakat (default True)
        strip_tatweel_: Remove kashida (default True)
        normalize_alef_: أإآٱ → ا (default True)
        normalize_ya_: ى → ي (default True)
        normalize_ta_: ة → ه (default True)
        normalize_hamza_: ئ → ي, ؤ → و (default True)
        collapse_whitespace: collapse runs of whitespace (default True)

    Returns:
        Normalized Arabic string.

    Example:
        >>> normalize_arabic("أَوْلَٰئِكَ عَلَيْهِمْ صَلَوَاتٌ")
        'اولئك عليهم صلوات'
    """
    if not text:
        return ""

    if strip_diacritics:
        text = strip_tashkeel(text)
    if strip_tatweel_:
        text = strip_tatweel(text)
    if normalize_alef_:
        text = normalize_alef(text)
    if normalize_ya_:
        text = normalize_ya(text)
    if normalize_ta_:
        text = normalize_ta_marbuta(text)
    if normalize_hamza_:
        text = normalize_hamza(text)
    if collapse_whitespace:
        text = _MULTI_SPACE_PATTERN.sub(" ", text)

    return text.strip()


def normalize_for_embedding(text: str) -> str:
    """Normalization level used BEFORE embedding.

    Same as `normalize_arabic` with all defaults.
    Kept as a separate function because we may want to tune it differently
    (e.g. keep ta marbuta for embeddings but normalize it for citation matching).
    """
    return normalize_arabic(text)


def normalize_for_match(text: str) -> str:
    """Normalization level used for citation verification (character matching).

    STRONGER normalization — also lowercases Latin chars and removes
    all non-letter characters. Used for the post-generation character-by-character
    Quran verification (Pilier 4, Module T5).
    """
    text = normalize_arabic(text)
    # Lowercase Latin
    text = text.lower()
    # Remove all non-Arabic-letter, non-Latin-letter characters
    text = re.sub(r"[^\u0621-\u064a\u0660-\u0669a-z0-9 ]", "", text)
    # Collapse whitespace again (after non-letter removal)
    text = _MULTI_SPACE_PATTERN.sub(" ", text)
    return text.strip()


# Convenience type for normalization level
NormalizationLevel = Literal["embedding", "match"]
