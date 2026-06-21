"""
Tests for the Arabic normalization module.

Run: pytest tests/test_arabic.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.arabic import (  # noqa: E402
    normalize_alef,
    normalize_arabic,
    normalize_for_match,
    normalize_ta_marbuta,
    normalize_ya,
    strip_tashkeel,
    strip_tatweel,
)


def test_strip_tashkeel_basic():
    """strip_tashkeel ONLY removes diacritics, NOT letter variants like أ."""
    assert strip_tashkeel("كِتَابٌ") == "كتاب"
    # أ (alef with hamza above) is a LETTER, not a diacritic — preserved
    assert strip_tashkeel("أَوْلَٰئِكَ") == "أولئك"


def test_strip_tashkeel_preserves_consonants():
    # No diacritics → unchanged
    assert strip_tashkeel("كتاب") == "كتاب"
    assert strip_tashkeel("") == ""


def test_strip_tatweel():
    assert strip_tatweel("كتـــاب") == "كتاب"
    assert strip_tatweel("بسم الله") == "بسم الله"


def test_normalize_alef():
    assert normalize_alef("أ") == "ا"
    assert normalize_alef("إ") == "ا"
    assert normalize_alef("آ") == "ا"
    assert normalize_alef("ٱ") == "ا"
    # Test in word
    assert normalize_alef("أحمد") == "احمد"
    assert normalize_alef("إبراهيم") == "ابراهيم"


def test_normalize_ya():
    assert normalize_ya("عيسى") == "عيسي"
    assert normalize_ya("موسى") == "موسي"
    assert normalize_ya("حتى") == "حتي"


def test_normalize_ta_marbuta():
    assert normalize_ta_marbuta("سنة") == "سنه"
    assert normalize_ta_marbuta("رحمة") == "رحمه"


def test_normalize_arabic_full_pipeline():
    # Quranic verse with diacritics + alef variants + ta marbuta + hamza carriers
    text = "أَوْلَٰئِكَ عَلَيْهِمْ صَلَوَاتٌ مِّن رَّبِّهِمْ"
    # After full pipeline:
    #   strip_tashkeel: أولئك عليهم صلوات من ربهم
    #   normalize_alef: اولئك عليهم صلوات من ربهم (أ → ا)
    #   normalize_hamza: اوليك عليهم صلوات من ربهم (ئ → ي)
    expected = "اوليك عليهم صلوات من ربهم"
    assert normalize_arabic(text) == expected


def test_normalize_arabic_empty():
    assert normalize_arabic("") == ""
    assert normalize_arabic(None) == ""  # type: ignore[arg-type]


def test_normalize_arabic_preserves_english():
    # Mixed Arabic + English should not damage the English
    text = "Zakat زكاة"
    result = normalize_arabic(text)
    assert "Zakat" in result
    assert "زكاة" in result or "زكاه" in result  # ta marbuta normalized


def test_normalize_for_match_removes_punctuation():
    text = "قُلْ هُوَ اللَّهُ أَحَدٌ، اللَّهُ الصَّمَدُ"
    result = normalize_for_match(text)
    # No diacritics, no comma
    assert "\u064b" not in result  # no fatha
    assert "," not in result
    assert "قل هو الله احد" in result or "قل هو الله احد" in result


def test_normalize_for_match_lowercases_latin():
    assert normalize_for_match("ALLAH") == "allah"
    assert normalize_for_match("Zakat") == "zakat"


def test_bismillah_normalization():
    """The most common phrase in Islam — must normalize correctly."""
    bismillah = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    result = normalize_arabic(bismillah)
    expected = "بسم الله الرحمن الرحيم"
    assert result == expected, f"Expected '{expected}', got '{result}'"


def test_ayat_al_kursi_normalization():
    """Ayat al-Kursi (2:255) — the most famous verse."""
    verse = "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ"
    result = normalize_arabic(verse)
    expected = "الله لا اله الا هو الحي القيوم"
    assert result == expected, f"Expected '{expected}', got '{result}'"


def test_idempotent():
    """Running normalize twice should give the same result as running once."""
    text = "أَوْلَٰئِكَ عَلَيْهِمْ صَلَوَاتٌ"
    once = normalize_arabic(text)
    twice = normalize_arabic(once)
    assert once == twice


def test_quranic_annotation_signs_stripped():
    """Quranic annotation signs (U+06D6-U+06ED) should be stripped."""
    text = "كِتَابٌ۠"  # has Quranic annotation sign
    result = normalize_arabic(text)
    assert "\u06e0" not in result  # No Quranic annotations
    assert "كتاب" in result
