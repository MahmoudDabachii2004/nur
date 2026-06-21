"""
Tests for the Source ID Protocol + URL generation.

Run: pytest tests/test_sources.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.sources import (  # noqa: E402
    SourceRef,
    parse_source_id_from_response,
    render_sources_for_prompt,
)


def test_quran_source_id():
    ref = SourceRef(kind="quran", surah=2, ayah=255)
    assert ref.source_id == "SRC-QURAN-2-255"


def test_quran_url():
    ref = SourceRef(kind="quran", surah=2, ayah=255)
    assert ref.url == "https://quran.com/2/255"


def test_hadith_source_id():
    ref = SourceRef(kind="hadith", collection="Sahih al-Bukhari", hadith_number=1)
    assert ref.source_id == "SRC-HADITH-BUKHARI-1"


def test_hadith_url():
    ref = SourceRef(kind="hadith", collection="Sahih al-Bukhari", hadith_number=1)
    assert ref.url == "https://sunnah.com/bukhari:1"


def test_hadith_url_muslim():
    ref = SourceRef(kind="hadith", collection="Sahih Muslim", hadith_number=123)
    assert ref.url == "https://sunnah.com/muslim:123"


def test_tafsir_ar_source_id():
    ref = SourceRef(kind="tafsir_ar", surah=1, ayah=1)
    assert ref.source_id == "SRC-TAFSIR-AR-1-1"


def test_tafsir_en_url():
    ref = SourceRef(kind="tafsir_en", surah=2, ayah=255)
    assert ref.url == "https://quran.com/tafsir/2/255"


def test_grade_weight_sahih():
    ref = SourceRef(kind="hadith", collection="Sahih al-Bukhari", hadith_number=1, grade="Sahih")
    assert ref.grade_weight == 1.30


def test_grade_weight_hasan():
    ref = SourceRef(kind="hadith", collection="Jami` at-Tirmidhi", hadith_number=1, grade="Hasan")
    assert ref.grade_weight == 1.10


def test_grade_weight_daif():
    ref = SourceRef(kind="hadith", collection="Sunan Abi Dawud", hadith_number=1, grade="Da'if")
    assert ref.grade_weight == 0.50


def test_grade_weight_mawdu():
    """Mawdu' (fabricated) hadiths are excluded from retrieval (weight = 0)."""
    ref = SourceRef(kind="hadith", collection="Sunan Abi Dawud", hadith_number=1, grade="Mawdu'")
    assert ref.grade_weight == 0.0


def test_grade_weight_quran_neutral():
    """Quran is the word of Allah — neutral weight (1.0)."""
    ref = SourceRef(kind="quran", surah=1, ayah=1)
    assert ref.grade_weight == 1.0


def test_grade_weight_unknown_grade():
    ref = SourceRef(kind="hadith", collection="Sahih al-Bukhari", hadith_number=1, grade=None)
    assert ref.grade_weight == 1.0  # unknown = neutral


def test_display_label_quran():
    ref = SourceRef(kind="quran", surah=2, ayah=255)
    assert ref.display_label == "Quran 2:255"


def test_display_label_hadith():
    ref = SourceRef(kind="hadith", collection="Sahih Muslim", hadith_number=42)
    assert "Sahih Muslim" in ref.display_label
    assert "42" in ref.display_label


def test_render_sources_for_prompt():
    sources = [
        SourceRef(kind="quran", surah=2, ayah=255, text_ar="...", text_en="..."),
        SourceRef(kind="hadith", collection="Sahih al-Bukhari", hadith_number=1, grade="Sahih"),
    ]
    rendered = render_sources_for_prompt(sources)
    assert "<retrieved_sources>" in rendered
    assert 'id="S1"' in rendered
    assert 'id="S2"' in rendered
    assert "SRC-QURAN-2-255" in rendered
    assert "SRC-HADITH-BUKHARI-1" in rendered
    assert "https://quran.com/2/255" in rendered
    assert "https://sunnah.com/bukhari:1" in rendered


def test_render_empty_sources():
    rendered = render_sources_for_prompt([])
    assert "No sources retrieved" in rendered


def test_parse_source_ids_from_response():
    text = "According to [S1] and [S3], the ruling is..."
    ids = parse_source_id_from_response(text)
    assert ids == [1, 3]


def test_parse_source_ids_dedup():
    text = "[S1] [S1] [S2] [S1]"
    ids = parse_source_id_from_response(text)
    assert ids == [1, 2]  # sorted unique


def test_parse_no_source_ids():
    text = "This response has no source IDs."
    ids = parse_source_id_from_response(text)
    assert ids == []
