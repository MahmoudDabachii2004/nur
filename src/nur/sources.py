"""
Source ID Protocol + clickable URLs for NUR.

WHY THIS MATTERS:
  Without a structured source protocol, the LLM may:
    - Invent references ("Sahih Bukhari #99999" that doesn't exist)
    - Attribute a hadith to the wrong collection
    - Paraphrase a verse and claim it's verbatim

  The Source ID Protocol (Pilier 7) solves this by:
    1. Injecting numbered source IDs into the prompt: [S1], [S2], ...
    2. Forcing the LLM to use only those IDs in its response
    3. Post-processing: map [SX] → rich display with Arabic text + translation + URL

  Each source also gets a stable, clickable URL (Pilier 12):
    - Quran:  https://quran.com/{surah}/{ayah}
    - Hadith: https://sunnah.com/{collection}:{hadith_number}
    - Tafsir: https://quran.com/tafsir/{surah}/{ayah}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceKind = Literal["quran", "hadith", "tafsir_ar", "tafsir_en", "scholar"]

# ----- Hadith collection slugs (for sunnah.com URLs) -----
HADITH_COLLECTION_SLUGS = {
    "Sahih al-Bukhari": "bukhari",
    "Sahih Muslim": "muslim",
    "Sunan Abi Dawud": "abudawud",
    "Jami` at-Tirmidhi": "tirmidhi",
    "Sunan an-Nasa'i": "nasai",
    "Sunan Ibn Majah": "ibnmajah",
    # Extended collections (Phase 5+)
    "Muwatta Malik": "malik",
    "Musnad Ahmad": "ahmad",
    "Sunan ad-Darimi": "darimi",
}


@dataclass(frozen=True)
class SourceRef:
    """A reference to a specific Islamic source (Quran verse, hadith, tafsir section, etc.).

    This is the canonical source identifier used across the NUR system.
    """

    kind: SourceKind
    """Type of source: quran, hadith, tafsir_ar, tafsir_en, scholar."""

    # ----- Location within the source -----
    surah: int | None = None       # For Quran/Tafsir (1-114)
    ayah: int | None = None        # For Quran/Tafsir (1-286)
    collection: str | None = None  # For Hadith (e.g. "Sahih al-Bukhari")
    hadith_number: int | None = None  # For Hadith (sunnah.com numbering)
    scholar: str | None = None     # For scholar opinions
    fatwa_id: str | None = None    # For IslamQA fatwas

    # ----- Content -----
    text_ar: str = ""         # Original Arabic (source of truth)
    text_en: str = ""         # English translation (comprehension aid)
    grade: str | None = None  # For Hadith: "Sahih", "Hasan", "Da'if", "Mawdu"

    # ----- Cross-references -----
    cross_refs: list[str] = ()  # Other SourceRef IDs that explain this one

    @property
    def source_id(self) -> str:
        """Canonical NUR source ID, e.g. 'SRC-QURAN-2-255' or 'SRC-HADITH-BUKHARI-1'."""
        if self.kind == "quran":
            return f"SRC-QURAN-{self.surah}-{self.ayah}"
        if self.kind == "hadith":
            slug = HADITH_COLLECTION_SLUGS.get(self.collection or "", "unknown").upper()
            return f"SRC-HADITH-{slug}-{self.hadith_number}"
        if self.kind == "tafsir_ar":
            return f"SRC-TAFSIR-AR-{self.surah}-{self.ayah}"
        if self.kind == "tafsir_en":
            return f"SRC-TAFSIR-EN-{self.surah}-{self.ayah}"
        if self.kind == "scholar":
            return f"SRC-SCHOLAR-{self.scholar}-{self.fatwa_id}"
        return "SRC-UNKNOWN"

    @property
    def url(self) -> str:
        """Stable, clickable URL to the original source online."""
        if self.kind == "quran":
            return f"https://quran.com/{self.surah}/{self.ayah}"
        if self.kind == "hadith":
            slug = HADITH_COLLECTION_SLUGS.get(self.collection or "", "")
            if slug and self.hadith_number:
                return f"https://sunnah.com/{slug}:{self.hadith_number}"
            return ""
        if self.kind in ("tafsir_ar", "tafsir_en"):
            return f"https://quran.com/tafsir/{self.surah}/{self.ayah}"
        if self.kind == "scholar":
            if self.fatwa_id:
                return f"https://islamqa.info/en/answers/{self.fatwa_id}"
            return ""
        return ""

    @property
    def display_label(self) -> str:
        """Human-readable label for UI display."""
        if self.kind == "quran":
            return f"Quran {self.surah}:{self.ayah}"
        if self.kind == "hadith":
            return f"{self.collection} #{self.hadith_number}"
        if self.kind == "tafsir_ar":
            return f"Tafsir Ibn Kathir (AR) — {self.surah}:{self.ayah}"
        if self.kind == "tafsir_en":
            return f"Tafsir Ibn Kathir (EN) — {self.surah}:{self.ayah}"
        if self.kind == "scholar":
            return f"{self.scholar} — Fatwa #{self.fatwa_id}"
        return "Unknown source"

    @property
    def grade_weight(self) -> float:
        """Authenticity weight for retrieval scoring (Pilier 3).

        Sahih: 1.30 (+30%)
        Hasan: 1.10 (+10%)
        Da'if: 0.50 (-50%)
        Mawdu: 0.00 (excluded from retrieval, kept only for fake-hadith detection)
        Quran/Tafsir/Scholar: 1.00 (neutral)
        """
        if self.kind != "hadith" or not self.grade:
            return 1.0

        g = self.grade.lower()
        if "mawdu" in g or "mawdu'" in g or "fabricated" in g:
            return 0.0
        if "munkar" in g or "rejected" in g:
            return 0.0
        if "da" in g and ("if" in g or "eef" in g) or "weak" in g:
            return 0.50
        if "hasan" in g or "good" in g:
            return 1.10
        if "sahih" in g or "authentic" in g:
            return 1.30
        return 1.0  # Unknown grade — neutral

    def to_prompt_block(self, index: int) -> str:
        """Render this source as an XML block for the LLM prompt (Pilier 7).

        Example:
            <document id="S1">
              <source_id>SRC-QURAN-2-255</source_id>
              <source_type>quran</source_type>
              <label>Quran 2:255</label>
              <grade>N/A (Word of Allah)</grade>
              <arabic>اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ...</arabic>
              <english>Allah! There is no deity except Him, the Ever-Living...</english>
              <url>https://quran.com/2/255</url>
            </document>
        """
        grade_display = self.grade if self.grade else "N/A (Word of Allah)" if self.kind == "quran" else "Unknown"
        return (
            f'<document id="S{index}">\n'
            f"  <source_id>{self.source_id}</source_id>\n"
            f"  <source_type>{self.kind}</source_type>\n"
            f"  <label>{self.display_label}</label>\n"
            f"  <grade>{grade_display}</grade>\n"
            f"  <arabic>{self.text_ar}</arabic>\n"
            f"  <english>{self.text_en}</english>\n"
            f"  <url>{self.url}</url>\n"
            f"</document>"
        )


def render_sources_for_prompt(sources: list[SourceRef]) -> str:
    """Render a list of sources as a single XML block for the LLM prompt.

    Args:
        sources: List of retrieved source references (post-reranking, top-K).

    Returns:
        Formatted XML string to insert into the system/user prompt.
    """
    if not sources:
        return "<retrieved_sources>\n  (No sources retrieved.)\n</retrieved_sources>"

    blocks = [s.to_prompt_block(i + 1) for i, s in enumerate(sources)]
    return "<retrieved_sources>\n" + "\n".join(blocks) + "\n</retrieved_sources>"


def parse_source_id_from_response(text: str) -> list[int]:
    """Extract [S1], [S2], ... references from an LLM response.

    Used to verify that the LLM only cited sources that were actually given.
    """
    import re

    pattern = re.compile(r"\[S(\d+)\]")
    matches = pattern.findall(text)
    return sorted({int(m) for m in matches})
