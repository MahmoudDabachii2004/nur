"""
NUR Phase 1 — Step 4: Normalize Arabic + Structure-Aware Chunking

This script processes raw downloaded data into JSONL chunks ready for embedding.

PRINCIPLE (Pilier 5 + 7 deadly sins, Sin #1):
  We NEVER chunk by character count. Each Islamic unit is semantically atomic:
    - 1 ayah    = 1 chunk  (Quran verse — word of Allah, indivisible)
    - 1 hadith  = 1 chunk  (isnad + matn — must stay together)
    - 1 tafsir section = 1 chunk  (commentary on 1-3 ayahs)

  This is the OPPOSITE of "RecursiveCharacterTextSplitter chunk_size=500" used
  by every other Islamic RAG project — and the cause of their broken retrieval.

OUTPUT (data/processed/*.jsonl):
  - quran.jsonl       — 6,236 chunks (Arabic + English + metadata)
  - hadith.jsonl      — 33,738 chunks (Arabic + English + grade + URL)
  - tafsir_ar.jsonl   — ~6,236 chunks (Arabic commentary)
  - tafsir_en.jsonl   — ~6,236 chunks (English commentary)

Each chunk schema (unified for ChromaDB):
  {
    "id": "SRC-QURAN-2-255",
    "kind": "quran",
    "surah": 2,
    "ayah": 255,
    "text_ar": "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ...",
    "text_en": "Allah! There is no deity except Him...",
    "text_ar_normalized": "الله لا اله الا هو الحي القيوم...",  # for embedding
    "text_en_normalized": "allah there is no deity except him...",  # lowercased
    "grade": null,            # for hadith only
    "grade_weight": 1.0,      # for retrieval scoring (Pilier 3)
    "collection": null,       # for hadith only
    "hadith_number": null,    # for hadith only
    "narrator": null,         # for hadith only (e.g. "Abu Hurairah")
    "url": "https://quran.com/2/255",
    "metadata": { ... }       # source-specific extra fields
  }

Usage:
  python scripts/04_normalize_and_chunk.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.arabic import normalize_arabic  # noqa: E402
from nur.config import (  # noqa: E402
    HADITH_DIR,
    PROCESSED_DIR,
    QURAN_DIR,
    TAFSIR_DIR,
)
from nur.sources import HADITH_COLLECTION_SLUGS, SourceRef  # noqa: E402

# ----- Collection name normalization (handle variations) -----
COLLECTION_ALIASES = {
    "bukhari": "Sahih al-Bukhari",
    "sahih bukhari": "Sahih al-Bukhari",
    "sahih al-bukhari": "Sahih al-Bukhari",
    "muslim": "Sahih Muslim",
    "sahih muslim": "Sahih Muslim",
    "abudawud": "Sunan Abi Dawud",
    "abu dawud": "Sunan Abi Dawud",
    "sunan abi dawud": "Sunan Abi Dawud",
    "tirmidhi": "Jami` at-Tirmidhi",
    "jami` at-tirmidhi": "Jami` at-Tirmidhi",
    "nasai": "Sunan an-Nasa'i",
    "sunan an-nasa'i": "Sunan an-Nasa'i",
    "ibnmajah": "Sunan Ibn Majah",
    "ibn majah": "Sunan Ibn Majah",
    "sunan ibn majah": "Sunan Ibn Majah",
}


def normalize_collection_name(name: str) -> str:
    """Normalize a collection name to its canonical form."""
    if not name:
        return ""
    return COLLECTION_ALIASES.get(name.strip().lower(), name.strip())


def normalize_english(text: str) -> str:
    """Light normalization for English text — lowercase, collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def strip_bom(text: str) -> str:
    """Strip Unicode BOM (U+FEFF) that some APIs (alquran.cloud) prepend."""
    if not text:
        return ""
    return text.lstrip("\ufeff").lstrip("\ufffe")


# ============================================================
# QURAN CHUNKING
# ============================================================

def process_quran() -> int:
    """Process Quran into JSONL chunks. Returns chunk count."""
    print("\n[Quran] Processing...")
    ar_path = QURAN_DIR / "quran-uthmani.json"
    en_path = QURAN_DIR / "en.sahih.json"

    if not ar_path.exists():
        print(f"  [SKIP] {ar_path.name} not found — run 01_download_quran.py first.")
        return 0

    # Load Arabic (source of truth)
    with ar_path.open("r", encoding="utf-8") as f:
        ar_data = json.load(f)

    # Load English (optional)
    en_data = None
    if en_path.exists():
        with en_path.open("r", encoding="utf-8") as f:
            en_data = json.load(f)

    # Build English lookup: (surah, ayah) → text
    en_lookup: dict[tuple[int, int], str] = {}
    if en_data:
        for surah in en_data.get("surahs", []):
            for ayah in surah.get("ayahs", []):
                en_lookup[(surah["number"], ayah["number"])] = ayah.get("text", "")

    out_path = PROCESSED_DIR / "quran.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for surah in ar_data.get("surahs", []):
            surah_num = surah["number"]
            surah_name_ar = surah.get("name", "")
            surah_name_en = surah.get("englishName", "")
            revelation_type = surah.get("revelationType", "")

            for ayah in surah.get("ayahs", []):
                ayah_num = ayah["number"]
                text_ar = strip_bom(ayah.get("text", "").strip())
                text_en = en_lookup.get((surah_num, ayah_num), "")

                # Special case: Bismillah at start of every surah except Surah 9 (At-Tawbah)
                # Al-Fatiha ayah 1 IS the Bismillah; for other surahs, ayah 1 includes
                # Bismillah as a prefix in the alquran.cloud API. We keep it as-is for now.

                # Normalize Arabic for embedding
                text_ar_norm = normalize_arabic(text_ar)
                text_en_norm = normalize_english(text_en)

                # Build the source ref for URL generation
                ref = SourceRef(
                    kind="quran",
                    surah=surah_num,
                    ayah=ayah_num,
                    text_ar=text_ar,
                    text_en=text_en,
                )

                chunk = {
                    "id": ref.source_id,
                    "kind": "quran",
                    "surah": surah_num,
                    "ayah": ayah_num,
                    "surah_name_ar": surah_name_ar,
                    "surah_name_en": surah_name_en,
                    "revelation_type": revelation_type,
                    "text_ar": text_ar,
                    "text_en": text_en,
                    "text_ar_normalized": text_ar_norm,
                    "text_en_normalized": text_en_norm,
                    "grade": None,
                    "grade_weight": 1.0,  # Quran is word of Allah — neutral weight
                    "collection": None,
                    "hadith_number": None,
                    "narrator": None,
                    "url": ref.url,
                    "metadata": {
                        "page": ayah.get("page"),
                        "juz": ayah.get("juz"),
                        "hizbQuarter": ayah.get("hizbQuarter"),
                        "ruku": ayah.get("ruku"),
                        "manzil": ayah.get("manzil"),
                        "sajda": ayah.get("sajda"),
                    },
                }

                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                count += 1

    print(f"  [OK] {count} chunks → {out_path.relative_to(PROJECT_ROOT)}")
    return count


# ============================================================
# HADITH CHUNKING
# ============================================================

def process_hadith() -> int:
    """Process all hadith collections. Returns total chunk count."""
    print("\n[Hadith] Processing...")
    meetif_dir = HADITH_DIR / "meetif"

    if not meetif_dir.exists():
        print(f"  [SKIP] {meetif_dir} not found — run 02_download_hadith.py first.")
        return 0

    out_path = PROCESSED_DIR / "hadith.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    grade_dist: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for collection_file in sorted(meetif_dir.glob("*.json")):
            if collection_file.name.startswith("_"):
                continue

            collection_name = collection_file.stem  # e.g. "Sahih al-Bukhari"
            print(f"  [{collection_name}] Loading...")

            with collection_file.open("r", encoding="utf-8") as fh:
                hadiths = json.load(fh)

            if not isinstance(hadiths, list):
                print(f"    [WARN] Not a list — skipping.")
                continue

            for idx, h in enumerate(hadiths):
                # meeAtif schema (verified empirically):
                #   Book, Chapter_Number, Chapter_Title_Arabic, Chapter_Title_English,
                #   Arabic_Text, English_Text, Grade, Reference (sunnah.com URL),
                #   In-book reference (e.g. "Book 1, Hadith 1")
                text_ar = strip_bom(str(h.get("Arabic_Text", h.get("Arabic", ""))).strip())
                text_en = str(h.get("English_Text", h.get("English", ""))).strip()
                grade = str(h.get("Grade", "")).strip() or None

                # The sunnah.com URL is already provided in the "Reference" field
                # (CRITICAL — see ARCHITECTURE.md Section 24, Phase 2:
                #  "sunnah.com URLs: stockées dans metadata Phase 1, NE PAS construire
                #  dynamiquement (numérotation décalée)")
                url = str(h.get("Reference", h.get("Sunnah_URL", ""))).strip() or ""

                # Extract hadith number from URL (e.g. "https://sunnah.com/bukhari:1" → 1)
                # or from "In-book reference" string (e.g. "Book 1, Hadith 1" → 1)
                hadith_num: int | None = None
                if url:
                    # Match the number after the colon
                    import re as _re
                    m = _re.search(r":(\d+)$", url)
                    if m:
                        hadith_num = int(m.group(1))
                if hadith_num is None:
                    # Try in_book_reference
                    in_book = str(h.get("In-book reference", ""))
                    m = _re.search(r"Hadith\s+(\d+)", in_book, _re.IGNORECASE)
                    if m:
                        hadith_num = int(m.group(1))
                if hadith_num is None:
                    hadith_num = idx + 1  # fallback

                # Narrator is embedded at the start of English_Text
                # (e.g. "Narrated 'Umar bin Al-Khattab: ...")
                # We don't extract it now — Phase 5+ can add this
                narrator = None

                # Build source ref for canonical ID and URL fallback
                ref = SourceRef(
                    kind="hadith",
                    collection=collection_name,
                    hadith_number=hadith_num,
                    text_ar=text_ar,
                    text_en=text_en,
                    grade=grade,
                )

                # Use URL from data if present, otherwise construct
                final_url = url if url else ref.url

                # Normalize text
                text_ar_norm = normalize_arabic(text_ar)
                text_en_norm = normalize_english(text_en)

                # Track grade distribution
                if grade:
                    g = grade.lower()
                    if "sahih" in g:
                        bucket = "Sahih"
                    elif "hasan" in g:
                        bucket = "Hasan"
                    elif ("da" in g and "if" in g) or ("da" in g and "eef" in g) or "weak" in g:
                        bucket = "Da'if"
                    elif "mawdu" in g or "munkar" in g:
                        bucket = "Mawdu/Munkar"
                    else:
                        bucket = "Other"
                    grade_dist[bucket] = grade_dist.get(bucket, 0) + 1

                chunk = {
                    "id": ref.source_id,
                    "kind": "hadith",
                    "surah": None,
                    "ayah": None,
                    "collection": collection_name,
                    "hadith_number": hadith_num,
                    "narrator": narrator,
                    "text_ar": text_ar,
                    "text_en": text_en,
                    "text_ar_normalized": text_ar_norm,
                    "text_en_normalized": text_en_norm,
                    "grade": grade,
                    "grade_weight": ref.grade_weight,
                    "url": final_url,
                    "metadata": {
                        "book": h.get("Book"),
                        "chapter_number": h.get("Chapter_Number"),
                        "chapter_arabic": h.get("Chapter_Title_Arabic"),
                        "chapter_english": h.get("Chapter_Title_English"),
                        "in_book_reference": h.get("In-book reference"),
                        "original_index": idx,
                    },
                }

                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total += 1

            print(f"    [{collection_name}] {len(hadiths)} hadiths processed")

    print(f"  [OK] {total} chunks → {out_path.relative_to(PROJECT_ROOT)}")
    if grade_dist:
        print(f"  Grade distribution:")
        for bucket, count in sorted(grade_dist.items(), key=lambda x: -x[1]):
            print(f"    {bucket:20s} {count}")
    return total


# ============================================================
# TAFSIR CHUNKING
# ============================================================

def process_tafsir() -> int:
    """Process Tafsir Ibn Kathir (AR + EN). Returns total chunk count."""
    print("\n[Tafsir] Processing...")
    total = 0

    for lang in ("ar", "en"):
        lang_dir = TAFSIR_DIR / lang
        if not lang_dir.exists():
            print(f"  [{lang.upper()}] {lang_dir} not found — run 03_download_tafsir.py first.")
            continue

        out_path = PROCESSED_DIR / f"tafsir_{lang}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            for surah_file in sorted(lang_dir.glob("*.json")):
                try:
                    surah_num = int(surah_file.stem)
                except ValueError:
                    continue

                with surah_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)

                # spa5k schema: list of {text, ayah, surah} dicts
                sections = []
                if isinstance(data, list):
                    sections = data
                elif isinstance(data, dict):
                    if "ayahs" in data:
                        sections = data["ayahs"]
                    elif "text" in data:
                        sections = [data]

                for i, section in enumerate(sections):
                    if not isinstance(section, dict):
                        continue

                    # spa5k: ayah field gives the ayah number this section comments on
                    ayah_num = section.get("ayah")
                    if not isinstance(ayah_num, int):
                        ayah_num = 1  # fallback

                    text = strip_bom(str(section.get("text", "")).strip())
                    if not text:
                        continue

                    ref = SourceRef(
                        kind=f"tafsir_{lang}",
                        surah=surah_num,
                        ayah=ayah_num,
                        text_ar=text if lang == "ar" else "",
                        text_en=text if lang == "en" else "",
                    )

                    # Normalize
                    if lang == "ar":
                        text_norm = normalize_arabic(text)
                    else:
                        text_norm = normalize_english(text)

                    chunk = {
                        "id": ref.source_id,
                        "kind": f"tafsir_{lang}",
                        "surah": surah_num,
                        "ayah": ayah_num,
                        "text_ar": text if lang == "ar" else "",
                        "text_en": text if lang == "en" else "",
                        "text_ar_normalized": text_norm if lang == "ar" else "",
                        "text_en_normalized": text_norm if lang == "en" else "",
                        "grade": None,
                        "grade_weight": 1.0,
                        "collection": None,
                        "hadith_number": None,
                        "narrator": None,
                        "url": ref.url,
                        "metadata": {
                            "tafsir": "Ibn Kathir",
                            "language": lang,
                            "section_index": i,
                            "text_length": len(text),
                        },
                    }

                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    count += 1

        print(f"  [{lang.upper()}] {count} chunks → {out_path.relative_to(PROJECT_ROOT)}")
        total += count

    return total


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("=" * 60)
    print("NUR Phase 1 — Step 4: Normalize Arabic + Structure-Aware Chunking")
    print("=" * 60)
    print(f"Output: {PROCESSED_DIR.relative_to(PROJECT_ROOT)}\n")
    print("Principle: 1 ayah = 1 chunk, 1 hadith = 1 chunk, 1 tafsir section = 1 chunk")
    print("           (NEVER character-based splitting — see ARCHITECTURE.md Sin #1)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    quran_count = process_quran()
    hadith_count = process_hadith()
    tafsir_count = process_tafsir()

    total = quran_count + hadith_count + tafsir_count

    # Write a summary
    summary_path = PROCESSED_DIR / "_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "quran_chunks": quran_count,
                "hadith_chunks": hadith_count,
                "tafsir_chunks": tafsir_count,
                "total_chunks": total,
                "embedding_target": "BAAI/bge-m3",
                "embedding_dim": 1024,
                "vector_db": "chromadb (dense) + json (sparse)",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("NORMALIZATION + CHUNKING COMPLETE")
    print("=" * 60)
    print(f"  Quran:  {quran_count:,} chunks")
    print(f"  Hadith: {hadith_count:,} chunks")
    print(f"  Tafsir: {tafsir_count:,} chunks (AR + EN)")
    print(f"  TOTAL:  {total:,} chunks ready for embedding")
    print(f"\nSummary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"\nNext step: Run colab/embed_nur_colab.py on Google Colab T4 (see colab/README.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
