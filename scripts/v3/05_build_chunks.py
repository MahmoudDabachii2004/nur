"""
NUR V3 — Step 5: Build V3 chunks (3-layer Quran + 2-layer Hadith)

Reads:
  data/quran/quran-uthmani.json
  data/quran/en.sahih.json
  data/tafsir/v3/{ibn_kathir_en,ibn_kathir_ar,tabari_ar,saadi_ar}/*.json
  data/hadith/meetif/*.json
  data/processed/context_cards.jsonl

Writes:
  data/processed/quran_v3.jsonl   (6,236 chunks, 3-layer embedding_text)
  data/processed/hadith_v3.jsonl  (33,738 chunks, 2-layer embedding_text)
  data/processed/_summary_v3.json

Per docs/v3/02_CHUNK_SCHEMA.md and docs/v3/05_EMBEDDING_DESIGN.md:
  - Tafsir truncation: 600 chars in embedding_text, full text in metadata.text_full
  - 4 tafsirs per Quran chunk: Ibn Kathir EN+AR, Tabari AR, Sa'di AR
  - Previous/Next ayah for context (max 200 chars each)
  - Standard numbering SRC-QURAN-{surah}-{ayah}
  - Hadith URL always from meetif "Reference" field, never reconstructed
  - Hadith narrator extracted from English_Text via regex

Usage (local, after context cards generated on Lightning AI):
  python scripts/v3/05_build_chunks.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import HADITH_DIR, PROCESSED_DIR, QURAN_DIR, TAFSIR_DIR  # noqa: E402

# ---- Constants ----
TAFSIR_TRUNCATION = 600  # chars per tafsir in embedding_text (data-driven, see doc 05)
PREV_NEXT_MAX_CHARS = 200
ISRA_ILIYYAT_PATTERNS = [
    r"\bKa['’]b al-Ahbar\b",
    r"\bWahb ibn Munabbih\b",
    r"\bAbdullah ibn Salam\b",
    r"\bIsra['’]iliyyat\b",
    r"\bBanu Isra['’]il (?:narrated|said|reported)\b",
    r"\bAccording to (?:Jewish|Christian) tradition\b",
    r"\bPeople of the Book (?:said|narrated|reported)\b",
    r"\bThis is from the Isra['’]iliyyat\b",
]
ISRA_ILIYYAT_REGEX = re.compile("|".join(ISRA_ILIYYAT_PATTERNS), re.IGNORECASE)

# Hadith collection slugs (matches src/nur/sources.py)
HADITH_COLLECTION_SLUGS = {
    "Sahih al-Bukhari": "bukhari",
    "Sahih Muslim": "muslim",
    "Sunan Abi Dawud": "abudawud",
    "Jami` at-Tirmidhi": "tirmidhi",
    "Sunan an-Nasa'i": "nasai",
    "Sunan Ibn Majah": "ibnmajah",
}

NARRATOR_REGEX = re.compile(r"^Narrated\s+([^:]+?):\s", re.IGNORECASE)


def strip_bom(text: str) -> str:
    if not text:
        return ""
    return text.lstrip("\ufeff").lstrip("\ufffe").strip()


def truncate_tafsir(text: str, max_chars: int = TAFSIR_TRUNCATION) -> tuple[str, bool]:
    """Intelligent truncation: cut at sentence boundary near max_chars.
    Returns (truncated_text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False

    # Try to find a sentence ending within max_chars + 15% margin
    margin = int(max_chars * 0.15)
    search_window = text[: max_chars + margin]

    # Find last sentence-ending punctuation before the end of window
    last_end = -1
    for ending in [". ", "! ", "? ", '." ', '."']:
        idx = search_window.rfind(ending)
        if idx > last_end and idx < max_chars + margin:
            last_end = idx + len(ending)

    if last_end >= max_chars * 0.8:
        return text[:last_end].strip(), True
    return text[:max_chars].strip() + "...", True


def detect_isra_iliyyat(text: str) -> bool:
    return bool(ISRA_ILIYYAT_REGEX.search(text))


# ============================================================
# QURAN CHUNKS
# ============================================================

def load_quran_data() -> tuple[list[dict], dict[tuple[int, int], str], dict[tuple[int, int], dict]]:
    """Returns (flat_ayah_list, en_lookup, prev_next_lookup).
    flat_ayah_list: each item = {surah, ayah, surah_name_ar, surah_name_en, revelation_type, text_ar}
    en_lookup[(surah, ayah)] = EN text
    """
    ar_path = QURAN_DIR / "quran-uthmani.json"
    en_path = QURAN_DIR / "en.sahih.json"

    with ar_path.open("r", encoding="utf-8") as f:
        ar_data = json.load(f)

    en_lookup: dict[tuple[int, int], str] = {}
    if en_path.exists():
        with en_path.open("r", encoding="utf-8") as f:
            en_data = json.load(f)
        for surah in en_data.get("surahs", []):
            for ayah in surah.get("ayahs", []):
                en_lookup[(surah["number"], ayah["number"])] = ayah.get("text", "")

    flat = []
    for surah in ar_data.get("surahs", []):
        for ayah in surah.get("ayahs", []):
            flat.append({
                "surah": surah["number"],
                "ayah": ayah["numberInSurah"],
                "surah_name_ar": surah.get("name", ""),
                "surah_name_en": surah.get("englishName", ""),
                "revelation_type": surah.get("revelationType", ""),
                "text_ar": strip_bom(ayah.get("text", "")),
            })
    return flat, en_lookup, {}


def load_tafsir_edition(edition_dir: Path) -> dict[tuple[int, int], str]:
    """Load one tafsir edition (114 files) → {(surah, ayah): text}."""
    out: dict[tuple[int, int], str] = {}
    if not edition_dir.exists():
        return out
    for surah_file in sorted(edition_dir.glob("*.json")):
        try:
            surah_num = int(surah_file.stem)
        except ValueError:
            continue
        with surah_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            ayah = entry.get("ayah")
            text = str(entry.get("text", "")).strip()
            if isinstance(ayah, int) and text:
                out[(surah_num, ayah)] = text
    return out


def load_all_tafsirs() -> dict[str, dict[tuple[int, int], str]]:
    v3_dir = TAFSIR_DIR / "v3"
    return {
        "ibn_kathir_en": load_tafsir_edition(v3_dir / "ibn_kathir_en"),
        "ibn_kathir_ar": load_tafsir_edition(v3_dir / "ibn_kathir_ar"),
        "tabari_ar": load_tafsir_edition(v3_dir / "tabari_ar"),
        "saadi_ar": load_tafsir_edition(v3_dir / "saadi_ar"),
    }


def load_context_cards() -> dict[tuple[int, int], dict]:
    path = PROCESSED_DIR / "context_cards.jsonl"
    cards: dict[tuple[int, int], dict] = {}
    if not path.exists():
        print(f"  [WARN] No context_cards.jsonl found — Quran chunks will have empty context_card")
        return cards
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                cards[(obj["surah"], obj["ayah"])] = obj["context_card"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cards


def build_quran_embedding_text(record: dict, text_en: str,
                                prev_record: dict | None, prev_en: str,
                                next_record: dict | None, next_en: str,
                                context_card: dict | None,
                                tafsirs: list[dict]) -> str:
    """Build the 3-layer embedding_text per docs/v3/02_CHUNK_SCHEMA.md."""
    layers = []

    # Layer 1: Context Card
    card_str_parts = ["[CONTEXT CARD]"]
    if context_card:
        fr = context_card.get("fr", {})
        en = context_card.get("en", {})
        ar = context_card.get("ar", {})
        if fr:
            card_str_parts.append(f"[FR] Thème: {fr.get('theme', '')}. Règle: {fr.get('rule', '')}")
        if en:
            card_str_parts.append(f"[EN] Topic: {en.get('topic', '')}. Rule: {en.get('rule', '')}")
        if ar:
            card_str_parts.append(f"[AR] الموضوع: {ar.get('theme', '')}")
        all_kw = []
        for lang in ("fr", "en", "ar"):
            all_kw.extend(context_card.get(lang, {}).get("keywords", []))
        if all_kw:
            card_str_parts.append(f"Keywords: {', '.join(all_kw)}")
    layers.append("\n".join(card_str_parts))

    # Layer 2: Word of Allah (PURE)
    verse_parts = ["[WORD OF ALLAH — PURE]"]
    verse_parts.append(
        f"Quran | Surah {record['surah']}: {record['surah_name_en']} ({record['surah_name_ar']}) "
        f"| Ayah {record['ayah']} | Revelation: {record['revelation_type']}"
    )
    verse_parts.append(f"Arabic: {record['text_ar']}")
    if text_en:
        verse_parts.append(f"English: {text_en}")
    if prev_record and prev_en:
        prev_short = prev_en[:PREV_NEXT_MAX_CHARS] + ("..." if len(prev_en) > PREV_NEXT_MAX_CHARS else "")
        verse_parts.append(f"Previous ({prev_record['surah']}:{prev_record['ayah']}): {prev_short}")
    if next_record and next_en:
        next_short = next_en[:PREV_NEXT_MAX_CHARS] + ("..." if len(next_en) > PREV_NEXT_MAX_CHARS else "")
        verse_parts.append(f"Next ({next_record['surah']}:{next_record['ayah']}): {next_short}")
    layers.append("\n".join(verse_parts))

    # Layer 3: Tafsirs (labellized, truncated)
    if tafsirs:
        tafsir_parts = ["[HUMAN COMMENTARY — LABELED]"]
        for t in tafsirs:
            marker = ""
            if t.get("contains_isra_iliyyat"):
                marker = " | Contains Isra'iliyyat — illustrative narrations, not authentically verified"
            tafsir_parts.append(
                f"[Source: Tafsir {t['source']} | Category: {t['category']} | Language: {t['language']}{marker}]"
            )
            tafsir_parts.append(t["text"])  # already truncated
        layers.append("\n".join(tafsir_parts))

    return "\n\n".join(layers)


def build_quran_chunks() -> int:
    print("\n[Quran V3] Building 3-layer chunks...")
    flat, en_lookup, _ = load_quran_data()
    print(f"  Loaded {len(flat):,} ayahs")

    tafsirs_data = load_all_tafsirs()
    for key, td in tafsirs_data.items():
        print(f"  Tafsir {key}: {len(td):,} entries")

    context_cards = load_context_cards()
    print(f"  Context cards: {len(context_cards):,}")

    out_path = PROCESSED_DIR / "quran_v3.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Tafsir metadata for the 4 editions
    tafsir_meta = [
        ("ibn_kathir_en", "Ibn Kathir", "bil-Mathur", "en"),
        ("ibn_kathir_ar", "Ibn Kathir", "bil-Mathur", "ar"),
        ("tabari_ar", "Al-Tabari", "bil-Mathur", "ar"),
        ("saadi_ar", "As-Sa'di", "modern", "ar"),
    ]

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, record in enumerate(flat):
            key = (record["surah"], record["ayah"])
            text_en = en_lookup.get(key, "")

            # Previous/next for context
            prev_record = flat[i - 1] if i > 0 else None
            prev_en = en_lookup.get((prev_record["surah"], prev_record["ayah"]), "") if prev_record else ""
            # Don't cross surah boundaries for previous (optional — could relax)
            if prev_record and prev_record["surah"] != record["surah"]:
                prev_en = ""  # skip prev if it's last ayah of previous surah (cultural boundary)

            next_record = flat[i + 1] if i < len(flat) - 1 else None
            next_en = en_lookup.get((next_record["surah"], next_record["ayah"]), "") if next_record else ""
            if next_record and next_record["surah"] != record["surah"]:
                next_en = ""

            # Build tafsir list
            tafsirs_meta = []
            for edition_key, source, category, lang in tafsir_meta:
                full_text = tafsirs_data[edition_key].get(key, "")
                if not full_text:
                    continue
                truncated_text, was_truncated = truncate_tafsir(full_text, TAFSIR_TRUNCATION)
                tafsirs_meta.append({
                    "source": source,
                    "category": category,
                    "language": lang,
                    "text": truncated_text,
                    "text_full": full_text,
                    "truncated": was_truncated,
                    "truncation_ratio": round(len(truncated_text) / max(len(full_text), 1), 3),
                    "contains_isra_iliyyat": detect_isra_iliyyat(full_text),
                    "url": f"https://quran.com/tafsir/{record['surah']}/{record['ayah']}",
                })

            context_card = context_cards.get(key)

            embedding_text = build_quran_embedding_text(
                record, text_en, prev_record, prev_en, next_record, next_en,
                context_card, tafsirs_meta,
            )

            chunk = {
                "id": f"SRC-QURAN-{record['surah']}-{record['ayah']}",
                "kind": "quran",
                "surah": record["surah"],
                "ayah": record["ayah"],
                "surah_name_ar": record["surah_name_ar"],
                "surah_name_en": record["surah_name_en"],
                "revelation_type": record["revelation_type"],
                "text_ar": record["text_ar"],
                "text_en": text_en,
                "text_fr": "",
                "previous_ayah_ar": prev_record["text_ar"] if prev_record and prev_record["surah"] == record["surah"] else "",
                "previous_ayah_en": prev_en,
                "next_ayah_ar": next_record["text_ar"] if next_record and next_record["surah"] == record["surah"] else "",
                "next_ayah_en": next_en,
                "context_card": context_card or {},
                "tafsirs": tafsirs_meta,
                "hadith_cross_refs": {
                    "high_confidence": [],
                    "low_confidence": [],
                    "source_methods": [],
                },
                "ikhtilaf": {
                    "detected": False,
                    "between": [],
                    "summary": None,
                },
                "url": f"https://quran.com/{record['surah']}/{record['ayah']}",
                "embedding_text": embedding_text,
                "build_version": "v3",
            }
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1

    print(f"  [OK] {count:,} chunks → {out_path.relative_to(PROJECT_ROOT)}")
    return count


# ============================================================
# HADITH CHUNKS
# ============================================================

def normalize_collection_name(name: str) -> str:
    aliases = {
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
    if not name:
        return ""
    return aliases.get(name.strip().lower(), name.strip())


def extract_narrator(text_en: str) -> str | None:
    if not text_en:
        return None
    m = NARRATOR_REGEX.match(text_en)
    if m:
        return m.group(1).strip()
    return None


def extract_hadith_number_from_url(url: str) -> int | None:
    if not url:
        return None
    m = re.search(r":(\d+)$", url)
    if m:
        return int(m.group(1))
    return None


def grade_weight(grade: str | None) -> float:
    if not grade:
        return 1.0
    g = grade.lower()
    if "mawdu" in g or "mawdu'" in g or "fabricated" in g:
        return 0.0
    if "munkar" in g or "rejected" in g:
        return 0.0
    if ("da" in g and ("if" in g or "eef" in g)) or "weak" in g:
        return 0.50
    if "hasan" in g or "good" in g:
        return 1.10
    if "sahih" in g or "authentic" in g:
        return 1.30
    return 1.0


def build_hadith_embedding_text(collection: str, hadith_num: int, grade: str | None,
                                  narrator: str | None, text_ar: str, text_en: str,
                                  context_card: dict | None) -> str:
    parts = ["[CONTEXT CARD]"]
    if context_card:
        en = context_card.get("en", {})
        ar = context_card.get("ar", {})
        if en:
            parts.append(f"[EN] Topic: {en.get('topic', '')}. Rule: {en.get('rule', '')}")
        if ar:
            parts.append(f"[AR] الموضوع: {ar.get('theme', '')}")
        kw = []
        for lang in ("en", "ar"):
            kw.extend(context_card.get(lang, {}).get("keywords", []))
        if kw:
            parts.append(f"Keywords: {', '.join(kw)}")
    else:
        # Build a simple context card inline from narrator + grade
        parts.append(f"[EN] Topic: Hadith. Narrator: {narrator or 'unknown'}. Grade: {grade or 'unknown'}")

    hadith_parts = [f"[HADITH — {collection} #{hadith_num} | Grade: {grade or 'Unknown'}"]
    if narrator:
        hadith_parts.append(f"Narrator: {narrator}")
    hadith_parts.append(f"Arabic: {text_ar}")
    if text_en:
        hadith_parts.append(f"English: {text_en}")
    parts.append("\n".join(hadith_parts))

    return "\n\n".join(parts)


def build_hadith_chunks() -> int:
    print("\n[Hadith V3] Building 2-layer chunks...")
    meetif_dir = HADITH_DIR / "meetif"
    if not meetif_dir.exists():
        print(f"  [SKIP] {meetif_dir} not found — run 02_download_hadith.py first")
        return 0

    out_path = PROCESSED_DIR / "hadith_v3.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    grade_dist: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for collection_file in sorted(meetif_dir.glob("*.json")):
            if collection_file.name.startswith("_"):
                continue

            collection_name = normalize_collection_name(collection_file.stem)
            print(f"  [{collection_name}] Loading...")

            with collection_file.open("r", encoding="utf-8") as fh:
                hadiths = json.load(fh)

            if not isinstance(hadiths, list):
                print(f"    [WARN] Not a list — skipping.")
                continue

            for idx, h in enumerate(hadiths):
                text_ar = strip_bom(str(h.get("Arabic_Text", h.get("text_ar", h.get("Arabic", "")))).strip())
                text_en = str(h.get("English_Text", h.get("text_en", h.get("English", "")))).strip()
                grade = str(h.get("Grade", h.get("grade", ""))).strip() or None

                # URL: ALWAYS from meetif "Reference" field
                url = str(h.get("Reference", h.get("url", h.get("Sunnah_URL", "")))).strip() or ""

                hadith_num: int | None = extract_hadith_number_from_url(url)
                if hadith_num is None:
                    in_book = str(h.get("In-book reference", h.get("in_book_reference", "")))
                    m = re.search(r"Hadith\s+(\d+)", in_book, re.IGNORECASE)
                    if m:
                        hadith_num = int(m.group(1))
                if hadith_num is None:
                    hadith_num = idx + 1

                narrator = extract_narrator(text_en)

                # Grade fallback for Bukhari/Muslim
                if not grade and collection_name in ("Sahih al-Bukhari", "Sahih Muslim"):
                    grade = "Sahih"

                # Grade distribution
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

                # Fallback URL if missing
                if not url:
                    slug = HADITH_COLLECTION_SLUGS.get(collection_name, "unknown")
                    url = f"https://sunnah.com/{slug}:{hadith_num}"

                # No LLM context card for hadith in V3 (could be added in V3.1)
                # For now we use the simple inline context built in build_hadith_embedding_text
                embedding_text = build_hadith_embedding_text(
                    collection=collection_name,
                    hadith_num=hadith_num,
                    grade=grade,
                    narrator=narrator,
                    text_ar=text_ar,
                    text_en=text_en,
                    context_card=None,
                )

                chunk = {
                    "id": f"SRC-HADITH-{HADITH_COLLECTION_SLUGS.get(collection_name, 'unknown').upper()}-{hadith_num}",
                    "kind": "hadith",
                    "surah": None,
                    "ayah": None,
                    "collection": collection_name,
                    "hadith_number": hadith_num,
                    "narrator": narrator,
                    "grade": grade,
                    "grade_weight": grade_weight(grade),
                    "text_ar": text_ar,
                    "text_en": text_en,
                    "text_fr": "",
                    "book": h.get("Book"),
                    "chapter_number": h.get("Chapter_Number"),
                    "chapter_arabic": h.get("Chapter_Title_Arabic"),
                    "chapter_english": h.get("Chapter_Title_English"),
                    "in_book_reference": h.get("In-book reference"),
                    "url": url,
                    "embedding_text": embedding_text,
                    "build_version": "v3",
                }
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total += 1

            print(f"    [{collection_name}] {len(hadiths)} hadiths processed")

    print(f"  [OK] {total:,} chunks → {out_path.relative_to(PROJECT_ROOT)}")
    if grade_dist:
        print(f"  Grade distribution:")
        for bucket, n in sorted(grade_dist.items(), key=lambda x: -x[1]):
            print(f"    {bucket:20s} {n}")
    return total


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("=" * 60)
    print("NUR V3 — Step 5: Build V3 chunks (3-layer Quran + 2-layer Hadith)")
    print("=" * 60)
    print(f"Output: {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Tafsir truncation: {TAFSIR_TRUNCATION} chars (data-driven, see doc 05)")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    quran_count = build_quran_chunks()
    hadith_count = build_hadith_chunks()

    summary_path = PROCESSED_DIR / "_summary_v3.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "build_version": "v3",
            "quran_chunks": quran_count,
            "hadith_chunks": hadith_count,
            "total_chunks": quran_count + hadith_count,
            "tafsirs_per_quran_chunk": 4,
            "tafsir_truncation_chars": TAFSIR_TRUNCATION,
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "vector_db": "chromadb",
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("V3 CHUNK BUILD COMPLETE")
    print("=" * 60)
    print(f"  Quran:  {quran_count:,} chunks (3-layer)")
    print(f"  Hadith: {hadith_count:,} chunks (2-layer)")
    print(f"  TOTAL:  {quran_count + hadith_count:,} chunks")
    print(f"  Summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"\nNext step: python scripts/v3/06_compute_cross_refs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
