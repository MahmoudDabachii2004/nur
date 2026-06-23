"""
NUR V3 — Step 5: Build V3 chunks (3 collections: Quran + Tafsir + Hadith)

ARCHITECTURE CHANGE (DEC-042):
  - quran_v3: verse + context card ONLY (no tafsir in embedding) — ~300 tokens
  - tafsir_v3: each tafsir SEPARATE with parent_verse_id — ~200 tokens
  - hadith_v3: hadith + grade — ~400 tokens

This fixes:
  1. Token length problem (all chunks < 512 tokens, BGE-M3 default works)
  2. Tafsir orphelin (tafsir_v3 has parent_verse_id linking to verse)
  3. Inter-collection competition (retrieval is SEQUENTIAL, not parallel)
  4. Semantic bridge (tafsir_v3 embeds full tafsir text → "smoking" matches "self-harm")

Reads:
  data/quran/quran-uthmani.json
  data/quran/en.sahih.json
  data/tafsir/v3/{ibn_kathir_en,ibn_kathir_ar,tabari_ar,saadi_ar}/*.json
  data/hadith/meetif/*.json
  data/processed/context_cards.jsonl

Writes:
  data/processed/quran_v3.jsonl   (6,236 chunks — verse + context card, tafsirs in metadata)
  data/processed/tafsir_v3.jsonl  (~25,000 chunks — 1 tafsir per chunk, parent_verse_id)
  data/processed/hadith_v3.jsonl  (33,738 chunks — hadith + grade)
  data/processed/_summary_v3.json

Usage:
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
PREV_NEXT_MAX_CHARS = 200
ISRA_ILIYYAT_PATTERNS = [
    r"\bKa['']b al-Ahbar\b",
    r"\bWahb ibn Munabbih\b",
    r"\bAbdullah ibn Salam\b",
    r"\bIsra['']iliyyat\b",
    r"\bBanu Isra['']il (?:narrated|said|reported)\b",
    r"\bAccording to (?:Jewish|Christian) tradition\b",
    r"\bPeople of the Book (?:said|narrated|reported)\b",
    r"\bThis is from the Isra['']iliyyat\b",
]
ISRA_ILIYYAT_REGEX = re.compile("|".join(ISRA_ILIYYAT_PATTERNS), re.IGNORECASE)

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


def detect_isra_iliyyat(text: str) -> bool:
    return bool(ISRA_ILIYYAT_REGEX.search(text))


# ============================================================
# QURAN CHUNKS (verse + context card ONLY, no tafsir in embedding)
# ============================================================

def load_quran_data():
    ar_path = QURAN_DIR / "quran-uthmani.json"
    en_path = QURAN_DIR / "en.sahih.json"

    with ar_path.open("r", encoding="utf-8") as f:
        ar_data = json.load(f)

    en_lookup = {}
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
    return flat, en_lookup


def load_tafsir_edition(edition_dir):
    out = {}
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


def load_all_tafsirs():
    v3_dir = TAFSIR_DIR / "v3"
    return {
        "ibn_kathir_en": load_tafsir_edition(v3_dir / "ibn_kathir_en"),
        "ibn_kathir_ar": load_tafsir_edition(v3_dir / "ibn_kathir_ar"),
        "tabari_ar": load_tafsir_edition(v3_dir / "tabari_ar"),
        "saadi_ar": load_tafsir_edition(v3_dir / "saadi_ar"),
    }


def load_context_cards():
    path = PROCESSED_DIR / "context_cards.jsonl"
    cards = {}
    if not path.exists():
        print(f"  [WARN] No context_cards.jsonl found")
        return cards
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                cards[(obj["surah"], obj["ayah"])] = obj["context_card"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cards


def build_quran_embedding_text(record, text_en, prev_record, prev_en,
                                next_record, next_en, context_card):
    """Build embedding_text: Context Card + Word of Allah ONLY (no tafsirs)."""
    layers = []

    # Layer 1: Context Card
    card_parts = ["[CONTEXT CARD]"]
    if context_card:
        fr = context_card.get("fr", {})
        en = context_card.get("en", {})
        ar = context_card.get("ar", {})
        if fr:
            card_parts.append(f"[FR] Theme: {fr.get('theme', '')}. Rule: {fr.get('rule', '')}")
        if en:
            card_parts.append(f"[EN] Topic: {en.get('topic', '')}. Rule: {en.get('rule', '')}")
        if ar:
            card_parts.append(f"[AR] Theme: {ar.get('theme', '')}")
        all_kw = []
        for lang in ("fr", "en", "ar"):
            all_kw.extend(context_card.get(lang, {}).get("keywords", []))
        if all_kw:
            card_parts.append(f"Keywords: {', '.join(str(k) for k in all_kw)}")
    layers.append("\n".join(card_parts))

    # Layer 2: Word of Allah (PURE)
    verse_parts = ["[WORD OF ALLAH]"]
    verse_parts.append(
        f"Quran {record['surah']}:{record['ayah']} | "
        f"{record['surah_name_en']} ({record['surah_name_ar']}) | "
        f"Revelation: {record['revelation_type']}"
    )
    verse_parts.append(f"Arabic: {record['text_ar']}")
    if text_en:
        verse_parts.append(f"English: {text_en}")
    if prev_record and prev_en and prev_record["surah"] == record["surah"]:
        prev_short = prev_en[:PREV_NEXT_MAX_CHARS] + ("..." if len(prev_en) > PREV_NEXT_MAX_CHARS else "")
        verse_parts.append(f"Previous ({prev_record['surah']}:{prev_record['ayah']}): {prev_short}")
    if next_record and next_en and next_record["surah"] == record["surah"]:
        next_short = next_en[:PREV_NEXT_MAX_CHARS] + ("..." if len(next_en) > PREV_NEXT_MAX_CHARS else "")
        verse_parts.append(f"Next ({next_record['surah']}:{next_record['ayah']}): {next_short}")
    layers.append("\n".join(verse_parts))

    return "\n\n".join(layers)


def build_quran_chunks():
    print("\n[Quran V3] Building verse+context_card chunks (NO tafsir in embedding)...")
    flat, en_lookup = load_quran_data()
    print(f"  Loaded {len(flat):,} ayahs")

    tafsirs_data = load_all_tafsirs()
    for key, td in tafsirs_data.items():
        print(f"  Tafsir {key}: {len(td):,} entries")

    context_cards = load_context_cards()
    print(f"  Context cards: {len(context_cards):,}")

    tafsir_meta = [
        ("ibn_kathir_en", "Ibn Kathir", "bil-Mathur", "en"),
        ("ibn_kathir_ar", "Ibn Kathir", "bil-Mathur", "ar"),
        ("tabari_ar", "Al-Tabari", "bil-Mathur", "ar"),
        ("saadi_ar", "As-Sa'di", "modern", "ar"),
    ]

    out_path = PROCESSED_DIR / "quran_v3.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, record in enumerate(flat):
            key = (record["surah"], record["ayah"])
            text_en = en_lookup.get(key, "")

            prev_record = flat[i - 1] if i > 0 else None
            prev_en = en_lookup.get((prev_record["surah"], prev_record["ayah"]), "") if prev_record else ""
            next_record = flat[i + 1] if i < len(flat) - 1 else None
            next_en = en_lookup.get((next_record["surah"], next_record["ayah"]), "") if next_record else ""

            # Build tafsir list for METADATA (full text preserved for LLM)
            tafsirs_meta = []
            for edition_key, source, category, lang in tafsir_meta:
                full_text = tafsirs_data[edition_key].get(key, "")
                if not full_text:
                    continue
                tafsirs_meta.append({
                    "source": source,
                    "category": category,
                    "language": lang,
                    "text_full": full_text,
                    "contains_isra_iliyyat": detect_isra_iliyyat(full_text),
                    "url": f"https://quran.com/tafsir/{record['surah']}/{record['ayah']}",
                })

            context_card = context_cards.get(key)

            # Embedding text = Context Card + Verse ONLY (no tafsirs)
            embedding_text = build_quran_embedding_text(
                record, text_en, prev_record, prev_en, next_record, next_en, context_card
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
                "context_card": context_card or {},
                "tafsirs": tafsirs_meta,
                "hadith_cross_refs": {
                    "high_confidence": [],
                    "low_confidence": [],
                    "source_methods": [],
                },
                "url": f"https://quran.com/{record['surah']}/{record['ayah']}",
                "embedding_text": embedding_text,
                "build_version": "v3",
            }
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            count += 1

    print(f"  [OK] {count:,} chunks -> {out_path.relative_to(PROJECT_ROOT)}")
    return count


# ============================================================
# TAFSIR CHUNKS (each tafsir SEPARATE, with parent_verse_id)
# ============================================================

def build_tafsir_chunks():
    print("\n[Tafsir V3] Building separate tafsir chunks with parent_verse_id...")
    tafsirs_data = load_all_tafsirs()

    tafsir_meta = [
        ("ibn_kathir_en", "Ibn Kathir", "bil-Mathur", "en", "IBNKATHIR-EN"),
        ("ibn_kathir_ar", "Ibn Kathir", "bil-Mathur", "ar", "IBNKATHIR-AR"),
        ("tabari_ar", "Al-Tabari", "bil-Mathur", "ar", "TABARI-AR"),
        ("saadi_ar", "As-Sa'di", "modern", "ar", "SAADI-AR"),
    ]

    out_path = PROCESSED_DIR / "tafsir_v3.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for edition_key, source, category, lang, id_slug in tafsir_meta:
            print(f"  [{source} ({lang.upper()})] Processing...")
            for (surah, ayah), full_text in sorted(tafsirs_data[edition_key].items()):
                parent_verse_id = f"SRC-QURAN-{surah}-{ayah}"

                # Embedding text = just the tafsir text with a label header
                embedding_text = (
                    f"[TAFSIR — {source} | {category} | {lang.upper()}]\n"
                    f"Surah {surah}, Ayah {ayah}\n"
                    f"{full_text}"
                )

                # Truncate embedding text if too long (BGE-M3 max 8192 tokens)
                # But keep full text in metadata
                embed_text_for_index = embedding_text
                if len(embed_text_for_index) > 12000:
                    embed_text_for_index = embed_text_for_index[:12000] + "\n...[truncated for embedding]"

                chunk = {
                    "id": f"SRC-TAFSIR-{id_slug}-{surah}-{ayah}",
                    "kind": "tafsir",
                    "source": source,
                    "category": category,
                    "language": lang,
                    "surah": surah,
                    "ayah": ayah,
                    "parent_verse_id": parent_verse_id,
                    "text_full": full_text,
                    "contains_isra_iliyyat": detect_isra_iliyyat(full_text),
                    "url": f"https://quran.com/tafsir/{surah}/{ayah}",
                    "embedding_text": embed_text_for_index,
                    "build_version": "v3",
                }
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                count += 1
            print(f"    {len(tafsirs_data[edition_key]):,} sections")

    print(f"  [OK] {count:,} chunks -> {out_path.relative_to(PROJECT_ROOT)}")
    return count


# ============================================================
# HADITH CHUNKS (same as before, 2-layer)
# ============================================================

def normalize_collection_name(name):
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


def extract_narrator(text_en):
    if not text_en:
        return None
    m = NARRATOR_REGEX.match(text_en)
    if m:
        return m.group(1).strip()
    return None


def extract_hadith_number_from_url(url):
    if not url:
        return None
    m = re.search(r":(\d+)$", url)
    if m:
        return int(m.group(1))
    return None


def grade_weight(grade):
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


def build_hadith_embedding_text(collection, hadith_num, grade, narrator, text_ar, text_en):
    parts = [f"[HADITH - {collection} #{hadith_num} | Grade: {grade or 'Unknown'}"]
    if narrator:
        parts.append(f"Narrator: {narrator}")
    parts.append(f"Arabic: {text_ar}")
    if text_en:
        parts.append(f"English: {text_en}")
    return "\n".join(parts)


def build_hadith_chunks():
    print("\n[Hadith V3] Building 2-layer chunks...")
    meetif_dir = HADITH_DIR / "meetif"
    if not meetif_dir.exists():
        print(f"  [SKIP] {meetif_dir} not found")
        return 0

    out_path = PROCESSED_DIR / "hadith_v3.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    grade_dist = {}
    seen_ids = {}

    with out_path.open("w", encoding="utf-8") as f:
        for collection_file in sorted(meetif_dir.glob("*.json")):
            if collection_file.name.startswith("_"):
                continue

            collection_name = normalize_collection_name(collection_file.stem)
            print(f"  [{collection_name}] Loading...")

            with collection_file.open("r", encoding="utf-8") as fh:
                hadiths = json.load(fh)

            if not isinstance(hadiths, list):
                continue

            for idx, h in enumerate(hadiths):
                text_ar = strip_bom(str(h.get("Arabic_Text", h.get("text_ar", h.get("Arabic", "")))).strip())
                text_en = str(h.get("English_Text", h.get("text_en", h.get("English", "")))).strip()
                grade = str(h.get("Grade", h.get("grade", ""))).strip() or None

                url = str(h.get("Reference", h.get("url", h.get("Sunnah_URL", "")))).strip() or ""

                hadith_num = extract_hadith_number_from_url(url)
                if hadith_num is None:
                    in_book = str(h.get("In-book reference", h.get("in_book_reference", "")))
                    m = re.search(r"Hadith\s+(\d+)", in_book, re.IGNORECASE)
                    if m:
                        hadith_num = int(m.group(1))
                if hadith_num is None:
                    hadith_num = idx + 1

                narrator = extract_narrator(text_en)

                if not grade and collection_name in ("Sahih al-Bukhari", "Sahih Muslim"):
                    grade = "Sahih"

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

                if not url:
                    slug = HADITH_COLLECTION_SLUGS.get(collection_name, "unknown")
                    url = f"https://sunnah.com/{slug}:{hadith_num}"

                # Dedup IDs
                slug = HADITH_COLLECTION_SLUGS.get(collection_name, "unknown").upper()
                base_id = f"SRC-HADITH-{slug}-{hadith_num}"
                if base_id in seen_ids:
                    seen_ids[base_id] += 1
                    chunk_id = f"{base_id}-n{seen_ids[base_id]}"
                else:
                    seen_ids[base_id] = 0
                    chunk_id = base_id

                embedding_text = build_hadith_embedding_text(
                    collection_name, hadith_num, grade, narrator, text_ar, text_en
                )

                chunk = {
                    "id": chunk_id,
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

    print(f"  [OK] {total:,} chunks -> {out_path.relative_to(PROJECT_ROOT)}")
    if grade_dist:
        print(f"  Grade distribution:")
        for bucket, n in sorted(grade_dist.items(), key=lambda x: -x[1]):
            print(f"    {bucket:20s} {n}")
    return total


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("NUR V3 — Step 5: Build 3-collection chunks")
    print("  quran_v3: verse + context card (no tafsir in embedding)")
    print("  tafsir_v3: each tafsir separate, with parent_verse_id")
    print("  hadith_v3: hadith + grade")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    quran_count = build_quran_chunks()
    tafsir_count = build_tafsir_chunks()
    hadith_count = build_hadith_chunks()

    summary_path = PROCESSED_DIR / "_summary_v3.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "build_version": "v3-3collections",
            "quran_chunks": quran_count,
            "tafsir_chunks": tafsir_count,
            "hadith_chunks": hadith_count,
            "total_chunks": quran_count + tafsir_count + hadith_count,
            "architecture": "3 collections: quran_v3 (verse+context), tafsir_v3 (separate, parent_verse_id), hadith_v3",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("V3 CHUNK BUILD COMPLETE")
    print("=" * 60)
    print(f"  Quran:  {quran_count:,} chunks (verse + context card)")
    print(f"  Tafsir: {tafsir_count:,} chunks (separate, parent_verse_id)")
    print(f"  Hadith: {hadith_count:,} chunks (2-layer)")
    print(f"  TOTAL:  {quran_count + tafsir_count + hadith_count:,} chunks")
    print(f"\nNext step: python scripts/v3/06_compute_cross_refs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
