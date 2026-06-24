"""
NUR V3 — Masterclass Build (Small-to-Big + Smart Split)

ARCHITECTURE:
  1. QURAN: Split AR/EN (2 child chunks per verse)
     - Child AR: Pure Arabic text (Bismillah stripped) → parent_verse_id
     - Child EN: Pure English text → parent_verse_id
     - Parent: Full metadata (Uthmani, tafsirs, context card, prev/next)
     - Dedup at retrieval: group by parent_verse_id

  2. TAFSIR: Smart split at ~400 tokens (sentence-aware)
     - Child: ~400 token sub-chunk → parent_tafsir_id, parent_verse_id
     - Parent: Full tafsir text in metadata
     - Overlap: 1 sentence between chunks

  3. HADITH: Smart split at ~400 tokens (sentence-aware, only if > 400 tokens)
     - Child: ~400 token sub-chunk → parent_hadith_id
     - Parent: Full hadith text in metadata
     - Grade, narrator, URL preserved on every child

SMART SPLITTER:
  - Splits on sentence boundaries (. ! ? ؟ ۔ » " ) ] })
  - NEVER cuts a sentence in half
  - 1 sentence overlap between chunks (context preservation)
  - Word-boundary fallback for very long single sentences
  - Target: 400 tokens (~1200 chars mixed AR/EN)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import HADITH_DIR, PROCESSED_DIR, QURAN_DIR, TAFSIR_DIR

# ============================================================
# CONSTANTS
# ============================================================
PREV_NEXT_MAX_CHARS = 200
MAX_CHUNK_CHARS = 1200  # ~400 tokens (mixed AR/EN at ~3 chars/token)
OVERLAP_SENTENCES = 1   # Number of sentences to overlap between chunks

BISMILLAH_AR = "\u0628\u0650\u0633\u0652\u0645\u0650 \u0671\u0644\u0644\u0651\u064e\u0647\u0650 \u0671\u0644\u0631\u0651\u064e\u062d\u0652\u0645\u064e\u0670\u0646\u0650 \u0671\u0644\u0631\u0651\u064e\u062d\u0650\u064a\u0645\u0650"

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

HADITH_COLLECTION_SLUGS = {
    "Sahih al-Bukhari": "bukhari",
    "Sahih Muslim": "muslim",
    "Sunan Abi Dawud": "abudawud",
    "Jami` at-Tirmidhi": "tirmidhi",
    "Sunan an-Nasa'i": "nasai",
    "Sunan Ibn Majah": "ibnmajah",
}
NARRATOR_REGEX = re.compile(r"^Narrated\s+([^:]+?):\s", re.IGNORECASE)

# Sentence endings for both AR and EN
SENTENCE_ENDINGS = r'(?<=[.!?؟۔»")\]])\s+'


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def strip_bom(text: str) -> str:
    if not text:
        return ""
    return text.lstrip("\ufeff").lstrip("\ufffe").strip()


def detect_isra_iliyyat(text: str) -> bool:
    return bool(ISRA_ILIYYAT_REGEX.search(text))


def smart_split(text: str, max_chars: int = MAX_CHUNK_CHARS,
                overlap_sentences: int = OVERLAP_SENTENCES) -> list[tuple[str, int, int]]:
    """Split text into sub-chunks respecting sentence boundaries.

    Rules:
      1. If text ≤ max_chars → 1 chunk (no split)
      2. If text > max_chars → split on sentence boundaries
      3. Overlap of N sentences between chunks (context preservation)
      4. Word-boundary fallback for very long single sentences
      5. NEVER cuts a sentence in half

    Returns:
        List of (chunk_text, chunk_index, total_chunks) tuples
    """
    if not text or not text.strip():
        return [(text, 0, 1)]

    if len(text) <= max_chars:
        return [(text, 0, 1)]

    # Split into sentences using multiple delimiters
    # Handles: . ! ? ؟ ۔ » " ) ] (both AR and EN sentence endings)
    sentences = re.split(SENTENCE_ENDINGS, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Fallback: if only 1 sentence and it's too long, split at word boundaries
    if len(sentences) <= 1:
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        for word in words:
            word_len = len(word) + 1
            if current_len + word_len > max_chars and current:
                chunks.append(" ".join(current))
                current = [word]
                current_len = word_len
            else:
                current.append(word)
                current_len += word_len
        if current:
            chunks.append(" ".join(current))
        return [(c, i, len(chunks)) for i, c in enumerate(chunks)]

    # Group sentences into chunks respecting max_chars
    chunks = []
    current_sentences = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # +1 for space

        if current_len + sentence_len > max_chars and current_sentences:
            # Save current chunk
            chunks.append(" ".join(current_sentences))

            # Start new chunk with overlap
            if overlap_sentences > 0 and len(current_sentences) >= overlap_sentences:
                overlap = current_sentences[-overlap_sentences:]
            else:
                overlap = []
            current_sentences = overlap + [sentence]
            current_len = sum(len(s) + 1 for s in current_sentences)
        else:
            current_sentences.append(sentence)
            current_len += sentence_len

    # Don't forget the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return [(c, i, len(chunks)) for i, c in enumerate(chunks)]


# ============================================================
# QURAN CHUNKS (Split AR/EN — Small-to-Big)
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
                en_lookup[(surah["number"], ayah["numberInSurah"])] = ayah.get("text", "")

    flat = []
    for surah in ar_data.get("surahs", []):
        surah_num = surah["number"]
        for ayah in surah.get("ayahs", []):
            ayah_num = ayah["numberInSurah"]
            text_ar_full = strip_bom(ayah.get("text", ""))

            # Strip Bismillah from embedding text (not from metadata)
            text_ar_clean = text_ar_full
            if surah_num != 1 and surah_num != 9 and ayah_num == 1:
                if text_ar_clean.startswith(BISMILLAH_AR):
                    text_ar_clean = text_ar_clean[len(BISMILLAH_AR):].strip()

            flat.append({
                "surah": surah_num,
                "ayah": ayah_num,
                "surah_name_ar": surah.get("name", ""),
                "surah_name_en": surah.get("englishName", ""),
                "revelation_type": surah.get("revelationType", ""),
                "text_ar_clean": text_ar_clean,      # For embedding (no Bismillah)
                "text_ar_full": text_ar_full,         # For LLM + verification (with Bismillah)
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
        "jalalayn_en": load_tafsir_edition(v3_dir / "jalalayn_en"),
        "jalalayn_ar": load_tafsir_edition(v3_dir / "jalalayn_ar"),
        "mukhtasar_en": load_tafsir_edition(v3_dir / "mukhtasar_en"),
        "mukhtasar_ar": load_tafsir_edition(v3_dir / "mukhtasar_ar"),
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


def build_quran_chunks():
    """Build Quran chunks: 2 child chunks per verse (AR pure + EN pure)."""
    print("\n[Quran V3] Building split AR/EN chunks (Small-to-Big)...")
    flat, en_lookup = load_quran_data()
    print(f"  Loaded {len(flat):,} ayahs")

    tafsirs_data = load_all_tafsirs()
    context_cards = load_context_cards()
    print(f"  Context cards: {len(context_cards):,}")

    tafsir_meta = [
        ("ibn_kathir_en", "Ibn Kathir", "bil-Mathur", "en"),
        ("ibn_kathir_ar", "Ibn Kathir", "bil-Mathur", "ar"),
        ("jalalayn_en", "Al-Jalalayn", "classical", "en"),
        ("jalalayn_ar", "Al-Jalalayn", "classical", "ar"),
        ("mukhtasar_en", "Al-Mukhtasar", "modern", "en"),
        ("mukhtasar_ar", "Al-Mukhtasar", "modern", "ar"),
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

            # Build tafsir list for PARENT metadata
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

            # Build PARENT metadata (shared by both AR and EN children)
            parent_verse_id = f"SRC-QURAN-{record['surah']}-{record['ayah']}"

            prev_ayah_ar = prev_record.get("text_ar_full", "") if prev_record and prev_record["surah"] == record["surah"] else ""
            next_ayah_ar = next_record.get("text_ar_full", "") if next_record and next_record["surah"] == record["surah"] else ""

            parent_meta = {
                "parent_verse_id": parent_verse_id,
                "kind": "quran",
                "surah": record["surah"],
                "ayah": record["ayah"],
                "surah_name_ar": record["surah_name_ar"],
                "surah_name_en": record["surah_name_en"],
                "revelation_type": record["revelation_type"],
                "text_ar": record["text_ar_full"],  # Full Uthmani WITH Bismillah
                "text_en": text_en,
                "text_fr": "",
                "previous_ayah_ar": prev_ayah_ar,
                "previous_ayah_en": prev_en if prev_record and prev_record["surah"] == record["surah"] else "",
                "next_ayah_ar": next_ayah_ar,
                "next_ayah_en": next_en if next_record and next_record["surah"] == record["surah"] else "",
                "context_card": context_card or {},
                "tafsirs": tafsirs_meta,
                "hadith_cross_refs": {"high_confidence": [], "low_confidence": [], "source_methods": []},
                "url": f"https://quran.com/{record['surah']}/{record['ayah']}",
                "build_version": "v3-masterclass",
            }

            # CHILD 1: Pure Arabic (Bismillah stripped)
            if record["text_ar_clean"]:
                chunk_ar = {
                    **parent_meta,
                    "id": f"{parent_verse_id}-AR",
                    "language": "ar",
                    "embedding_text": record["text_ar_clean"],  # PURE Arabic
                }
                f.write(json.dumps(chunk_ar, ensure_ascii=False) + "\n")
                count += 1

            # CHILD 2: Pure English
            if text_en:
                chunk_en = {
                    **parent_meta,
                    "id": f"{parent_verse_id}-EN",
                    "language": "en",
                    "embedding_text": text_en,  # PURE English
                }
                f.write(json.dumps(chunk_en, ensure_ascii=False) + "\n")
                count += 1

    print(f"  [OK] {count:,} child chunks -> {out_path.relative_to(PROJECT_ROOT)}")
    print(f"       ({count // 2:,} verses × 2 languages)")
    return count


# ============================================================
# TAFSIR CHUNKS (Smart Split — Parent-Child)
# ============================================================

def build_tafsir_chunks():
    """Build Tafsir chunks: smart split at ~400 tokens."""
    print("\n[Tafsir V3] Building smart-split chunks (Parent-Child)...")
    tafsirs_data = load_all_tafsirs()

    tafsir_meta = [
        ("ibn_kathir_en", "Ibn Kathir", "bil-Mathur", "en", "IBNKATHIR-EN"),
        ("ibn_kathir_ar", "Ibn Kathir", "bil-Mathur", "ar", "IBNKATHIR-AR"),
        ("jalalayn_en", "Al-Jalalayn", "classical", "en", "JALALAYN-EN"),
        ("jalalayn_ar", "Al-Jalalayn", "classical", "ar", "JALALAYN-AR"),
        ("mukhtasar_en", "Al-Mukhtasar", "modern", "en", "MUKHTASAR-EN"),
        ("mukhtasar_ar", "Al-Mukhtasar", "modern", "ar", "MUKHTASAR-AR"),
    ]

    out_path = PROCESSED_DIR / "tafsir_v3.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    split_count = 0
    not_split_count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for edition_key, source, category, lang, id_slug in tafsir_meta:
            print(f"  [{source} ({lang.upper()})] Processing...")
            for (surah, ayah), full_text in sorted(tafsirs_data[edition_key].items()):
                parent_verse_id = f"SRC-QURAN-{surah}-{ayah}"
                parent_tafsir_id = f"SRC-TAFSIR-{id_slug}-{surah}-{ayah}"

                # Smart split the tafsir
                sub_chunks = smart_split(full_text, MAX_CHUNK_CHARS, OVERLAP_SENTENCES)

                if len(sub_chunks) > 1:
                    split_count += 1
                else:
                    not_split_count += 1

                for chunk_text, chunk_idx, total_chunks in sub_chunks:
                    child_id = f"{parent_tafsir_id}-part{chunk_idx + 1}" if total_chunks > 1 else parent_tafsir_id

                    chunk = {
                        "id": child_id,
                        "kind": "tafsir",
                        "source": source,
                        "category": category,
                        "language": lang,
                        "surah": surah,
                        "ayah": ayah,
                        "parent_verse_id": parent_verse_id,
                        "parent_tafsir_id": parent_tafsir_id,
                        "text_chunk": chunk_text,          # The specific sub-chunk (for highlighting)
                        "text_full": full_text,             # Full tafsir (for LLM)
                        "chunk_index": chunk_idx,
                        "total_chunks": total_chunks,
                        "contains_isra_iliyyat": detect_isra_iliyyat(full_text),
                        "url": f"https://quran.com/tafsir/{surah}/{ayah}",
                        "embedding_text": f"[Tafsir {source} - Surah {surah}:{ayah}]\n{chunk_text}",
                        "build_version": "v3-masterclass",
                    }
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    count += 1

            print(f"    {len(tafsirs_data[edition_key]):,} original tafsirs processed")

    print(f"  [OK] {count:,} child chunks -> {out_path.relative_to(PROJECT_ROOT)}")
    print(f"       Split: {split_count:,} | Not split: {not_split_count:,}")
    return count


# ============================================================
# HADITH CHUNKS (Smart Split — Parent-Child)
# ============================================================

def normalize_collection_name(name):
    aliases = {
        "bukhari": "Sahih al-Bukhari", "sahih bukhari": "Sahih al-Bukhari", "sahih al-bukhari": "Sahih al-Bukhari",
        "muslim": "Sahih Muslim", "sahih muslim": "Sahih Muslim",
        "abudawud": "Sunan Abi Dawud", "abu dawud": "Sunan Abi Dawud", "sunan abi dawud": "Sunan Abi Dawud",
        "tirmidhi": "Jami` at-Tirmidhi", "jami` at-tirmidhi": "Jami` at-Tirmidhi",
        "nasai": "Sunan an-Nasa'i", "sunan an-nasa'i": "Sunan an-Nasa'i",
        "ibnmajah": "Sunan Ibn Majah", "ibn majah": "Sunan Ibn Majah", "sunan ibn majah": "Sunan Ibn Majah",
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


def build_hadith_chunks():
    """Build Hadith chunks: smart split at ~400 tokens (only if > 400)."""
    print("\n[Hadith V3] Building smart-split chunks (Parent-Child)...")
    meetif_dir = HADITH_DIR / "meetif"
    if not meetif_dir.exists():
        print(f"  [SKIP] {meetif_dir} not found")
        return 0

    out_path = PROCESSED_DIR / "hadith_v3.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    split_count = 0
    not_split_count = 0
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
                    parent_hadith_id = f"{base_id}-n{seen_ids[base_id]}"
                else:
                    seen_ids[base_id] = 0
                    parent_hadith_id = base_id

                # Build full embedding text (AR + EN combined for hadith)
                full_embed = f"[Hadith {collection_name} #{hadith_num}]\n"
                if narrator:
                    full_embed += f"Narrator: {narrator}\n"
                if text_ar:
                    full_embed += f"Arabic: {text_ar}\n"
                if text_en:
                    full_embed += f"English: {text_en}"

                # Smart split the hadith
                sub_chunks = smart_split(full_embed, MAX_CHUNK_CHARS, OVERLAP_SENTENCES)

                if len(sub_chunks) > 1:
                    split_count += 1
                else:
                    not_split_count += 1

                # Parent metadata (shared by all children)
                parent_meta = {
                    "kind": "hadith",
                    "parent_hadith_id": parent_hadith_id,
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
                    "build_version": "v3-masterclass",
                }

                for chunk_text, chunk_idx, total_chunks in sub_chunks:
                    child_id = f"{parent_hadith_id}-part{chunk_idx + 1}" if total_chunks > 1 else parent_hadith_id

                    chunk = {
                        **parent_meta,
                        "id": child_id,
                        "text_chunk": chunk_text,
                        "text_full": full_embed,
                        "chunk_index": chunk_idx,
                        "total_chunks": total_chunks,
                        "embedding_text": chunk_text,
                    }
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total += 1

            print(f"    [{collection_name}] {len(hadiths)} hadiths processed")

    print(f"  [OK] {total:,} child chunks -> {out_path.relative_to(PROJECT_ROOT)}")
    print(f"       Split: {split_count:,} | Not split: {not_split_count:,}")
    if grade_dist:
        print(f"  Grade distribution:")
        for bucket, n in sorted(grade_dist.items(), key=lambda x: -x[1]):
            print(f"    {bucket:20s} {n}")
    return total


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("NUR V3 — MASTERCLASS BUILD (Small-to-Big + Smart Split)")
    print("=" * 70)
    print("  Quran:  Split AR/EN (2 child chunks per verse, pure text)")
    print("  Tafsir: Smart split at ~400 tokens (sentence-aware)")
    print("  Hadith: Smart split at ~400 tokens (sentence-aware)")
    print(f"  Max chunk chars: {MAX_CHUNK_CHARS} (~400 tokens)")
    print(f"  Overlap: {OVERLAP_SENTENCES} sentence(s)")
    print("=" * 70)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    quran_count = build_quran_chunks()
    tafsir_count = build_tafsir_chunks()
    hadith_count = build_hadith_chunks()

    summary_path = PROCESSED_DIR / "_summary_v3.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "build_version": "v3-masterclass",
            "architecture": "Small-to-Big + Smart Split",
            "quran_chunks": quran_count,
            "tafsir_chunks": tafsir_count,
            "hadith_chunks": hadith_count,
            "total_chunks": quran_count + tafsir_count + hadith_count,
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "overlap_sentences": OVERLAP_SENTENCES,
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("MASTERCLASS BUILD COMPLETE")
    print("=" * 70)
    print(f"  Quran:  {quran_count:,} child chunks ({quran_count // 2:,} verses × 2 langs)")
    print(f"  Tafsir: {tafsir_count:,} child chunks (smart-split)")
    print(f"  Hadith: {hadith_count:,} child chunks (smart-split)")
    print(f"  TOTAL:  {quran_count + tafsir_count + hadith_count:,} child chunks")
    print(f"\nNext step: python scripts/v3/06_compute_cross_refs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
