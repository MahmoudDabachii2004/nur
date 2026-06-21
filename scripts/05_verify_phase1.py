"""
NUR Phase 1 — Step 5: Verify processed data

Quick sanity checks on the JSONL chunks before embedding.

Verifies:
  1. All expected files exist
  2. Chunk counts match expectations (Quran: 6,236, Hadith: ~33,738, Tafsir: ~12,000)
  3. Arabic normalization is working (no diacritics in normalized fields)
  4. Source IDs follow the expected format
  5. URLs are well-formed
  6. Grade distribution looks sane

Usage:
  python scripts/05_verify_phase1.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import PROCESSED_DIR  # noqa: E402

# Diacritics pattern (must be absent from normalized text)
TASHKEEL = re.compile(
    "["
    "\u0610-\u061a"
    "\u064b-\u065f"
    "\u0670"
    "\u06d6-\u06dc"
    "\u06df-\u06e8"
    "\u06ea-\u06ed"
    "]"
)

# Valid source ID formats
SOURCE_ID_PATTERNS = {
    "quran": re.compile(r"^SRC-QURAN-\d+-\d+$"),
    "hadith": re.compile(r"^SRC-HADITH-(BUKHARI|MUSLIM|ABUDAWUD|TIRMIDHI|NASAI|IBNMAJAH|MALIK|AHMAD|DARIMI|UNKNOWN)-\d+$"),
    "tafsir_ar": re.compile(r"^SRC-TAFSIR-AR-\d+-\d+$"),
    "tafsir_en": re.compile(r"^SRC-TAFSIR-EN-\d+-\d+$"),
}


def check_file(name: str, expected_min: int = 0) -> tuple[int, list[dict]]:
    """Load and validate a JSONL file. Returns (count, sample_chunks)."""
    path = PROCESSED_DIR / name
    if not path.exists():
        print(f"  [MISSING] {name}")
        return 0, []

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"  [OK] {name}: {len(chunks):,} chunks")
    if len(chunks) < expected_min:
        print(f"      [WARN] Expected ≥ {expected_min:,}, got {len(chunks):,}")

    return len(chunks), chunks[:3]


def validate_chunk(chunk: dict) -> list[str]:
    """Run validation checks on a single chunk. Returns list of issues."""
    issues = []
    kind = chunk.get("kind", "")

    # 1. Has required fields
    for field in ("id", "kind", "url"):
        if not chunk.get(field):
            issues.append(f"missing/empty: {field}")

    # 1b. Must have AT LEAST ONE normalized text field
    # (tafsir_en has only English; tafsir_ar has only Arabic; quran/hadith have both)
    has_ar = bool(chunk.get("text_ar_normalized", ""))
    has_en = bool(chunk.get("text_en_normalized", ""))
    if not has_ar and not has_en:
        issues.append("missing both text_ar_normalized and text_en_normalized")

    # 2. ID matches pattern
    if kind in SOURCE_ID_PATTERNS:
        if not SOURCE_ID_PATTERNS[kind].match(chunk["id"]):
            issues.append(f"bad ID format: {chunk['id']}")

    # 3. Normalized Arabic has no diacritics
    norm = chunk.get("text_ar_normalized", "")
    if norm and TASHKEEL.search(norm):
        issues.append("normalized Arabic still has diacritics")

    # 4. URL is well-formed
    url = chunk.get("url", "")
    if url and not url.startswith("https://"):
        issues.append(f"bad URL: {url}")

    # 5. Grade weight is in valid range
    gw = chunk.get("grade_weight")
    if gw is not None and not (0.0 <= gw <= 2.0):
        issues.append(f"grade_weight out of range: {gw}")

    return issues


def main() -> int:
    print("=" * 60)
    print("NUR Phase 1 — Step 5: Verify Processed Data")
    print("=" * 60)
    print(f"Input: {PROCESSED_DIR.relative_to(PROJECT_ROOT)}\n")

    if not PROCESSED_DIR.exists():
        print("[FATAL] Processed directory does not exist.")
        print("        Run scripts/04_normalize_and_chunk.py first.")
        return 1

    # Check each file
    print("[1] Checking files and counts...")
    quran_count, quran_sample = check_file("quran.jsonl", expected_min=6200)
    hadith_count, hadith_sample = check_file("hadith.jsonl", expected_min=30000)
    tafsir_ar_count, tafsir_ar_sample = check_file("tafsir_ar.jsonl", expected_min=5000)
    tafsir_en_count, tafsir_en_sample = check_file("tafsir_en.jsonl", expected_min=5000)

    # Validate sample chunks
    print("\n[2] Validating sample chunks...")
    samples = quran_sample + hadith_sample + tafsir_ar_sample + tafsir_en_sample
    issues_total = 0
    for c in samples:
        issues = validate_chunk(c)
        if issues:
            print(f"  [{c.get('id', '???')}] {issues}")
            issues_total += len(issues)
    if issues_total == 0:
        print(f"  [OK] All {len(samples)} sample chunks passed validation.")
    else:
        print(f"  [WARN] {issues_total} issues found in sample chunks.")

    # Hadith grade distribution
    print("\n[3] Hadith grade distribution...")
    if (PROCESSED_DIR / "hadith.jsonl").exists():
        grades = Counter()
        collections = Counter()
        with (PROCESSED_DIR / "hadith.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                collections[c.get("collection", "unknown")] += 1
                g = c.get("grade", "")
                if g:
                    g_lower = g.lower()
                    if "sahih" in g_lower:
                        bucket = "Sahih"
                    elif "hasan" in g_lower:
                        bucket = "Hasan"
                    elif ("da" in g_lower and "if" in g_lower) or "weak" in g_lower:
                        bucket = "Da'if"
                    elif "mawdu" in g_lower or "munkar" in g_lower:
                        bucket = "Mawdu/Munkar"
                    else:
                        bucket = "Other"
                    grades[bucket] += 1

        print("  Collections:")
        for col, count in collections.most_common():
            print(f"    {col:30s} {count:,}")
        print("  Grades:")
        for grade, count in grades.most_common():
            print(f"    {grade:20s} {count:,}")

    # Final summary
    total = quran_count + hadith_count + tafsir_ar_count + tafsir_en_count
    print("\n" + "=" * 60)
    print("PHASE 1 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Total chunks: {total:,}")
    print(f"  Quran:  {quran_count:,}")
    print(f"  Hadith: {hadith_count:,}")
    print(f"  Tafsir AR: {tafsir_ar_count:,}")
    print(f"  Tafsir EN: {tafsir_en_count:,}")
    print(f"\nIf counts look right, next step:")
    print(f"  1. Upload data/processed/*.jsonl to Google Colab")
    print(f"  2. Run colab/embed_nur_colab.py on a T4 GPU")
    print(f"  3. Download data/chroma_db/ and data/sparse/ back to your Mac")
    print(f"  4. Continue to Phase 2 (RAG pipeline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
