"""
NUR V3 — Step 6: Compute Quran → Hadith cross-references

For each Quran chunk, parses its 4 tafsirs (Ibn Kathir EN + AR, Tabari, Sa'di)
and extracts hadith citations (e.g. "It is reported in Bukhari that... #1234").

Updates data/processed/quran_v3.jsonl in-place by populating:
  chunk["hadith_cross_refs"]["high_confidence"] = [list of SRC-HADITH-...]
  chunk["hadith_cross_refs"]["source_methods"] = ["tafsir_parsing"]

Then validates each ref against data/processed/hadith_v3.jsonl IDs to ensure
we only reference hadiths that actually exist in our DB.

Usage (local):
  python scripts/v3/06_compute_cross_refs.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import PROCESSED_DIR  # noqa: E402

QURAN_V3_PATH = PROCESSED_DIR / "quran_v3.jsonl"
HADITH_V3_PATH = PROCESSED_DIR / "hadith_v3.jsonl"
OUTPUT_PATH = PROCESSED_DIR / "quran_v3.jsonl"  # update in-place

# Patterns: match "Bukhari 123", "Sahih Muslim #456", "Abu Dawud 789", etc.
# Tafsirs cite collections in many ways:
#   - "It is reported in Bukhari 1234" → Bukhari + number
#   - "Al-Bukhari recorded that..." → Bukhari, no number (skip, can't link without num)
#   - "The Two Sahihs recorded" → skip (ambiguous)
#   - "Sahih Muslim #456" → Muslim + number
#   - "In Hadith number 1234" → skip (no collection)
# We require BOTH collection name AND number in proximity (within 80 chars).
COLLECTION_PATTERNS = [
    # (regex with capture group for number, slug)
    # Bukhari: "Bukhari 1234", "al-Bukhari 1234", "Sahih Bukhari 1234", "Bukhari recorded in book 1234"
    (re.compile(
        r"\b(?:Sahih\s+)?(?:al[-\s])?Bukhari['’]?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "BUKHARI"),
    # Muslim: "Muslim 1234", "Sahih Muslim 1234"
    (re.compile(
        r"\b(?:Sahih\s+)?Muslim['’]?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "MUSLIM"),
    # Abi Dawud / Abu Dawud
    (re.compile(
        r"\b(?:Sunan\s+)?(?:Abi\s+Dawud|Abu\s+Dawud|AbuDawud)['’]?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "ABUDAWUD"),
    # Tirmidhi
    (re.compile(
        r"\b(?:Jami`?\s+)?(?:at[-\s])?Tirmidhi['’]?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "TIRMIDHI"),
    # Nasa'i
    (re.compile(
        r"\b(?:Sunan\s+)?(?:an[-\s])?Nasa['’]?i['’]?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "NASAI"),
    # Ibn Majah
    (re.compile(
        r"\b(?:Sunan\s+)?Ibn\s+Majah['’]?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "IBNMAJAH"),
]


def load_hadith_ids() -> set[str]:
    """Load all hadith chunk IDs from hadith_v3.jsonl."""
    ids = set()
    if not HADITH_V3_PATH.exists():
        print(f"  [WARN] {HADITH_V3_PATH} not found — cannot validate refs")
        return ids
    with HADITH_V3_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if "id" in obj:
                    ids.add(obj["id"])
            except json.JSONDecodeError:
                continue
    return ids


def extract_hadith_refs_from_text(text: str) -> list[tuple[str, int]]:
    """Extract list of (slug, hadith_num) tuples found in text.
    Returns deduped list."""
    refs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for regex, slug in COLLECTION_PATTERNS:
        for m in regex.finditer(text):
            num_str = m.group(1)
            if not num_str:
                continue
            try:
                num = int(num_str)
            except ValueError:
                continue
            # Sanity check: hadith numbers in canonical collections are < 8000
            if num < 1 or num > 8000:
                continue
            key = (slug, num)
            if key not in seen:
                seen.add(key)
                refs.append(key)
    return refs


def build_source_id(slug: str, hadith_num: int) -> str:
    return f"SRC-HADITH-{slug}-{hadith_num}"


def main() -> int:
    print("=" * 60)
    print("NUR V3 — Step 6: Compute Quran → Hadith cross-references")
    print("=" * 60)

    if not QURAN_V3_PATH.exists():
        print(f"[FATAL] {QURAN_V3_PATH} not found — run 05_build_chunks.py first")
        return 1

    print("\nLoading hadith IDs for validation...")
    valid_hadith_ids = load_hadith_ids()
    print(f"  {len(valid_hadith_ids):,} hadith IDs loaded")

    print("\nProcessing Quran chunks...")
    chunks_out = []
    total_refs_found = 0
    chunks_with_refs = 0

    with QURAN_V3_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                chunk = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Concatenate all 4 tafsir full texts for parsing
            all_tafsir_text = " ".join(
                t.get("text_full", "") for t in chunk.get("tafsirs", [])
            )

            refs = extract_hadith_refs_from_text(all_tafsir_text)
            valid_refs = []
            for slug, num in refs:
                source_id = build_source_id(slug, num)
                if source_id in valid_hadith_ids:
                    valid_refs.append(source_id)
                # else: skip — the cited hadith isn't in our DB

            # Dedupe while preserving order
            seen = set()
            deduped_refs = []
            for r in valid_refs:
                if r not in seen:
                    seen.add(r)
                    deduped_refs.append(r)

            chunk["hadith_cross_refs"] = {
                "high_confidence": deduped_refs,
                "low_confidence": [],
                "source_methods": ["tafsir_parsing"] if deduped_refs else [],
            }

            if deduped_refs:
                chunks_with_refs += 1
                total_refs_found += len(deduped_refs)

            chunks_out.append(chunk)

    # Write back to the same file
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks_out:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n  [OK] Cross-refs computed")
    print(f"  Chunks with ≥1 cross-ref: {chunks_with_refs:,} / {len(chunks_out):,}")
    print(f"  Total cross-refs found:   {total_refs_found:,}")
    avg = total_refs_found / max(chunks_with_refs, 1)
    print(f"  Avg refs per linked chunk: {avg:.1f}")

    # Save a small summary
    summary_path = PROCESSED_DIR / "_cross_refs_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "chunks_total": len(chunks_out),
            "chunks_with_refs": chunks_with_refs,
            "total_refs": total_refs_found,
            "method": "tafsir_parsing",
            "validated_against": "hadith_v3.jsonl IDs",
        }, f, ensure_ascii=False, indent=2)

    print(f"  Summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"\nNext step: python scripts/v3/07_embed_and_index.py (on Lightning AI L40S)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
