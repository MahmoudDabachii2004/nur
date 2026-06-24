"""
NUR V3 — Step 6: Compute cross-references (Quran → Hadith + Tafsir → Hadith)

Parses tafsir text in BOTH quran_v3.jsonl and tafsir_v3.jsonl to extract
hadith citations (e.g. "It is reported in Bukhari that... #1234").

Updates both files in-place by populating:
  chunk["hadith_cross_refs"]["high_confidence"] = [list of SRC-HADITH-...]
  chunk["hadith_cross_refs"]["source_methods"] = ["tafsir_parsing"]

Usage:
  python3 scripts/v3/06_compute_cross_refs.py
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
TAFSIR_V3_PATH = PROCESSED_DIR / "tafsir_v3.jsonl"
HADITH_V3_PATH = PROCESSED_DIR / "hadith_v3.jsonl"

# Patterns: match "Bukhari 123", "Sahih Muslim #456", "Abu Dawud 789", etc.
COLLECTION_PATTERNS = [
    (re.compile(
        r"\b(?:Sahih\s+)?(?:al[-\s])?Bukhari['']?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "BUKHARI"),
    (re.compile(
        r"\b(?:Sahih\s+)?Muslim['']?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "MUSLIM"),
    (re.compile(
        r"\b(?:Sunan\s+)?(?:Abi\s+Dawud|Abu\s+Dawud|AbuDawud)['']?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "ABUDAWUD"),
    (re.compile(
        r"\b(?:Jami`?\s+)?(?:at[-\s])?Tirmidhi['']?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "TIRMIDHI"),
    (re.compile(
        r"\b(?:Sunan\s+)?(?:an[-\s])?Nasa['']?i['']?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
        re.IGNORECASE), "NASAI"),
    (re.compile(
        r"\b(?:Sunan\s+)?Ibn\s+Majah['']?(?:s)?\b[^#\d]{0,80}#?(\d{1,5})\b",
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
    """Extract list of (slug, hadith_num) tuples found in text."""
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
            if num < 1 or num > 8000:
                continue
            key = (slug, num)
            if key not in seen:
                seen.add(key)
                refs.append(key)
    return refs


def build_source_id(slug: str, hadith_num: int) -> str:
    return f"SRC-HADITH-{slug}-{hadith_num}"


def process_file(file_path: Path, valid_hadith_ids: set, label: str) -> tuple[int, int]:
    """Process a JSONL file and add hadith_cross_refs to each chunk.
    Returns (chunks_with_refs, total_refs)."""
    if not file_path.exists():
        print(f"\n  [{label}] {file_path.name} not found — skipping")
        return 0, 0

    print(f"\n  [{label}] Processing {file_path.name}...")
    chunks_out = []
    chunks_with_refs = 0
    total_refs = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                chunk = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # For Quran chunks: parse all tafsirs in metadata
            # For Tafsir chunks: parse text_full
            if chunk.get("kind") == "quran":
                all_tafsir_text = " ".join(
                    t.get("text_full", "") for t in chunk.get("tafsirs", [])
                )
            elif chunk.get("kind") == "tafsir":
                all_tafsir_text = chunk.get("text_full", "") or chunk.get("text_chunk", "")
            else:
                continue

            refs = extract_hadith_refs_from_text(all_tafsir_text)
            valid_refs = []
            for slug, num in refs:
                source_id = build_source_id(slug, num)
                if source_id in valid_hadith_ids:
                    valid_refs.append(source_id)

            # Dedupe
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
                total_refs += len(deduped_refs)

            chunks_out.append(chunk)

    # Write back
    with file_path.open("w", encoding="utf-8") as f:
        for chunk in chunks_out:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"    {label}: {chunks_with_refs:,} chunks with refs, {total_refs:,} total refs")
    return chunks_with_refs, total_refs


def main() -> int:
    print("=" * 60)
    print("NUR V3 — Step 6: Compute cross-references (Quran + Tafsir → Hadith)")
    print("=" * 60)

    print("\nLoading hadith IDs for validation...")
    valid_hadith_ids = load_hadith_ids()
    print(f"  {len(valid_hadith_ids):,} hadith IDs loaded")

    # Process Quran chunks (parse tafsirs in metadata)
    quran_refs, quran_total = process_file(
        QURAN_V3_PATH, valid_hadith_ids, "Quran"
    )

    # Process Tafsir chunks (parse text_full in each chunk)
    tafsir_refs, tafsir_total = process_file(
        TAFSIR_V3_PATH, valid_hadith_ids, "Tafsir"
    )

    total_chunks_with_refs = quran_refs + tafsir_refs
    total_refs = quran_total + tafsir_total

    print(f"\n{'=' * 50}")
    print(f"CROSS-REFERENCE SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Quran chunks with refs:  {quran_refs:,}")
    print(f"  Tafsir chunks with refs: {tafsir_refs:,}")
    print(f"  Total chunks with refs:  {total_chunks_with_refs:,}")
    print(f"  Total cross-refs found:  {total_refs:,}")

    summary_path = PROCESSED_DIR / "_cross_refs_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "quran_chunks_with_refs": quran_refs,
            "tafsir_chunks_with_refs": tafsir_refs,
            "total_chunks_with_refs": total_chunks_with_refs,
            "total_refs": total_refs,
            "method": "tafsir_parsing",
            "validated_against": "hadith_v3.jsonl IDs",
        }, f, ensure_ascii=False, indent=2)

    print(f"  Summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"\nNext step: zip and upload to Kaggle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
