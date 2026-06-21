"""
NUR Phase 1 — Step 2: Download Hadith

Source: meeAtif/hadith_datasets on HuggingFace
  - 6 collections of Kutub al-Sittah (the 6 canonical Sunni hadith books)
  - 33,738 hadiths total
  - Arabic + English in same entry (not separated — huge win)
  - Includes grades (Sahih, Hasan, Da'if)
  - Includes sunnah.com URLs (no URL construction needed — see ARCHITECTURE.md
    Section 24, Phase 2: "sunnah.com URLs: stockées dans metadata Phase 1,
    NE PAS construire dynamiquement (numérotation décalée)")
  - Direct JSON download via HuggingFace resolve URL (no `datasets` package needed)

Collections:
  - Sahih al-Bukhari   (~7,008 hadiths)
  - Sahih Muslim       (~5,362 hadiths)
  - Sunan Abi Dawud    (~4,590 hadiths)
  - Jami` at-Tirmidhi  (~3,956 hadiths)
  - Sunan an-Nasa'i    (~5,662 hadiths)
  - Sunan Ibn Majah    (~4,341 hadiths)

Output: data/hadith/meetif/{Collection Name}.json

Usage:
  python scripts/02_download_hadith.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import HADITH_DIR  # noqa: E402

MEETIF_DIR = HADITH_DIR / "meetif"

# HuggingFace raw file URL pattern for meeAtif/hadith_datasets
HF_BASE = "https://huggingface.co/datasets/meeAtif/hadith_datasets/resolve/main"

# The 6 canonical collections (Kutub al-Sittah)
COLLECTIONS = [
    "Sahih al-Bukhari",
    "Sahih Muslim",
    "Sunan Abi Dawud",
    "Jami` at-Tirmidhi",
    "Sunan an-Nasa'i",
    "Sunan Ibn Majah",
]


def download_collection(collection_name: str) -> list[dict] | None:
    """Download a single hadith collection as JSON.

    The meeAtif dataset stores each collection as a separate JSON file
    with the exact name (e.g. "Sahih al-Bukhari.json").
    """
    filename = f"{collection_name}.json"
    # URL-encode the filename (spaces → %20, backticks → %60)
    encoded = urllib.parse.quote(filename)
    url = f"{HF_BASE}/{encoded}"

    print(f"  [GET] {url}")
    try:
        resp = requests.get(url, timeout=180)
        if resp.status_code == 404:
            # Fallback: try with %20 only (not full quote)
            alt_url = f"{HF_BASE}/{filename.replace(' ', '%20')}"
            print(f"  [404] Trying alternate URL: {alt_url}")
            resp = requests.get(alt_url, timeout=180)

        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            print(f"  [FAIL] Expected list, got {type(data).__name__}")
            return None

        return data

    except requests.RequestException as e:
        print(f"  [FAIL] {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  [FAIL] JSON parse error: {e}")
        return None


def inspect_schema(hadith: dict) -> None:
    """Print the schema of a hadith entry (for first hadith in collection)."""
    print(f"  Schema keys: {list(hadith.keys())}")
    # Print sample (truncated)
    for k, v in hadith.items():
        s = str(v)
        if len(s) > 80:
            s = s[:77] + "..."
        print(f"    {k}: {s}")


def save_collection(collection_name: str, hadiths: list[dict]) -> Path:
    """Save a hadith collection to disk."""
    MEETIF_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MEETIF_DIR / f"{collection_name}.json"
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(hadiths, f, ensure_ascii=False, indent=2)
    size_kb = filepath.stat().st_size / 1024
    print(f"  [SAVE] {filepath.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")
    return filepath


def summarize_collection(hadiths: list[dict]) -> dict:
    """Compute summary stats for a collection."""
    total = len(hadiths)
    graded = sum(1 for h in hadiths if str(h.get("Grade", "")).strip())

    # Grade distribution
    grade_dist: dict[str, int] = {}
    for h in hadiths:
        grade = str(h.get("Grade", "")).strip()
        if grade:
            # Normalize grade bucket
            g_lower = grade.lower()
            if "sahih" in g_lower:
                bucket = "Sahih"
            elif "hasan" in g_lower:
                bucket = "Hasan"
            elif "da" in g_lower and ("if" in g_lower or "eef" in g_lower):
                bucket = "Da'if"
            elif "mawdu" in g_lower or "munkar" in g_lower:
                bucket = "Mawdu/Munkar"
            else:
                bucket = "Other"
            grade_dist[bucket] = grade_dist.get(bucket, 0) + 1

    return {
        "total": total,
        "graded": graded,
        "grade_distribution": grade_dist,
    }


def main() -> int:
    print("=" * 60)
    print("NUR Phase 1 — Step 2: Download Hadith (Kutub al-Sittah)")
    print("=" * 60)
    print(f"Source: meeAtif/hadith_datasets (HuggingFace)")
    print(f"Output: {MEETIF_DIR.relative_to(PROJECT_ROOT)}\n")

    MEETIF_DIR.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    grand_graded = 0
    all_summaries: dict[str, dict] = {}
    schema_inspected = False

    for i, collection in enumerate(COLLECTIONS, 1):
        print(f"\n[{i}/{len(COLLECTIONS)}] {collection}")
        print("-" * 50)

        filepath = MEETIF_DIR / f"{collection}.json"
        if filepath.exists():
            print(f"  [SKIP] Already exists — loading from disk.")
            with filepath.open("r", encoding="utf-8") as f:
                hadiths = json.load(f)
        else:
            hadiths = download_collection(collection)
            if hadiths is None:
                print(f"  [ERROR] Failed to download {collection}.")
                continue
            save_collection(collection, hadiths)
            time.sleep(1)  # be polite to HuggingFace CDN

        # Inspect schema on first successful download
        if hadiths and not schema_inspected:
            print("\n  Schema inspection (first hadith):")
            inspect_schema(hadiths[0])
            schema_inspected = True

        # Summarize
        summary = summarize_collection(hadiths)
        all_summaries[collection] = summary
        grand_total += summary["total"]
        grand_graded += summary["graded"]

        print(f"\n  Summary:")
        print(f"    Total:  {summary['total']}")
        print(f"    Graded: {summary['graded']} ({summary['graded']/max(summary['total'],1)*100:.1f}%)")
        if summary["grade_distribution"]:
            print(f"    Grade distribution:")
            for bucket, count in sorted(summary["grade_distribution"].items(), key=lambda x: -x[1]):
                print(f"      {bucket:20s} {count}")

    # ----- Grand summary -----
    print("\n" + "=" * 60)
    print("HADITH DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nTotal collections: {len(all_summaries)}")
    print(f"Total hadiths:     {grand_total:,}")
    print(f"Total graded:      {grand_graded:,} ({grand_graded/max(grand_total,1)*100:.1f}%)")

    # Save combined summary
    summary_path = MEETIF_DIR / "_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "meeAtif/hadith_datasets",
                "url": "https://huggingface.co/datasets/meeAtif/hadith_datasets",
                "collections": all_summaries,
                "grand_total": grand_total,
                "grand_graded": grand_graded,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSummary saved: {summary_path.relative_to(PROJECT_ROOT)}")

    print(f"\nNext step: python scripts/03_download_tafsir.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
