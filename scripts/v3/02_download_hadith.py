"""
NUR V3 — Step 2: Download Hadith collections from meeAtif/hadith_datasets

Source: https://huggingface.co/datasets/meeAtif/hadith_datasets
License: MIT (per docs/DATA_SOURCES.md)

Downloads 6 canonical Sunni collections (Kutub al-Sittah):
  - Sahih al-Bukhari     (~7,008 hadiths)
  - Sahih Muslim         (~5,362 hadiths)
  - Sunan Abi Dawud      (~4,590 hadiths)
  - Jami` at-Tirmidhi    (~3,956 hadiths)
  - Sunan an-Nasa'i      (~5,662 hadiths)
  - Sunan Ibn Majah      (~4,341 hadiths)

Output: data/hadith/meetif/<CollectionName>.json

Each JSON file is a list of hadith objects. The "Reference" field (URL sunnah.com)
is canonical and MUST be preserved in chunk metadata. Never reconstruct URLs.

Usage:
  python scripts/v3/02_download_hadith.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import HADITH_DIR  # noqa: E402

# Direct HuggingFace resolve URLs (per docs/DATA_SOURCES.md)
BASE_URL = "https://huggingface.co/datasets/meeAtif/hadith_datasets/resolve/main"

COLLECTIONS = [
    "Sahih al-Bukhari",
    "Sahih Muslim",
    "Sunan Abi Dawud",
    "Jami` at-Tirmidhi",
    "Sunan an-Nasa'i",
    "Sunan Ibn Majah",
]


def download_file(url: str) -> dict | list | None:
    try:
        # HuggingFace redirects to CDN — follow redirects
        resp = requests.get(url, timeout=180, allow_redirects=True)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"    [FAIL] {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"    [FAIL] JSON parse: {e}")
        return None


def main() -> int:
    print("=" * 60)
    print("NUR V3 — Step 2: Download Hadith collections")
    print("=" * 60)
    print(f"Source: {BASE_URL}")
    print(f"Output: {HADITH_DIR.relative_to(PROJECT_ROOT)}/meetif/\n")

    meetif_dir = HADITH_DIR / "meetif"
    meetif_dir.mkdir(parents=True, exist_ok=True)

    total_hadiths = 0

    for collection_name in COLLECTIONS:
        out_path = meetif_dir / f"{collection_name}.json"
        if out_path.exists():
            print(f"[{collection_name}] [SKIP] already exists.")
            with out_path.open("r", encoding="utf-8") as f:
                hadiths = json.load(f)
            if isinstance(hadiths, list):
                total_hadiths += len(hadiths)
            continue

        # URL-encode the space in collection name
        from urllib.parse import quote
        url = f"{BASE_URL}/{quote(collection_name)}.json"
        print(f"[{collection_name}] Downloading from {url}...")
        data = download_file(url)
        if data is None:
            print(f"  [FAIL] Could not download {collection_name}")
            continue

        if not isinstance(data, list):
            print(f"  [WARN] Expected list, got {type(data).__name__}")
            continue

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        total_hadiths += len(data)
        size_kb = out_path.stat().st_size / 1024
        print(f"  [OK] {len(data):,} hadiths ({size_kb:.0f} KB)")

        # Be polite
        time.sleep(0.5)

    # Save summary
    summary_path = meetif_dir / "_summary.json"
    summary = {
        "source": "meeAtif/hadith_datasets on HuggingFace",
        "base_url": BASE_URL,
        "collections": COLLECTIONS,
        "total_hadiths": total_hadiths,
        "license": "MIT",
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("HADITH DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Total hadiths: {total_hadiths:,}")
    print(f"  Output: {meetif_dir.relative_to(PROJECT_ROOT)}/")
    print(f"\nNext step: python scripts/v3/03_download_tafsirs.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
