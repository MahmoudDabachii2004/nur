"""
NUR Phase 1 — Step 3: Download Tafsir Ibn Kathir

Source: spa5k/tafsir_api (https://github.com/spa5k/tafsir_api)
  - 1.15 GB repo (too big for jsDelivr's 50 MB limit — must use raw GitHub)
  - Path: tafsir/ar-tafsir-ibn-kathir/{surah}.json
  - Path: tafsir/en-tafisr-ibn-kathir/{surah}.json   ← NOTE THE TYPO "tafisr"
  - Schema: list of {text, ayah, surah} dicts (one per ayah group)

Why Tafsir Ibn Kathir:
  - Most widely accepted classical tafsir in Sunni Islam
  - Written by Ibn Kathir (1301-1373 CE), student of Ibn Taymiyyah
  - Uses Quran-explains-Quran methodology (verses cross-reference each other)
  - Includes hadith commentary (essential for Phase 8 cross-references)

Output: data/tafsir/
  ├── ar/          — Arabic Tafsir Ibn Kathir (114 files, one per surah)
  ├── en/          — English Tafsir Ibn Kathir (114 files)
  └── _summary.json

Usage:
  python scripts/03_download_tafsir.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import TAFSIR_DIR  # noqa: E402

# raw.githubusercontent.com — works for any repo size (no 50MB jsDelivr limit)
RAW_BASE = "https://raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir"

# Note: the EN folder has a typo in the upstream repo ("tafisr" not "tafsir")
EDITIONS = {
    "ar": {
        "dir": TAFSIR_DIR / "ar",
        "path": "ar-tafsir-ibn-kathir",
    },
    "en": {
        "dir": TAFSIR_DIR / "en",
        "path": "en-tafisr-ibn-kathir",  # upstream typo, must match exactly
    },
}


def download_surah(lang: str, surah_num: int, edition_path: str) -> list | None:
    """Download a single surah's tafsir file. Returns the list of {text, ayah, surah} dicts."""
    url = f"{RAW_BASE}/{edition_path}/{surah_num}.json"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"    [FAIL] Surah {surah_num}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"    [FAIL] Surah {surah_num} JSON parse: {e}")
        return None


def download_language(lang: str, edition_path: str, out_dir: Path) -> dict:
    """Download all 114 surahs for one language edition.

    Returns summary stats.
    """
    print(f"\n[{lang.upper()}] Tafsir Ibn Kathir")
    print("-" * 50)

    out_dir.mkdir(parents=True, exist_ok=True)

    surah_count = 0
    section_count = 0
    total_bytes = 0
    failed_surahs: list[int] = []

    for surah_num in range(1, 115):  # 1..114 inclusive
        out_file = out_dir / f"{surah_num:03d}.json"

        if out_file.exists():
            # Count existing
            with out_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                section_count += len(data)
            surah_count += 1
            if surah_num % 25 == 0:
                print(f"  [{surah_num:3d}/114] [SKIP] exists")
            continue

        data = download_surah(lang, surah_num, edition_path)
        if data is None:
            failed_surahs.append(surah_num)
            continue

        with out_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        section_count += len(data) if isinstance(data, list) else 0
        total_bytes += out_file.stat().st_size
        surah_count += 1

        # Progress indicator
        if surah_num % 10 == 0:
            print(f"  [{surah_num:3d}/114] downloaded ({section_count:,} sections so far)")

        # Be polite to GitHub (no CDN — they throttle aggressive requests)
        time.sleep(0.2)

    print(f"\n  [{lang.upper()}] Done: {surah_count}/114 surahs, {section_count:,} tafsir sections, {total_bytes/1024/1024:.1f} MB")
    if failed_surahs:
        print(f"  [{lang.upper()}] Failed surahs: {failed_surahs}")

    return {
        "language": lang,
        "surah_count": surah_count,
        "section_count": section_count,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "failed_surahs": failed_surahs,
    }


def inspect_first_file(out_dir: Path, lang: str) -> None:
    """Inspect schema of first downloaded file."""
    first_file = out_dir / "001.json"
    if not first_file.exists():
        return
    with first_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n  Schema inspection ({lang.upper()} Surah 1):")
    if isinstance(data, list) and data:
        print(f"    Type: list of {len(data)} entries")
        print(f"    First entry keys: {list(data[0].keys())}")
        # Sample
        if isinstance(data[0], dict):
            for k, v in data[0].items():
                s = str(v)
                if len(s) > 80:
                    s = s[:77] + "..."
                print(f"      {k}: {s}")


def main() -> int:
    print("=" * 60)
    print("NUR Phase 1 — Step 3: Download Tafsir Ibn Kathir")
    print("=" * 60)
    print(f"Source: spa5k/tafsir_api (via raw.githubusercontent.com)")
    print(f"Output: {TAFSIR_DIR.relative_to(PROJECT_ROOT)}")

    summaries = {}

    for lang, info in EDITIONS.items():
        summary = download_language(lang, info["path"], info["dir"])
        summaries[lang] = summary
        inspect_first_file(info["dir"], lang)

    # Save summary
    summary_path = TAFSIR_DIR / "_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "spa5k/tafsir_api",
                "url": "https://github.com/spa5k/tafsir_api",
                "cdn": "raw.githubusercontent.com (repo >50MB jsDelivr limit)",
                "note": "EN folder has upstream typo 'tafisr' (not 'tafsir')",
                "editions": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("TAFSIR DOWNLOAD COMPLETE")
    print("=" * 60)
    for lang, s in summaries.items():
        print(f"  {lang.upper()}: {s['surah_count']}/114 surahs, {s['section_count']:,} sections, {s['total_mb']} MB")
    print(f"\nSummary: {summary_path.relative_to(PROJECT_ROOT)}")

    print(f"\nNext step: python scripts/04_normalize_and_chunk.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
