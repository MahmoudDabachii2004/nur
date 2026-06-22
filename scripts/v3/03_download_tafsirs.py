"""
NUR V3 — Step 3: Download Tafsirs from spa5k/tafsir_api upstream

⚠️ The existing data/tafsir/en_ibn_kathir.parquet has 70% EMPTY entries
   (4,341 / 6,235 rows are empty strings). We MUST re-fetch from upstream.

Downloads 4 tafsirs (V3 spec):
  1. en-tafisr-ibn-kathir  (note: upstream typo "tafisr")  — Ibn Kathir EN
  2. ar-tafsir-ibn-kathir                              — Ibn Kathir AR
  3. ar-tafsir-al-tabari                               — Al-Tabari AR
  4. ar-tafsir-as-saadi                                — As-Sa'di AR

Output structure:
  data/tafsir/v3/
    ├── ibn_kathir_en/   (114 files: 001.json ... 114.json)
    ├── ibn_kathir_ar/
    ├── tabari_ar/
    ├── saadi_ar/
    └── _summary.json

Each file: list of {text, ayah, surah} dicts (1 per ayah).

Usage:
  python scripts/v3/03_download_tafsirs.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import TAFSIR_DIR  # noqa: E402

RAW_BASE = "https://raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir"

# Note: the EN Ibn Kathir folder has a typo upstream ("tafisr" not "tafsir")
TAFSIR_EDITIONS = {
    "ibn_kathir_en": {
        "upstream_path": "en-tafisr-ibn-kathir",  # typo upstream
        "language": "en",
        "display_name": "Ibn Kathir",
        "category": "bil-Mathur",
    },
    "ibn_kathir_ar": {
        "upstream_path": "ar-tafsir-ibn-kathir",
        "language": "ar",
        "display_name": "Ibn Kathir",
        "category": "bil-Mathur",
    },
    "tabari_ar": {
        "upstream_path": "ar-tafsir-al-tabari",
        "language": "ar",
        "display_name": "Al-Tabari",
        "category": "bil-Mathur",
    },
    "saadi_ar": {
        "upstream_path": "ar-tafsir-as-saadi",
        "language": "ar",
        "display_name": "As-Sa'di",
        "category": "modern",
    },
}


def download_surah(upstream_path: str, surah_num: int) -> list | None:
    url = f"{RAW_BASE}/{upstream_path}/{surah_num}.json"
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


def download_edition(edition_key: str, edition_info: dict, out_dir: Path) -> dict:
    print(f"\n[{edition_key}] {edition_info['display_name']} ({edition_info['language'].upper()})")
    print("-" * 60)

    out_dir.mkdir(parents=True, exist_ok=True)

    surah_count = 0
    section_count = 0
    empty_count = 0
    failed_surahs: list[int] = []

    for surah_num in range(1, 115):
        out_file = out_dir / f"{surah_num:03d}.json"

        if out_file.exists():
            with out_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                section_count += len(data)
                empty_count += sum(1 for d in data if not str(d.get("text", "")).strip())
            surah_count += 1
            if surah_num % 25 == 0:
                print(f"  [{surah_num:3d}/114] [SKIP] exists")
            continue

        data = download_surah(edition_info["upstream_path"], surah_num)
        if data is None:
            failed_surahs.append(surah_num)
            continue

        with out_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if isinstance(data, list):
            section_count += len(data)
            empty_count += sum(1 for d in data if not str(d.get("text", "")).strip())

        surah_count += 1

        if surah_num % 10 == 0:
            print(f"  [{surah_num:3d}/114] downloaded ({section_count:,} sections so far)")

        # Be polite to GitHub (no CDN)
        time.sleep(0.15)

    print(f"\n  [{edition_key}] Done: {surah_count}/114 surahs, {section_count:,} sections, "
          f"{empty_count:,} empty")
    if failed_surahs:
        print(f"  [{edition_key}] Failed surahs: {failed_surahs}")

    return {
        "edition_key": edition_key,
        "display_name": edition_info["display_name"],
        "language": edition_info["language"],
        "category": edition_info["category"],
        "upstream_path": edition_info["upstream_path"],
        "surah_count": surah_count,
        "section_count": section_count,
        "empty_count": empty_count,
        "failed_surahs": failed_surahs,
    }


def main() -> int:
    print("=" * 60)
    print("NUR V3 — Step 3: Download Tafsirs (4 editions)")
    print("=" * 60)
    print(f"Source: spa5k/tafsir_api (raw.githubusercontent.com)")
    print(f"Output: {TAFSIR_DIR.relative_to(PROJECT_ROOT)}/v3/")

    v3_dir = TAFSIR_DIR / "v3"
    v3_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for edition_key, edition_info in TAFSIR_EDITIONS.items():
        out_dir = v3_dir / edition_key
        summary = download_edition(edition_key, edition_info, out_dir)
        summaries[edition_key] = summary

    # Save combined summary
    summary_path = v3_dir / "_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "spa5k/tafsir_api",
                "url": "https://github.com/spa5k/tafsir_api",
                "note": "Re-fetched from upstream because en_ibn_kathir.parquet had 70% empty entries",
                "editions": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("TAFSIR DOWNLOAD COMPLETE")
    print("=" * 60)
    for key, s in summaries.items():
        print(f"  {key}: {s['surah_count']}/114 surahs, {s['section_count']:,} sections, "
              f"{s['empty_count']:,} empty")
    print(f"\n  Summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"\nNext step: python scripts/v3/04_generate_context_cards.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
