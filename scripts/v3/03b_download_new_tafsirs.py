"""
NUR V3 — Download the 3 final tafsirs (6 editions: EN+AR)

Downloads:
  1. Ibn Kathir (EN + AR) — already downloaded, will skip if exists
  2. Al-Jalalayn (EN + AR) — NEW
  3. Al-Mukhtasar (EN + AR) — NEW

These 3 tafsirs are the standard reference tafsirs available in BOTH
English and Arabic on Quran.com / spa5k.

Usage:
  python3 scripts/v3/03b_download_new_tafsirs.py
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

TAFSIR_EDITIONS = {
    # Already downloaded (will skip if exists)
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
    # NEW — Al-Jalalayn
    "jalalayn_en": {
        "upstream_path": "en-al-jalalayn",
        "language": "en",
        "display_name": "Al-Jalalayn",
        "category": "classical",
    },
    "jalalayn_ar": {
        "upstream_path": "ar-tafsir-al-jalalayn",
        "language": "ar",
        "display_name": "Al-Jalalayn",
        "category": "classical",
    },
    # NEW — Al-Mukhtasar
    "mukhtasar_en": {
        "upstream_path": "en-tafsir-al-mukhtasar",
        "language": "en",
        "display_name": "Al-Mukhtasar",
        "category": "modern",
    },
    "mukhtasar_ar": {
        "upstream_path": "ar-tafsir-al-mukhtasar",
        "language": "ar",
        "display_name": "Al-Mukhtasar",
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
    print("NUR V3 — Download 3 Tafsirs (6 editions: EN+AR)")
    print("=" * 60)
    print(f"  1. Ibn Kathir (EN + AR) — Primary, bil-Mathur")
    print(f"  2. Al-Jalalayn (EN + AR) — Classical, concise")
    print(f"  3. Al-Mukhtasar (EN + AR) — Modern, accessible")
    print(f"Source: spa5k/tafsir_api (raw.githubusercontent.com)")
    print(f"Output: {TAFSIR_DIR.relative_to(PROJECT_ROOT)}/v3/")

    v3_dir = TAFSIR_DIR / "v3"
    v3_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for edition_key, edition_info in TAFSIR_EDITIONS.items():
        out_dir = v3_dir / edition_key
        summary = download_edition(edition_key, edition_info, out_dir)
        summaries[edition_key] = summary

    summary_path = v3_dir / "_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "spa5k/tafsir_api",
                "url": "https://github.com/spa5k/tafsir_api",
                "tafsirs": ["Ibn Kathir", "Al-Jalalayn", "Al-Mukhtasar"],
                "languages": ["en", "ar"],
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
    print(f"\nNext step: python3 scripts/v3/05_build_chunks.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
