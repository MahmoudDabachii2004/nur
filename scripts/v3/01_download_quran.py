"""
NUR V3 — Step 1: Download Quran from alquran.cloud

Downloads:
  data/quran/quran-uthmani.json  — Arabic Uthmani (source of truth, 6,236 ayahs)
  data/quran/en.sahih.json       — Saheeh International EN translation
  data/quran/quran-meta.json     — Surah metadata (names, revelation type)

Source: http://api.alquran.cloud/v1
No API key required.

Usage:
  python scripts/v3/01_download_quran.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import QURAN_DIR  # noqa: E402

ALQURAN_CLOUD_BASE = "http://api.alquran.cloud/v1"

EDITIONS = {
    "arabic_uthmani": "quran-uthmani",
    "english_saheeh": "en.sahih",
}


def download_edition(edition_id: str) -> dict | None:
    url = f"{ALQURAN_CLOUD_BASE}/quran/{edition_id}"
    print(f"  [GET] {url}")
    try:
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            print(f"  [FAIL] API status: {data.get('status')}")
            return None
        return data["data"]
    except requests.RequestException as e:
        print(f"  [FAIL] Request error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"  [FAIL] Parse error: {e}")
        return None


def download_meta() -> dict | None:
    url = f"{ALQURAN_CLOUD_BASE}/meta"
    print(f"  [GET] {url}")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            return None
        return data["data"]
    except Exception as e:
        print(f"  [FAIL] {e}")
        return None


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size_kb = path.stat().st_size / 1024
    print(f"  [SAVE] {path.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")


def validate_quran(uthmani_data: dict) -> bool:
    surahs = uthmani_data.get("surahs", [])
    ayah_count = sum(len(s.get("ayahs", [])) for s in surahs)
    print(f"  Surahs: {len(surahs)} (expected 114)")
    print(f"  Ayahs:  {ayah_count} (expected 6,236)")
    if len(surahs) != 114 or ayah_count != 6236:
        print("  [WARN] Count mismatch — data may be truncated.")
        return False
    print("  [OK] Counts match expected values.")
    return True


def main() -> int:
    print("=" * 60)
    print("NUR V3 — Step 1: Download Quran")
    print("=" * 60)
    print(f"Output: {QURAN_DIR.relative_to(PROJECT_ROOT)}\n")

    QURAN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Arabic Uthmani
    print("[1/3] Downloading Arabic Uthmani (source of truth)...")
    ar_path = QURAN_DIR / "quran-uthmani.json"
    if ar_path.exists():
        print(f"  [SKIP] {ar_path.name} already exists.")
        with ar_path.open("r", encoding="utf-8") as f:
            ar_data = json.load(f)
    else:
        ar_data = download_edition(EDITIONS["arabic_uthmani"])
        if ar_data is None:
            print("\n[FATAL] Failed to download Arabic Quran — aborting.")
            return 1
        save_json(ar_data, ar_path)
        time.sleep(1)

    print("\n[Validation] Checking Arabic Quran integrity...")
    validate_quran(ar_data)

    # 2. English Saheeh International
    print("\n[2/3] Downloading English Saheeh International...")
    en_path = QURAN_DIR / "en.sahih.json"
    if en_path.exists():
        print(f"  [SKIP] {en_path.name} already exists.")
    else:
        en_data = download_edition(EDITIONS["english_saheeh"])
        if en_data is None:
            print("\n[WARN] Failed to download English — continuing.")
        else:
            save_json(en_data, en_path)
            en_ayahs = sum(len(s.get("ayahs", [])) for s in en_data.get("surahs", []))
            print(f"  English ayahs: {en_ayahs}")
        time.sleep(1)

    # 3. Metadata
    print("\n[3/3] Downloading Quran metadata (surah names, revelation type)...")
    meta_path = QURAN_DIR / "quran-meta.json"
    if meta_path.exists():
        print(f"  [SKIP] {meta_path.name} already exists.")
    else:
        meta_data = download_meta()
        if meta_data is None:
            print("\n[WARN] Failed to download metadata — continuing.")
        else:
            save_json(meta_data, meta_path)
            surah_count = len(meta_data.get("surahs", {}).get("references", []))
            print(f"  Surahs in metadata: {surah_count}")

    print("\n" + "=" * 60)
    print("Quran download complete!")
    print("=" * 60)
    print(f"\nNext step: python scripts/v3/02_download_hadith.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
