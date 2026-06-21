"""
NUR Phase 1 — Step 1: Download Quran

Source: alquran.cloud API (https://alquran.cloud/api)
  - Most reliable public Quran API
  - Returns Uthmani Arabic + multiple translations
  - No API key required, no rate limit (reasonable use)

Downloads (saved to data/quran/):
  - quran-uthmani.json   — Arabic Uthmani text (source of truth, 6,236 ayahs)
  - en.sahih.json        — English Saheeh International translation
  - quran-meta.json      — Surah metadata (names, revelation type, ayah counts)

Usage:
  python scripts/01_download_quran.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

# Make sure we can import the nur package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nur.config import QURAN_DIR  # noqa: E402

ALQURAN_CLOUD_BASE = "http://api.alquran.cloud/v1"

# Editions to download — Arabic is the source of truth, English for comprehension.
# (French dropped per project direction: English + Arabic only.)
EDITIONS = {
    "arabic_uthmani": "quran-uthmani",   # Source of truth — Uthmani script
    "english_saheeh": "en.sahih",        # Saheeh International (most widely accepted EN)
    "quran_meta": None,                  # Surah metadata, special endpoint
}


def download_edition(name: str, edition_id: str) -> dict | None:
    """Download a full Quran edition from alquran.cloud.

    Returns the parsed JSON data, or None on failure.
    """
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
    """Download Quran metadata (surah names, revelation type, ayah counts)."""
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


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size_kb = path.stat().st_size / 1024
    print(f"  [SAVE] {path.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")


def validate_quran(uthmani_data: dict) -> bool:
    """Sanity check: 114 surahs, 6,236 ayahs."""
    surahs = uthmani_data.get("surahs", [])
    ayah_count = sum(len(s.get("ayahs", [])) for s in surahs)

    print(f"  Surahs: {len(surahs)} (expected 114)")
    print(f"  Ayahs:  {ayah_count} (expected 6,236)")

    if len(surahs) != 114:
        print("  [WARN] Surah count mismatch — data may be truncated.")
        return False
    if ayah_count != 6236:
        print("  [WARN] Ayah count mismatch — data may be truncated.")
        return False
    print("  [OK] Counts match expected values.")
    return True


def main() -> int:
    print("=" * 60)
    print("NUR Phase 1 — Step 1: Download Quran")
    print("=" * 60)
    print(f"Output directory: {QURAN_DIR.relative_to(PROJECT_ROOT)}\n")

    QURAN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Arabic Uthmani (source of truth)
    print("[1/3] Downloading Arabic Uthmani text (source of truth)...")
    ar_path = QURAN_DIR / "quran-uthmani.json"
    if ar_path.exists():
        print(f"  [SKIP] {ar_path.name} already exists.")
        with ar_path.open("r", encoding="utf-8") as f:
            ar_data = json.load(f)
    else:
        ar_data = download_edition("arabic_uthmani", EDITIONS["arabic_uthmani"])
        if ar_data is None:
            print("\n[FATAL] Failed to download Arabic Quran — aborting.")
            return 1
        save_json(ar_data, ar_path)
        time.sleep(1)

    # Validate
    print("\n[Validation] Checking Arabic Quran integrity...")
    validate_quran(ar_data)

    # 2. English Saheeh International
    print("\n[2/3] Downloading English Saheeh International translation...")
    en_path = QURAN_DIR / "en.sahih.json"
    if en_path.exists():
        print(f"  [SKIP] {en_path.name} already exists.")
    else:
        en_data = download_edition("english_saheeh", EDITIONS["english_saheeh"])
        if en_data is None:
            print("\n[WARN] Failed to download English translation — continuing.")
        else:
            save_json(en_data, en_path)
            # Quick validation
            en_ayahs = sum(len(s.get("ayahs", [])) for s in en_data.get("surahs", []))
            print(f"  English ayahs: {en_ayahs}")
        time.sleep(1)

    # 3. Metadata (surah names, revelation type)
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
    print(f"\nNext step: python scripts/02_download_hadith.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
