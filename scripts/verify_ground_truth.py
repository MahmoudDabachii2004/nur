"""
verify_ground_truth.py — Fetch the ACTUAL DB chunk IDs for each ground truth verse.

This script does TWO things:
  1. Fetches the EXACT verse text from alquran.cloud API by surah:ayah (standard)
  2. Searches the DB for chunks containing that exact text

This eliminates all matching errors — we match by the actual verse text,
not by description snippets.

USAGE:
  python3 scripts/verify_ground_truth.py
"""

import sys
import os
import json
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from scripts.ground_truth import GROUND_TRUTH


def fetch_verse_text(surah: int, ayah: int) -> str:
    """Fetch the exact English text of a verse from alquran.cloud API.

    Uses the /ayah/{surah}:{ayah}/en.sahih endpoint which returns the
    standard-numbered verse text.

    Args:
        surah: Surah number (1-114, standard numbering).
        ayah: Ayah number within the surah (standard numbering).

    Returns:
        The English text of the verse, or empty string on failure.
    """
    url = f"http://api.alquran.cloud/v1/ayah/{surah}:{ayah}/en.sahih"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "OK":
            return data["data"]["text"]
    except Exception as e:
        print(f"    API error for {surah}:{ayah}: {e}")
    return ""


def search_db_by_text(client: chromadb.api.client.Client, text_en: str, surah_num: int) -> str | None:
    """Search the Quran DB collection for a chunk matching the exact verse text.

    Searches by surah_num + first 50 chars of text_en to find the exact chunk.

    Args:
        client: ChromaDB client.
        text_en: The exact English text from the API.
        surah_num: The surah number to narrow the search.

    Returns:
        The chunk ID if found, None otherwise.
    """
    col = client.get_collection("quran_dense")

    # Get all chunks for this surah (max ~286 per surah — well under SQL limit)
    res = col.get(where={"surah_num": surah_num}, include=["metadatas"])

    # Match by first 50 characters of text_en (handles minor whitespace diffs)
    search_text = text_en[:50].lower().strip()

    for i, cid in enumerate(res["ids"]):
        m = res["metadatas"][i]
        db_text = m.get("text_en", "")[:50].lower().strip()
        if db_text == search_text:
            return cid

    # If exact 50-char match fails, try progressively shorter prefixes
    for prefix_len in [40, 30, 20]:
        search_text = text_en[:prefix_len].lower().strip()
        for i, cid in enumerate(res["ids"]):
            m = res["metadatas"][i]
            db_text = m.get("text_en", "")[:prefix_len].lower().strip()
            if db_text == search_text:
                return cid

    return None


def main():
    """Verify each ground truth verse against the actual DB."""
    client = chromadb.PersistentClient(path="./data/chroma_db")

    results = {}

    for query, gt in GROUND_TRUTH.items():
        print(f"\n{'='*70}")
        print(f"  Query: '{query[:60]}'")
        print(f"{'='*70}")

        quran_results = []
        hadith_results = []

        # ---- Quran verses ----
        for surah, ayah_std, desc in gt.get("expected_quran_standard", []):
            # Step 1: Fetch exact text from API
            print(f"  [{surah}:{ayah_std}] Fetching from API...", end=" ")
            api_text = fetch_verse_text(surah, ayah_std)

            if not api_text:
                print("❌ API FAILED")
                quran_results.append({
                    "standard": f"{surah}:{ayah_std}",
                    "db_chunk_id": None,
                    "api_text_preview": "",
                    "desc": desc,
                })
                continue

            print(f"OK ({len(api_text)} chars)")

            # Step 2: Search DB for this exact text
            chunk_id = search_db_by_text(client, api_text, surah)

            if chunk_id:
                # Get the DB metadata for verification
                col = client.get_collection("quran_dense")
                res = col.get(ids=[chunk_id], include=["metadatas"])
                m = res["metadatas"][0]
                print(f"    ✅ Found in DB: {chunk_id} (ayah_num={m['ayah_num']})")
                print(f"    DB text: {m['text_en'][:80]}...")
                print(f"    API text: {api_text[:80]}...")
                quran_results.append({
                    "standard": f"{surah}:{ayah_std}",
                    "db_chunk_id": chunk_id,
                    "db_ayah_num": m["ayah_num"],
                    "api_text_preview": api_text[:80],
                    "db_text_preview": m["text_en"][:80],
                    "desc": desc,
                })
            else:
                print(f"    ❌ NOT FOUND in DB by text match")
                print(f"    API text was: {api_text[:80]}...")
                quran_results.append({
                    "standard": f"{surah}:{ayah_std}",
                    "db_chunk_id": None,
                    "api_text_preview": api_text[:80],
                    "desc": desc,
                })

        # ---- Hadiths ----
        hadith_col = client.get_collection("hadith_dense")
        for collection, number, desc in gt.get("expected_hadith", []):
            expected_id = f"hadith_{collection}_{number}"
            res = hadith_col.get(ids=[expected_id], include=["metadatas"])
            if res["ids"]:
                m = res["metadatas"][0]
                print(f"  ✅ {expected_id} → FOUND")
                print(f"     EN: {m.get('text_en','')[:80]}...")
                hadith_results.append({
                    "expected_id": expected_id,
                    "found": True,
                    "text_en_preview": m.get("text_en", "")[:80],
                    "desc": desc,
                })
            else:
                print(f"  ❌ {expected_id} NOT FOUND — desc: {desc[:60]}")
                hadith_results.append({
                    "expected_id": expected_id,
                    "found": False,
                    "text_en_preview": "",
                    "desc": desc,
                })

        results[query] = {
            "quran": quran_results,
            "hadith": hadith_results,
        }

    # Print the verified ground truth as a Python dict
    print(f"\n\n{'='*70}")
    print(f"  VERIFIED GROUND TRUTH — copy-paste into ground_truth.py")
    print(f"{'='*70}")
    print()
    print("VERIFIED_DB_IDS = {")
    for query, data in results.items():
        print(f"    \"{query}\": {{")
        print(f"        \"quran_chunk_ids\": [")
        for q in data["quran"]:
            if q["db_chunk_id"]:
                print(f"            (\"{q['db_chunk_id']}\", \"{q['standard']} — {q['desc'][:50]}\"),")
            else:
                print(f"            # ❌ {q['standard']} NOT FOUND — {q['desc'][:50]}")
        print(f"        ],")
        print(f"        \"hadith_chunk_ids\": [")
        for h in data["hadith"]:
            status = "✅" if h["found"] else "❌"
            print(f"            (\"{h['expected_id']}\", \"{h['desc'][:50]}\"),  # {status}")
        print(f"        ],")
        print(f"    }},")
    print("}")


if __name__ == "__main__":
    main()
