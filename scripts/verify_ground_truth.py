"""
verify_ground_truth.py — Fetch the ACTUAL DB chunk IDs for each ground truth verse.

This script searches the DB by TEXT CONTENT (not by ayah number) to find the
real chunk ID for each expected verse. This eliminates any offset calculation
errors — we get the ground truth directly from the DB.

The output is a copy-paste-ready Python dict that can replace the manual
SURAH_CUMULATIVE table in ground_truth.py.

USAGE:
  python3 scripts/verify_ground_truth.py
"""

import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from scripts.ground_truth import GROUND_TRUTH


def main():
    """Search the DB for each ground truth verse by text content."""
    client = chromadb.PersistentClient(path="./data/chroma_db")
    quran_col = client.get_collection("quran_dense")
    hadith_col = client.get_collection("hadith_dense")

    # Get ALL quran chunks at once (6236 — fits in memory)
    print("Loading all Quran chunks from DB...")
    all_quran = quran_col.get(include=["metadatas"])
    print(f"  Loaded {len(all_quran['ids'])} Quran chunks")

    # Get ALL hadith chunks at once (33738 — fits in memory)
    print("Loading all Hadith chunks from DB...")
    all_hadith = hadith_col.get(include=["metadatas"])
    print(f"  Loaded {len(all_hadith['ids'])} Hadith chunks")

    results = {}

    for query, gt in GROUND_TRUTH.items():
        print(f"\n{'='*70}")
        print(f"  Query: '{query[:60]}'")
        print(f"{'='*70}")

        quran_results = []
        hadith_results = []

        # ---- Quran verses ----
        for surah, ayah_std, desc in gt.get("expected_quran_standard", []):
            # Search by surah_num first, then match by text snippet
            # We use the first 40 chars of the description as a search key
            search_key = desc.lower()[:40]

            found = False
            for i, cid in enumerate(all_quran["ids"]):
                m = all_quran["metadatas"][i]
                if m.get("surah_num") != surah:
                    continue
                text_en = m.get("text_en", "").lower()
                # Try to match by key phrases from the description
                if search_key in text_en or desc[:30].lower() in text_en:
                    quran_results.append({
                        "standard": f"{surah}:{ayah_std}",
                        "db_chunk_id": cid,
                        "db_ayah_num": m.get("ayah_num"),
                        "text_en_preview": m.get("text_en", "")[:80],
                    })
                    print(f"  ✅ {surah}:{ayah_std} → {cid} (ayah_num={m['ayah_num']})")
                    print(f"     EN: {m.get('text_en','')[:80]}...")
                    found = True
                    break

            if not found:
                # Try broader search — just match surah + first 20 chars
                for i, cid in enumerate(all_quran["ids"]):
                    m = all_quran["metadatas"][i]
                    if m.get("surah_num") != surah:
                        continue
                    text_en = m.get("text_en", "").lower()
                    if desc[:20].lower() in text_en:
                        quran_results.append({
                            "standard": f"{surah}:{ayah_std}",
                            "db_chunk_id": cid,
                            "db_ayah_num": m.get("ayah_num"),
                            "text_en_preview": m.get("text_en", "")[:80],
                        })
                        print(f"  ✅ {surah}:{ayah_std} → {cid} (ayah_num={m['ayah_num']}) [broad match]")
                        found = True
                        break

            if not found:
                print(f"  ❌ {surah}:{ayah_std} NOT FOUND in DB — desc: {desc[:60]}")
                # Print all chunks for this surah to help debug
                surah_chunks = [(cid, all_quran["metadatas"][i]) for i, cid in enumerate(all_quran["ids"]) if all_quran["metadatas"][i].get("surah_num") == surah]
                print(f"     Surah {surah} has {len(surah_chunks)} chunks in DB")
                # Print first 5
                for cid, m in surah_chunks[:5]:
                    print(f"       {cid} (ayah={m['ayah_num']}): {m.get('text_en','')[:60]}...")

        # ---- Hadiths ----
        for collection, number, desc in gt.get("expected_hadith", []):
            expected_id = f"hadith_{collection}_{number}"
            # Direct lookup by chunk ID
            res = hadith_col.get(ids=[expected_id], include=["metadatas"])
            if res["ids"]:
                m = res["metadatas"][0]
                hadith_results.append({
                    "expected_id": expected_id,
                    "found": True,
                    "text_en_preview": m.get("text_en", "")[:80],
                })
                print(f"  ✅ {expected_id} → FOUND")
                print(f"     EN: {m.get('text_en','')[:80]}...")
            else:
                hadith_results.append({
                    "expected_id": expected_id,
                    "found": False,
                    "text_en_preview": "",
                })
                print(f"  ❌ {expected_id} NOT FOUND in DB — desc: {desc[:60]}")

        results[query] = {
            "quran": quran_results,
            "hadith": hadith_results,
        }

    # Print the verified ground truth as a Python dict
    print(f"\n\n{'='*70}")
    print(f"  VERIFIED GROUND TRUTH (copy-paste into ground_truth.py)")
    print(f"{'='*70}")
    print()
    print("# Verified DB chunk IDs for each expected verse/hadith")
    print("# Generated by scripts/verify_ground_truth.py")
    print("VERIFIED_GROUND_TRUTH = {")
    for query, data in results.items():
        print(f"    \"{query}\": {{")
        print(f"        \"quran_chunk_ids\": [")
        for q in data["quran"]:
            print(f"            \"{q['db_chunk_id']}\",  # {q['standard']} — {q['text_en_preview'][:50]}")
        print(f"        ],")
        print(f"        \"hadith_chunk_ids\": [")
        for h in data["hadith"]:
            status = "✅" if h["found"] else "❌"
            print(f"            \"{h['expected_id']}\",  # {status} — {h['text_en_preview'][:50]}")
        print(f"        ],")
        print(f"    }},")
    print("}")


if __name__ == "__main__":
    main()
