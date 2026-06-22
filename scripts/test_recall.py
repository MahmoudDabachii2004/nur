"""
test_recall.py — Recall audit for the NUR retriever.

Prints the top 100 chunks retrieved by RRF for a given query, as a flat list.
No panels, no colors, no formatting — just the raw data so it's easy to
copy-paste back to the agent for analysis.

USAGE:
  python3 scripts/test_recall.py
  python3 scripts/test_recall.py --query "Is prayer obligatory"
  python3 scripts/test_recall.py --query "..." --top-k 200

OUTPUT FORMAT (one chunk per line):
  rank | chunk_id | source | rrf_score | dense_rank | sparse_rank | text_preview

At the end, prints a RECALL CHECK section showing whether specific key chunks
(known-correct answers) appeared in the results.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nur.pipeline import NURPipeline
from src.nur.config import settings


# ============================================================
# Known-correct chunks to check recall against.
# Add entries here as we discover recall gaps in testing.
#
# IMPORTANT: Quran chunk IDs use GLOBAL ayah numbering (cumulative
# from surah 1), NOT standard surah:ayah. To avoid confusion, we
# search by METADATA (surah_num + standard ayah_num) instead of by
# chunk ID. The script fetches metadata for all retrieved chunks and
# matches against these expected entries.
#
# For hadiths, the chunk ID format is hadith_{collection_slug}_{number}
# which IS consistent, so we can match by chunk ID directly.
# ============================================================

# Hadith collection slug mapping (matches HADITH_COLLECTION_SLUGS in sources.py)
_HADITH_SLUGS = {
    "bukhari": "Sahih al-Bukhari",
    "muslim": "Sahih Muslim",
    "abudawud": "Sunan Abi Dawud",
    "tirmidhi": "Jami` at-Tirmidhi",
    "nasai": "Sunan an-Nasa'i",
    "ibnmajah": "Sunan Ibn Majah",
}


RECALL_CHECKS: dict[str, dict] = {
    "Is prayer obligatory": {
        "query": "Is prayer obligatory",
        # IMPORTANT: The NUR dataset uses a DIFFERENT ayah numbering than the
        # standard surah:ayah. The numbers below are the DB's ayah_num values,
        # NOT the standard ones. Verified via direct ChromaDB queries:
        #   - quran_2_10 (ayah_num=10) = standard 2:3 "establish prayer"
        #   - quran_2_50 (ayah_num=50) = standard 2:43 "establish prayer + zakah"
        #   - quran_2_117 (ayah_num=117) = standard 2:110 "establish prayer + zakah"
        #   - quran_2_284 (ayah_num=284) = standard 2:277 "establish prayer + zakah"
        #   - quran_4_596 (ayah_num=596) = standard 4:103 "prayer completed, remember Allah"
        # The DB appears to use cumulative/global numbering within each surah
        # (basmala counted as ayah 1 + offset). This is a known data-quality
        # issue to address in a future commit.
        "expected_quran": [
            # (surah_num, db_ayah_num, description)
            (2, 10, "Quran 2:3 (DB:2:10) — 'establish prayer' (foundational)"),
            (2, 50, "Quran 2:43 (DB:2:50) — 'establish prayer and give zakah'"),
            (2, 117, "Quran 2:110 (DB:2:117) — 'establish prayer and give zakah'"),
            (2, 284, "Quran 2:277 (DB:2:284) — 'establish prayer and give zakah'"),
            (4, 596, "Quran 4:103 (DB:4:596) — 'prayer decreed, remember Allah'"),
        ],
        "expected_hadith": [
            ("bukhari", 8, "Bukhari #8 — Islam built on 5 pillars"),
            ("muslim", 8, "Muslim #8 — 5 pillars hadith"),
            ("bukhari", 1, "Bukhari #1 — actions by intention"),
        ],
    },
}


def main() -> None:
    """Run the recall audit and print results as a flat list."""
    parser = argparse.ArgumentParser(description="NUR retriever recall audit.")
    parser.add_argument(
        "--query", "-q",
        default="Is prayer obligatory",
        help="The query to test (default: 'Is prayer obligatory').",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=100,
        help="Number of chunks to retrieve (default: 100).",
    )
    args = parser.parse_args()

    # Pre-flight check
    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY not set (needed for the Architect to decompose).")
        sys.exit(1)

    print(f"# NUR Recall Audit")
    print(f"# Query: {args.query}")
    print(f"# Top-K: {args.top_k}")
    print(f"# Models: Architect={settings.llm_architect}, BGE-M3 on auto-detected device")
    print()

    # Initialize pipeline (loads BGE-M3 on first query)
    print("# Initializing pipeline...", flush=True)
    pipeline = NURPipeline()
    print(f"# Pipeline ready. BGE-M3 device: {pipeline._device}")
    print()

    # Step 1: Architect decomposes the query
    print("# Step 1: Architect decomposing query...", flush=True)
    sub_questions = pipeline.generator.decompose_query(args.query)
    all_queries = [args.query] + sub_questions
    print(f"# Sub-questions ({len(sub_questions)}):")
    for i, sq in enumerate(sub_questions, 1):
        print(f"#   {i}. {sq}")
    print()

    # Step 2: Retrieve top-K chunks per (query, source) — NO truncation to 30
    # We patch the _retrieve call to use top_k=args.top_k instead of the default 30.
    print(f"# Step 2: Retrieving top-{args.top_k} per (query, source)...", flush=True)
    retrieved = pipeline._retrieve(all_queries, top_k=args.top_k)
    print(f"# Total unique chunks retrieved: {len(retrieved)}")
    print()

    # Print the flat list — one chunk per line, easy to copy-paste
    print(f"# ===== TOP {len(retrieved)} CHUNKS (ranked by RRF score) =====")
    print(f"# rank | chunk_id | source | rrf_score | dense_rank | sparse_rank | text_preview")
    print()

    # Build a lookup of chunk metadata for previews AND recall checking (batch fetch)
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.chroma_path if hasattr(settings, 'chroma_path') else './data/chroma_db'))
    
    # Group chunk IDs by source for batch fetching
    by_source: dict[str, list[str]] = {}
    for chunk in retrieved:
        by_source.setdefault(chunk["source"], []).append(chunk["id"])
    
    # Fetch metadata + documents for all retrieved chunks
    # metadata_map stores: chunk_id → {preview: str, surah_num: int, ayah_num: int, hadith_number: int, ...}
    metadata_map: dict[str, dict] = {}
    for source, chunk_ids in by_source.items():
        try:
            col = client.get_collection(f"{source}_dense")
            res = col.get(ids=chunk_ids, include=["documents", "metadatas"])
            for i, cid in enumerate(res["ids"]):
                doc = res["documents"][i] if i < len(res["documents"]) else ""
                meta = res["metadatas"][i] if i < len(res["metadatas"]) else {}
                preview = doc[:80].replace("\n", " ").replace("\r", " ")
                metadata_map[cid] = {
                    "preview": preview,
                    "surah_num": meta.get("surah_num"),
                    "ayah_num": meta.get("ayah_num"),
                    "hadith_number": meta.get("hadith_number"),
                    "collection": meta.get("collection"),
                    "collection_url_slug": meta.get("collection_url_slug"),
                }
        except Exception as e:
            print(f"# WARNING: could not fetch metadata for {source}: {e}", file=sys.stderr)

    for rank, chunk in enumerate(retrieved, 1):
        cid = chunk["id"]
        source = chunk["source"]
        rrf = chunk["rrf_score"]
        d_rank = str(chunk["dense_rank"]) if chunk["dense_rank"] else "-"
        s_rank = str(chunk["sparse_rank"]) if chunk["sparse_rank"] else "-"
        preview = metadata_map.get(cid, {}).get("preview", "(no preview)")
        print(f"{rank:3d} | {cid:30s} | {source:10s} | {rrf:.6f} | {d_rank:>9s} | {s_rank:>10s} | {preview}")

    # Step 3: Recall check — did the key chunks appear?
    # Match by metadata (surah_num + ayah_num for Quran, collection_slug + hadith_number for hadith)
    # NOT by chunk ID, because Quran chunk IDs use global numbering which is error-prone.
    print()
    print("# ===== RECALL CHECK =====")
    check = RECALL_CHECKS.get(args.query)
    if check:
        found = 0
        total = 0

        # Check Quran verses by surah_num + ayah_num across quran + tafsir collections.
        # A verse is considered "found" if it appears as a direct Quran chunk OR
        # as a Tafsir chunk (AR or EN) for that same surah:ayah — both contain
        # the verse's content and are valid retrieval hits.
        print(f"# Expected Quran verses for '{args.query}':")
        for surah_num, ayah_num, description in check.get("expected_quran", []):
            total += 1
            # Search retrieved chunks across all source types for matching surah+ayah
            match_rank = None
            match_source = None
            for i, chunk in enumerate(retrieved, 1):
                meta = metadata_map.get(chunk["id"], {})
                if (
                    meta.get("surah_num") == surah_num
                    and meta.get("ayah_num") == ayah_num
                ):
                    match_rank = i
                    match_source = chunk["source"]
                    break
            if match_rank:
                print(f"#   [FOUND rank {match_rank:3d}] {match_source:10s} {surah_num}:{ayah_num} — {description}")
                found += 1
            else:
                print(f"#   [MISS       ] quran/tafsir  {surah_num}:{ayah_num} — {description}")

        # Check Hadiths by collection_slug + hadith_number (chunk ID format is consistent)
        print()
        print(f"# Expected Hadiths for '{args.query}':")
        for collection_slug, hadith_number, description in check.get("expected_hadith", []):
            total += 1
            expected_id = f"hadith_{collection_slug}_{hadith_number}"
            match_rank = None
            for i, chunk in enumerate(retrieved, 1):
                if chunk["id"] == expected_id:
                    match_rank = i
                    break
            if match_rank:
                print(f"#   [FOUND rank {match_rank:3d}] {expected_id:30s} — {description}")
                found += 1
            else:
                print(f"#   [MISS       ] {expected_id:30s} — {description}")

        print()
        pct = found / total * 100 if total > 0 else 0
        print(f"# Recall: {found}/{total} = {pct:.0f}%")
        if found < total:
            print("# ⚠️  Some key chunks are missing — see MISS entries above.")
            print("#     Consider: improving Architect prompt, increasing --top-k, or adding query expansion.")
        else:
            print("# ✅ All key chunks found — recall is good for this query.")
    else:
        print(f"# No recall check defined for '{args.query}'.")
        print("# Add an entry to RECALL_CHECKS in this script to track expected chunks.")

    print()
    print("# Done.")


if __name__ == "__main__":
    main()
