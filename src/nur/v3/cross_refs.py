"""
NUR V3 Runtime — Canal 1: Cross-refs auto-pull

When a Quran chunk is retrieved in Phase A, automatically pull the hadiths
that are listed in its `hadith_cross_refs.high_confidence` metadata field.

Per docs/v3/02_CHUNK_SCHEMA.md "Cross-refs Quran → Hadith" and
docs/v3/04_RETRIEVAL_PIPELINE.md Step 3.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from .retriever_hadith import _get_hadith_collection  # noqa: E402


def auto_pull_hadiths(quran_chunks: list[dict], max_per_chunk: int = 3, max_total: int = 5) -> list[dict]:
    """For each Quran chunk, fetch hadiths listed in its hadith_cross_refs.

    Args:
        quran_chunks: top-K Quran chunks from Phase A
        max_per_chunk: cap cross-refs per chunk (avoid flooding)
        max_total: total auto-pulled hadiths cap

    Returns:
        List of hadith chunk dicts (deduplicated, capped at max_total).
        Each chunk gets "auto_pulled" = True and "auto_pulled_from" = quran_chunk_id.
    """
    col = _get_hadith_collection()

    hadith_ids_to_pull: list[tuple[str, str]] = []  # (hadith_id, quran_source_id)
    seen_hadith_ids: set[str] = set()

    for quran_chunk in quran_chunks:
        meta = quran_chunk.get("metadata", {})
        # ChromaDB flattens nested dicts in metadata, so cross-refs might be in
        # hadith_cross_refs_high_confidence as a pipe-separated string
        cross_refs_str = meta.get("hadith_cross_refs_high_confidence", "")
        if not cross_refs_str:
            continue

        # Parse pipe-separated list of source IDs
        hadith_ids = [h.strip() for h in cross_refs_str.split("|") if h.strip()]
        hadith_ids = hadith_ids[:max_per_chunk]

        for hid in hadith_ids:
            if hid not in seen_hadith_ids:
                seen_hadith_ids.add(hid)
                hadith_ids_to_pull.append((hid, quran_chunk["id"]))
                if len(hadith_ids_to_pull) >= max_total:
                    break
        if len(hadith_ids_to_pull) >= max_total:
            break

    if not hadith_ids_to_pull:
        print("[CrossRefs] No auto-pullable hadiths found in top Quran chunks")
        return []

    # Fetch all needed hadiths in one ChromaDB call
    hadith_ids = [hid for hid, _ in hadith_ids_to_pull]
    results = col.get(ids=hadith_ids, include=["documents", "metadatas"])

    id_to_source = {hid: qid for hid, qid in hadith_ids_to_pull}

    pulled = []
    for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        pulled.append({
            "id": cid,
            "embedding_text": doc,
            "metadata": meta,
            "auto_pulled": True,
            "auto_pulled_from": id_to_source.get(cid, ""),
        })

    print(f"[CrossRefs] Auto-pulled {len(pulled)} hadiths from Quran chunks")
    return pulled
