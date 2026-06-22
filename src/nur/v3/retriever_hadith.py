"""
NUR V3 Runtime — Phase B: Hadith retriever

Multi-query dense (ChromaDB) + sparse (BM25-like JSON) retrieval on hadith_v3,
fused with RRF, then reranked with bge-reranker-v2-m3, WITH authenticity
weighting (Sahih ×1.30, Hasan ×1.10, Da'if ×0.50, Mawdu ×0.00).

Per docs/v3/04_RETRIEVAL_PIPELINE.md Step 4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from nur.config import settings, DATA_DIR  # noqa: E402

from .reranker import rerank_chunks, compute_confidence  # noqa: E402
from .retriever_quran import (  # noqa: E402
    _encode_query,
    _encode_query_sparse,
    _rrf_fusion,
)

CHROMA_PATH = DATA_DIR / "chroma_db_v3"
SPARSE_HADITH_PATH = DATA_DIR / "sparse_v3" / "hadith_v3_sparse.json"

_chroma_client = None
_hadith_collection = None
_sparse_index = None


def _get_hadith_collection():
    global _chroma_client, _hadith_collection
    if _hadith_collection is not None:
        return _hadith_collection
    import chromadb
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _hadith_collection = _chroma_client.get_collection("hadith_v3_dense")
    return _hadith_collection


def _load_sparse_index():
    global _sparse_index
    if _sparse_index is not None:
        return _sparse_index
    if not SPARSE_HADITH_PATH.exists():
        print(f"[HadithRetriever] WARN: sparse index not found at {SPARSE_HADITH_PATH}")
        return None
    with SPARSE_HADITH_PATH.open("r", encoding="utf-8") as f:
        _sparse_index = json.load(f)
    return _sparse_index


def _dense_search(query_embedding: np.ndarray, top_k: int = 100) -> list[tuple[str, float]]:
    col = _get_hadith_collection()
    results = col.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["distances"],
    )
    ids = results["ids"][0]
    distances = results["distances"][0]
    similarities = [1.0 / (1.0 + float(d)) for d in distances]
    return list(zip(ids, similarities))


def _sparse_search(query_sparse: dict[int, float], top_k: int = 100) -> list[tuple[str, float]]:
    index = _load_sparse_index()
    if not index or not query_sparse:
        return []
    scores: dict[str, float] = {}
    for chunk_id, doc_weights in index.items():
        score = 0.0
        doc_weights_int = {int(k): float(v) for k, v in doc_weights.items()}
        for term_id, q_weight in query_sparse.items():
            if term_id in doc_weights_int:
                score += q_weight * doc_weights_int[term_id]
        if score > 0:
            scores[chunk_id] = score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]


def _fetch_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    col = _get_hadith_collection()
    results = col.get(ids=chunk_ids, include=["documents", "metadatas"])
    chunks = []
    for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        chunks.append({
            "id": cid,
            "embedding_text": doc,
            "metadata": meta,
        })
    id_to_chunk = {c["id"]: c for c in chunks}
    return [id_to_chunk[cid] for cid in chunk_ids if cid in id_to_chunk]


def retrieve_hadith(
    queries: list[str],
    top_k_initial: int = 400,
    top_k_rerank: int = 5,
) -> tuple[list[dict], str, float]:
    """Phase B retrieval: multi-query dense+sparse → RRF → rerank → top 5.

    With authenticity weighting (Pillar 3): Sahih ×1.30, Hasan ×1.10, Da'if ×0.50.

    Returns:
        (top_chunks, phase_b_status, confidence_b)
    """
    print(f"[Phase B] Retrieving Hadith chunks for {len(queries)} queries...")

    pooled: dict[str, float] = {}

    for q in queries:
        q_dense = _encode_query(q)
        dense_results = _dense_search(q_dense, top_k=100)
        q_sparse = _encode_query_sparse(q)
        sparse_results = _sparse_search(q_sparse, top_k=100)
        fused = _rrf_fusion(dense_results, sparse_results, k=settings.rrf_k, alpha_dense=settings.rrf_alpha_dense)
        for cid, score in fused.items():
            if cid not in pooled or score > pooled[cid]:
                pooled[cid] = score

    ranked = sorted(pooled.items(), key=lambda x: -x[1])
    top_ids = [cid for cid, _ in ranked[:top_k_initial]]
    print(f"[Phase B] RRF pool: {len(top_ids)} chunks")

    chunks = _fetch_chunks_by_ids(top_ids)

    print(f"[Phase B] Reranking {len(chunks)} chunks (with authenticity weighting)...")
    top_chunks = rerank_chunks(
        query=queries[0],
        chunks=chunks,
        top_k=top_k_rerank,
        apply_authenticity_weight=True,  # Hadith: apply Sahih/Hasan/Da'if weighting
    )

    # Confidence B uses authenticity-weighted score
    status, confidence = compute_confidence(top_chunks, apply_authenticity_weight=True)
    print(f"[Phase B] Status: {status} (confidence={confidence:.3f})")

    return top_chunks, status, confidence
