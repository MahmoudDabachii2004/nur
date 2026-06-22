"""
NUR V3 Runtime — Phase A: Quran+Tafsir retriever

Multi-query dense (ChromaDB) + sparse (BM25-like JSON) retrieval on quran_v3,
fused with Reciprocal Rank Fusion (RRF), then reranked with bge-reranker-v2-m3.

Per docs/v3/04_RETRIEVAL_PIPELINE.md Step 2.

Output: top 5 Quran chunks + confidence A (STRONG/WEAK/EMPTY).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from nur.config import settings, DATA_DIR  # noqa: E402

from .reranker import rerank_chunks, compute_confidence  # noqa: E402

CHROMA_PATH = DATA_DIR / "chroma_db_v3"
SPARSE_QURAN_PATH = DATA_DIR / "sparse_v3" / "quran_v3_sparse.json"

# Cached singletons
_chroma_client = None
_quran_collection = None
_sparse_index = None
_bge_m3 = None


def _load_bge_m3():
    global _bge_m3
    if _bge_m3 is not None:
        return _bge_m3
    from FlagEmbedding import BGEM3FlagModel
    print("[QuranRetriever] Loading BGE-M3...")
    _bge_m3 = BGEM3FlagModel(settings.embedding_model, use_fp16=True)
    print("[QuranRetriever] BGE-M3 loaded.")
    return _bge_m3


def _get_quran_collection():
    global _chroma_client, _quran_collection
    if _quran_collection is not None:
        return _quran_collection
    import chromadb
    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _quran_collection = _chroma_client.get_collection("quran_v3_dense")
    return _quran_collection


def _load_sparse_index():
    """Load sparse lexical weights for quran_v3."""
    global _sparse_index
    if _sparse_index is not None:
        return _sparse_index
    if not SPARSE_QURAN_PATH.exists():
        print(f"[QuranRetriever] WARN: sparse index not found at {SPARSE_QURAN_PATH}")
        return None
    with SPARSE_QURAN_PATH.open("r", encoding="utf-8") as f:
        _sparse_index = json.load(f)
    return _sparse_index


def _encode_query(query: str) -> np.ndarray:
    """Encode user query with BGE-M3 → L2-normalized dense vector."""
    model = _load_bge_m3()
    output = model.encode([query], return_dense=True, return_sparse=False)
    dense = output["dense_vecs"]
    if hasattr(dense, "cpu"):
        dense = dense.cpu().numpy()
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dense / norms


def _encode_query_sparse(query: str) -> dict[int, float]:
    """Encode user query with BGE-M3 → sparse lexical weights dict."""
    model = _load_bge_m3()
    output = model.encode([query], return_dense=False, return_sparse=True)
    lw = output["lexical_weights"][0]
    if not lw:
        return {}
    return {int(k): float(v) for k, v in lw.items()}


def _dense_search(query_embedding: np.ndarray, top_k: int = 100) -> list[tuple[str, float]]:
    """Run ChromaDB dense search. Returns [(chunk_id, similarity_score)]."""
    col = _get_quran_collection()
    results = col.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["distances"],
    )
    ids = results["ids"][0]
    distances = results["distances"][0]
    # ChromaDB returns L2 distance (lower = better). Convert to similarity (higher = better).
    # We use 1 / (1 + distance) as a similarity proxy (any monotonic transform works for RRF).
    similarities = [1.0 / (1.0 + float(d)) for d in distances]
    return list(zip(ids, similarities))


def _sparse_search(query_sparse: dict[int, float], top_k: int = 100) -> list[tuple[str, float]]:
    """Run sparse search (dot product between query and document lexical weights)."""
    index = _load_sparse_index()
    if not index or not query_sparse:
        return []

    scores: dict[str, float] = {}
    for chunk_id, doc_weights in index.items():
        # Dot product between query and document sparse vectors
        # Only iterate over query terms (smaller set)
        score = 0.0
        doc_weights_int = {int(k): float(v) for k, v in doc_weights.items()}
        for term_id, q_weight in query_sparse.items():
            if term_id in doc_weights_int:
                score += q_weight * doc_weights_int[term_id]
        if score > 0:
            scores[chunk_id] = score

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]


def _rrf_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 25,
    alpha_dense: float = 0.4,
) -> dict[str, float]:
    """Reciprocal Rank Fusion of dense + sparse results.

    Args:
        dense_results: [(chunk_id, score)] sorted by score desc
        sparse_results: [(chunk_id, score)] sorted by score desc
        k: RRF smoothing constant (standard: 60, Islamic RAG tuned: 25 per DEC-034)
        alpha_dense: weight for dense (sparse gets 1 - alpha_dense)

    Returns: {chunk_id: rrf_score}
    """
    fused: dict[str, float] = {}
    alpha_sparse = 1.0 - alpha_dense

    for rank, (chunk_id, _) in enumerate(dense_results):
        rrf_score = alpha_dense * (1.0 / (k + rank + 1))
        fused[chunk_id] = fused.get(chunk_id, 0.0) + rrf_score

    for rank, (chunk_id, _) in enumerate(sparse_results):
        rrf_score = alpha_sparse * (1.0 / (k + rank + 1))
        fused[chunk_id] = fused.get(chunk_id, 0.0) + rrf_score

    return fused


def _fetch_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    """Fetch full chunk data from ChromaDB by IDs."""
    col = _get_quran_collection()
    results = col.get(ids=chunk_ids, include=["documents", "metadatas"])
    chunks = []
    for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        chunks.append({
            "id": cid,
            "embedding_text": doc,
            "metadata": meta,
        })
    # Preserve the order of input chunk_ids (ChromaDB doesn't guarantee order)
    id_to_chunk = {c["id"]: c for c in chunks}
    return [id_to_chunk[cid] for cid in chunk_ids if cid in id_to_chunk]


def retrieve_quran(
    queries: list[str],
    top_k_initial: int = 400,
    top_k_rerank: int = 5,
) -> tuple[list[dict], str, float]:
    """Phase A retrieval: multi-query dense+sparse → RRF → rerank → top 5.

    Args:
        queries: list of queries (user question + sub-questions from Architect)
        top_k_initial: pool size after RRF (400 = DEC-034 sweet spot)
        top_k_rerank: final top-K (5 = Groq TPM budget)

    Returns:
        (top_chunks, phase_a_status, confidence_a)
        top_chunks: list of top-5 chunk dicts (with rerank_score + final_score)
        phase_a_status: "STRONG" | "WEAK" | "EMPTY"
        confidence_a: max rerank score (0.0 - 1.0)
    """
    print(f"[Phase A] Retrieving Quran chunks for {len(queries)} queries...")

    # Multi-query: accumulate RRF scores across all queries
    pooled: dict[str, float] = {}

    for q in queries:
        # Dense search
        q_dense = _encode_query(q)
        dense_results = _dense_search(q_dense, top_k=100)

        # Sparse search
        q_sparse = _encode_query_sparse(q)
        sparse_results = _sparse_search(q_sparse, top_k=100)

        # RRF fusion for this query
        fused = _rrf_fusion(dense_results, sparse_results, k=settings.rrf_k, alpha_dense=settings.rrf_alpha_dense)

        # Accumulate across queries (max-pooling: keep the highest RRF score)
        for cid, score in fused.items():
            if cid not in pooled or score > pooled[cid]:
                pooled[cid] = score

    # Sort by pooled RRF score, take top_k_initial
    ranked = sorted(pooled.items(), key=lambda x: -x[1])
    top_ids = [cid for cid, _ in ranked[:top_k_initial]]
    print(f"[Phase A] RRF pool: {len(top_ids)} chunks (capped at {top_k_initial})")

    # Fetch full chunk data
    chunks = _fetch_chunks_by_ids(top_ids)

    # Rerank with bge-reranker-v2-m3
    # Use the original user question (queries[0]) for reranking
    print(f"[Phase A] Reranking {len(chunks)} chunks...")
    top_chunks = rerank_chunks(
        query=queries[0],  # rerank against the raw user question
        chunks=chunks,
        top_k=top_k_rerank,
        apply_authenticity_weight=False,  # Quran grade_weight is 1.0 anyway
    )

    # Compute confidence A
    status, confidence = compute_confidence(top_chunks, apply_authenticity_weight=False)
    print(f"[Phase A] Status: {status} (confidence={confidence:.3f})")

    return top_chunks, status, confidence
