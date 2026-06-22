"""
NUR V3 Runtime — Reranker module

Uses bge-reranker-v2-m3 cross-encoder to score (query, chunk) pairs.
Implements Pillar 3 (Authenticity-Weighted Retrieval) by multiplying
reranker score by grade_weight.

Per docs/v3/04_RETRIEVAL_PIPELINE.md Step 2e + 4e.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from nur.config import settings  # noqa: E402

# Lazy-loaded singleton
_RERANKER = None


def _load_reranker():
    """Lazy-load bge-reranker-v2-m3 model. Cached as singleton."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER

    try:
        from FlagEmbedding import FlagReranker
    except ImportError as e:
        raise ImportError(
            "FlagEmbedding not installed. Run: pip install FlagEmbedding torch"
        ) from e

    print("[Reranker] Loading bge-reranker-v2-m3...")
    _RERANKER = FlagReranker(
        settings.reranker_model,
        use_fp16=True,  # faster on GPU, also works on CPU
    )
    print("[Reranker] Loaded.")
    return _RERANKER


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    apply_authenticity_weight: bool = True,
) -> list[dict]:
    """Rerank chunks by (query, chunk_text) relevance.

    Args:
        query: User's question (or sub-question).
        chunks: List of chunk dicts. Each must have "embedding_text" key.
        top_k: Number of top chunks to return.
        apply_authenticity_weight: If True, multiply score by grade_weight
            (Sahih 1.30, Hasan 1.10, Da'if 0.50, Mawdu 0.00).

    Returns:
        List of top_k chunks sorted by final_score desc.
        Each chunk dict gets "rerank_score" and "final_score" added.
    """
    if not chunks:
        return []

    reranker = _load_reranker()

    # Build (query, chunk_text) pairs
    pairs = [(query, c.get("embedding_text", "")) for c in chunks]

    # Score all pairs. normalize=True gives sigmoid scores in [0, 1]
    scores = reranker.compute_score(pairs, normalize=True)

    # Handle single pair case (FlagReranker returns scalar)
    if len(chunks) == 1:
        scores = [scores]
    elif not isinstance(scores, list):
        scores = list(scores)

    # Attach scores
    for chunk, score in zip(chunks, scores):
        rerank_score = float(score)
        grade_weight = (
            chunk.get("metadata", {}).get("grade_weight", 1.0)
            if apply_authenticity_weight
            else 1.0
        )
        final_score = rerank_score * grade_weight
        chunk["rerank_score"] = rerank_score
        chunk["final_score"] = final_score

    # Sort by final_score desc
    ranked = sorted(chunks, key=lambda c: -c["final_score"])
    return ranked[:top_k]


def compute_confidence(ranked_chunks: list[dict], apply_authenticity_weight: bool = True) -> tuple[str, float]:
    """Compute confidence level for a phase.

    Returns (status, score) where:
      status: "STRONG" | "WEAK" | "EMPTY"
      score:  the max rerank_score (or final_score if authenticity-weighted)

    Thresholds (per docs/v3/04_RETRIEVAL_PIPELINE.md Step 2f):
      STRONG ≥ 0.5
      WEAK   > 0.3
      EMPTY  ≤ 0.3
    """
    if not ranked_chunks:
        return "EMPTY", 0.0

    if apply_authenticity_weight:
        max_score = max(c.get("final_score", 0) for c in ranked_chunks)
    else:
        max_score = max(c.get("rerank_score", 0) for c in ranked_chunks)

    if max_score >= 0.5:
        return "STRONG", max_score
    elif max_score > 0.3:
        return "WEAK", max_score
    else:
        return "EMPTY", max_score
