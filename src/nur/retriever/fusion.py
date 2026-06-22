"""
fusion.py

This file fuses the ranked results from DenseRetriever and SparseRetriever into
a single ranked list using Reciprocal Rank Fusion (RRF).

WHY THIS EXISTS (see docs/RAG_PIPELINE_ARCHITECTURE.md Step 2):
  Dense and sparse retrievers return scores on completely different scales:
    - Dense similarity is bounded in [0, 1] (cosine of normalized vectors).
    - Sparse dot-product is unbounded (depends on how many tokens overlap and
      their weights).
  We cannot simply add or average the raw scores — that would let whichever
  retriever happens to produce larger numbers dominate. RRF solves this by
  operating on RANKS, not scores. The rank is just the position of a chunk in
  each retriever's result list (1st, 2nd, 3rd...).

THE RRF FORMULA:
  For each chunk c that appears in either retriever's top-K:
    score_rrf(c) = alpha * (1 / (k + rank_dense(c)))
                 + (1 - alpha) * (1 / (k + rank_sparse(c)))

  where:
    k      = 25  (smoothing constant — keeps the denominator non-zero and
                  controls how steeply rank decays. Lower k = steeper decay,
                  meaning rank #1 matters much more than rank #10.)
    alpha  = 0.4 (weight given to DENSE retrieval. Sparse gets 1 - alpha = 0.6,
                  so sparse slightly dominates. This is the project decision
                  recorded in docs/RAG_PIPELINE_ARCHITECTURE.md and
                  src/nur/config.py — exact keyword matches are weighted more
                  heavily because they are more trustworthy for Islamic text
                  where a single word like "Riba" or "Zakat" pinpoints a
                  concept unambiguously.)

  Chunks that appear in ONLY one retriever's list still get a score — they
  just miss one term of the sum. This is correct behavior: a chunk that ranks
  #1 in dense but isn't in sparse's top-K should still rank highly overall,
  just slightly penalized for not having lexical overlap with the query.

DEDUPLICATION:
  The same chunk ID may appear in both lists (the ideal case — it means both
  retrievers agree it's relevant). RRF naturally handles this: the chunk gets
  both the dense term and the sparse term added together, which boosts its
  final score. No explicit deduplication is needed; the score dict is keyed
  by chunk ID.
"""

from __future__ import annotations

from typing import Any

from src.nur.config import settings


class RRFFuser:
    """Fuses dense and sparse retrieval results via Reciprocal Rank Fusion.

    The fuser is stateless — it does not hold any cached data between calls.
    All state lives in the input arguments. This makes it trivial to test in
    isolation with synthetic ranked lists.
    """

    def __init__(
        self,
        k: int = settings.rrf_k,
        alpha_dense: float = settings.rrf_alpha_dense,
    ) -> None:
        """Configure the RRF constants.

        Args:
            k: Smoothing constant for the rank denominator. Default is 25
               (from settings, matching the architecture doc). Higher k
               flattens the score differences between ranks; lower k makes
               rank #1 dominate more.
            alpha_dense: Weight given to the dense retriever (0.0 to 1.0).
                         Sparse gets (1 - alpha_dense). Default is 0.4,
                         meaning sparse dominates at 0.6.
        """
        if k <= 0:
            raise ValueError(f"RRF k must be positive, got {k}.")
        if not 0.0 <= alpha_dense <= 1.0:
            raise ValueError(
                f"alpha_dense must be in [0.0, 1.0], got {alpha_dense}."
            )
        self.k = k
        self.alpha_dense = alpha_dense
        self.alpha_sparse = 1.0 - alpha_dense

    def fuse(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        top_k: int = settings.top_k_initial,
    ) -> list[dict[str, Any]]:
        """Fuse two ranked lists into one via RRF.

        Args:
            dense_results: Ranked list from DenseRetriever.search(). Each item
                           must have an 'id' key. Other keys (document,
                           metadata, similarity) are ignored by the fusion
                           math but may be attached to the output for
                           downstream convenience.
            sparse_results: Ranked list from SparseRetriever.search(). Each
                            item must have an 'id' key and a 'score' key.
            top_k: Maximum number of fused results to return. Default is 30
                   (from settings.top_k_initial), matching the architecture
                   doc's "collect roughly 30 unique chunks" step.

        Returns:
            A list of dicts sorted by RRF score descending, each with keys:
              - 'id': the chunk ID
              - 'rrf_score': the fused score (float)
              - 'dense_rank': rank in the dense list (None if absent)
              - 'sparse_rank': rank in the sparse list (None if absent)
            Truncated to top_k items.
        """
        # Build rank maps: chunk_id -> 1-indexed rank position.
        # Rank 1 = first result, rank 2 = second, etc.
        # This matches the standard RRF convention.
        dense_rank: dict[str, int] = {res["id"]: i + 1 for i, res in enumerate(dense_results)}
        sparse_rank: dict[str, int] = {res["id"]: i + 1 for i, res in enumerate(sparse_results)}

        # Union of all chunk IDs that appeared in either list.
        # A chunk only in dense gets sparse_rank = None (skipped in the sparse term).
        # A chunk only in sparse gets dense_rank = None (skipped in the dense term).
        all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())

        # Compute RRF score for each chunk.
        # Chunks missing from one list simply don't get that term added —
        # they are NOT penalized with a synthetic low rank. This is the
        # standard RRF behavior.
        fused: list[dict[str, Any]] = []
        for chunk_id in all_ids:
            score = 0.0
            d_rank = dense_rank.get(chunk_id)
            s_rank = sparse_rank.get(chunk_id)

            if d_rank is not None:
                score += self.alpha_dense * (1.0 / (self.k + d_rank))
            if s_rank is not None:
                score += self.alpha_sparse * (1.0 / (self.k + s_rank))

            fused.append(
                {
                    "id": chunk_id,
                    "rrf_score": score,
                    "dense_rank": d_rank,
                    "sparse_rank": s_rank,
                }
            )

        # Sort by RRF score descending. Ties (same score) are broken by
        # chunk ID for deterministic output — important for testing.
        fused.sort(key=lambda x: (-x["rrf_score"], x["id"]))

        return fused[:top_k]
