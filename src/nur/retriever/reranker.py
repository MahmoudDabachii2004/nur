"""
reranker.py

This file implements the cross-encoder reranker for NUR — the precision layer
that sits between the retriever (recall) and the LLM (generation).

WHY THIS EXISTS (see docs/RAG_PIPELINE_ARCHITECTURE.md Step 3):
  The retriever (dense + sparse + RRF) finds CANDIDATES — chunks that are
  semantically related to the query. But it ranks them by surface similarity,
  which is unreliable for Islamic text. Example: for "Is prayer obligatory?",
  the retriever ranks tafsirs that say "prière obligatoire" (FR) higher than
  Quran verses that say "establish prayer" (EN) — even though both are equally
  relevant, because the French word "obligatoire" matches "obligatory" while
  "establish" does not.

  The cross-encoder reranker solves this by concatenating [query] + [chunk] into
  a single sequence and using transformer attention to compute a TRUE relevance
  score. It understands that "establish prayer" IS about "prayer obligation" —
  not because of hardcoded rules, but because it was trained on millions of
  query-document pairs.

  This is the critical anti-hallucination layer per Pillar 4. We do not trust
  the LLM to evaluate if it has "enough" information. We trust the cross-encoder.

HOW IT WORKS:
  - Model: BAAI/bge-reranker-v2-m3 (568M params, multilingual)
  - Input: a list of [query, chunk_text] pairs
  - Output: a relevance score for each pair, normalized to [0, 1] via sigmoid
  - The sigmoid normalization allows a meaningful abstention threshold (0.35)

AUTHENTICITY WEIGHTING (Pillar 3):
  After the reranker scores each chunk, we multiply the score by the hadith
  grade weight (Sahih ×1.3, Hasan ×1.1, Da'if ×0.5, Mawdu ×0). This ensures
  that a relevant Sahih hadith beats a slightly-more-relevant Da'if one.
  Quran and Tafsir sources get a neutral weight of 1.0 (they are not graded).

LAZY LOADING:
  The reranker model (568M params, ~1.2GB) is loaded on first use, not at
  construction time. This keeps the pipeline fast to instantiate.
"""

from __future__ import annotations

from typing import Any

from FlagEmbedding import FlagReranker

from src.nur.config import settings
from src.nur.grades import get_grade_info
from src.nur.sources import SourceRef


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model.

    The reranker takes a user query and a list of chunks (each with its text
    and metadata), scores each chunk's relevance to the query, optionally
    applies authenticity weighting, and returns the top-K chunks sorted by
    final score.

    The model is lazy-loaded on first call to `rerank()`.
    """

    def __init__(
        self,
        model_name: str = settings.reranker_model,
        device: str | None = None,
    ) -> None:
        """Configure the reranker. The model is NOT loaded here.

        Args:
            model_name: HuggingFace model ID. Default: BAAI/bge-reranker-v2-m3.
            device: Device for inference ('mps', 'cuda', 'cpu'). If None,
                    auto-detects the best available device.
        """
        self.model_name = model_name
        self._device = device or self._auto_detect_device()
        self._model: FlagReranker | None = None

    @staticmethod
    def _auto_detect_device() -> str:
        """Pick the best available device. Priority: CUDA > MPS > CPU."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def model(self) -> FlagReranker:
        """Lazily load and cache the reranker model on first access.

        Loading takes ~2-5 seconds (568M weights from local cache).
        Subsequent accesses return the cached instance.
        """
        if self._model is None:
            print(f"Loading reranker ({self.model_name}) on {self._device}...")
            # use_fp16=True on GPU/MPS for speed; False on CPU for stability
            use_fp16 = self._device != "cpu"
            self._model = FlagReranker(
                self.model_name,
                use_fp16=use_fp16,
            )
            print(f"✅ Reranker loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        source_refs: list[SourceRef] | None = None,
        top_k: int = settings.top_k_rerank,
        apply_authenticity_weight: bool = True,
        normalize: bool = True,
    ) -> list[dict[str, Any]]:
        """Score and rerank chunks by true relevance to the query.

        For each chunk, computes:
          1. A cross-encoder relevance score (0.0 to 1.0 via sigmoid)
          2. (Optional) An authenticity weight based on hadith grade
          3. A final score = relevance_score × authenticity_weight

        Then sorts by final score descending and returns the top-K.

        Args:
            query: The user's original question.
            chunks: List of chunk dicts from the retriever. Each must have
                    'id', 'source', and 'document' (the text to score).
            source_refs: Optional list of SourceRef objects (same order as
                         chunks) for grade lookup. If None, no authenticity
                         weighting is applied.
            top_k: Number of chunks to return after reranking. Default: 10
                   (TPM-limited: 10 chunks × 1,660 tokens = 16,600 tokens =
                   59% of Scout's 30K TPM).
            apply_authenticity_weight: If True (default), multiply the
                                       reranker score by the hadith grade
                                       weight. Quran/Tafsir get 1.0.
            normalize: If True (default), apply sigmoid normalization to
                       get scores in [0, 1]. The 0.35 abstention threshold
                       requires normalized scores.

        Returns:
            A list of dicts sorted by final_score descending, each with:
              - 'id': chunk ID
              - 'source': source type
              - 'reranker_score': raw cross-encoder score (sigmoid normalized)
              - 'authenticity_weight': the grade weight applied (1.0 if none)
              - 'final_score': reranker_score × authenticity_weight
              - 'rrf_score': the original RRF score (for reference)
              - 'rrf_rank': the original RRF rank (1-indexed)
            Truncated to top_k items.
        """
        if not chunks:
            return []

        # Build the [query, chunk_text] pairs for the cross-encoder.
        # We use the 'document' field which contains the full chunk text
        # (including the bilingual context prefix from Phase 1).
        pairs = []
        for chunk in chunks:
            doc = chunk.get("document", "")
            if not doc:
                # If no document text, use a placeholder — the reranker
                # will give it a low score, which is correct behavior.
                doc = "(empty)"
            pairs.append([query, doc])

        # Score all pairs in one batch call. normalize=True applies sigmoid
        # so scores are in [0, 1] — required for the 0.35 abstention threshold.
        print(f"  Reranking {len(pairs)} chunks...")
        scores = self.model.compute_score(pairs, normalize=normalize)

        # compute_score returns a list for multiple pairs, or a float for one.
        if isinstance(scores, (int, float)):
            scores = [scores]

        # Build a lookup of source_refs by chunk ID for grade lookup
        ref_map: dict[str, SourceRef] = {}
        if source_refs:
            for ref in source_refs:
                # SourceRef doesn't store chunk_id directly, but we can
                # reconstruct it from the source_id. However, it's simpler
                # to match by index if the caller ensures alignment.
                # For now, we skip grade weighting if source_refs don't
                # align with chunks by index.
                pass

        # Assemble the results
        results: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            reranker_score = float(scores[i]) if i < len(scores) else 0.0

            # Authenticity weighting (Pillar 3)
            authenticity_weight = 1.0
            if apply_authenticity_weight and source_refs and i < len(source_refs):
                ref = source_refs[i]
                if ref.kind == "hadith" and ref.grade:
                    grade_info = get_grade_info(ref.grade)
                    # Map grade level to weight from settings
                    level = grade_info["level"]
                    if level == "sahih":
                        authenticity_weight = settings.grade_weight_sahih
                    elif level == "hasan":
                        authenticity_weight = settings.grade_weight_hasan
                    elif level == "daif":
                        authenticity_weight = settings.grade_weight_daif
                    elif level == "mawdu":
                        authenticity_weight = settings.grade_weight_mawdu

            final_score = reranker_score * authenticity_weight

            results.append({
                "id": chunk["id"],
                "source": chunk["source"],
                "reranker_score": reranker_score,
                "authenticity_weight": authenticity_weight,
                "final_score": final_score,
                "rrf_score": chunk.get("rrf_score", 0.0),
                "rrf_rank": chunk.get("rrf_rank", 0),
            })

        # Sort by final_score descending
        results.sort(key=lambda x: -x["final_score"])

        return results[:top_k]

    def should_abstain(
        self,
        reranked_results: list[dict[str, Any]],
        threshold: float = 0.35,
    ) -> bool:
        """Check if the pipeline should abstain from answering (Pillar 4).

        Per docs/RAG_PIPELINE_ARCHITECTURE.md Step 3:
          "If the Top 1 chunk score < 0.35, abort generation. Return
           'I do not have sufficient reliable sources to answer this question.'"

        This is the mathematical abstention gate. We do not trust the LLM
        to decide if it has enough information — we trust the cross-encoder.

        Args:
            reranked_results: The output of rerank(), sorted by final_score.
            threshold: The minimum reranker_score (NOT final_score) for the
                       top chunk. Default: 0.35 (from the architecture doc).

        Returns:
            True if the pipeline should abstain (top chunk score < threshold),
            False if it should proceed with generation.
        """
        if not reranked_results:
            return True

        top_score = reranked_results[0].get("reranker_score", 0.0)
        return top_score < threshold
