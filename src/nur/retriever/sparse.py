"""
sparse.py

This file handles lexical (sparse) search over the NUR Islamic text collections.
It loads the precomputed BGE-M3 sparse vectors from JSON files and scores each
chunk against a query using dot-product on sparse vectors.

WHY THIS EXISTS (see docs/PILLARS.md Pillar 2 and docs/RAG_PIPELINE_ARCHITECTURE.md Step 2):
  Dense search alone misses exact keyword matches. A French user asking about
  "Riba" (usury) needs the lexical engine to catch that exact word, not just
  semantically nearby verses. Sparse search complements dense search, and the
  two are fused via Reciprocal Rank Fusion (RRF) in fusion.py.

HOW IT WORKS:
  - At ingestion time (scripts/colab_indexer.py), every chunk was encoded with
    BGE-M3 in sparse mode. The output is a dict {token_id: weight} per chunk,
    stored to JSON as {"indices": [...], "values": [...]} keyed by chunk ID.
  - At query time, the caller encodes the user's question with the same model
    and passes the resulting {token_id: weight} dict to SparseRetriever.search().
  - This class builds an INVERTED INDEX on load: token_id -> [(chunk_id, weight),
    ...]. Query scoring then iterates only over posting lists for tokens that
    appear in the query, making it O(sum of posting-list sizes) instead of
    O(all_chunks * avg_tokens_per_chunk). This is the standard sparse-retrieval
    data structure (same one BM25 uses).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.nur.config import SPARSE_PATH


class SparseRetriever:
    """Loads BGE-M3 sparse vectors from JSON and runs dot-product search.

    Each source (quran, hadith, tafsir_ar, tafsir_en) has its own JSON file
    under SPARSE_PATH. Files are loaded lazily on first query to that source
    and cached in memory for subsequent queries.
    """

    def __init__(self, sparse_path: str | Path = str(SPARSE_PATH)) -> None:
        """Set up the retriever with the path to the sparse JSON directory.

        Args:
            sparse_path: Directory containing {source}_sparse.json files.
                         Defaults to the project's data/sparse/ directory.
        """
        self.sparse_path = Path(sparse_path)
        # Inverted index per source: {source: {token_id: [(chunk_id, weight), ...]}}
        self._indexes: dict[str, dict[int, list[tuple[str, float]]]] = {}
        # Total chunk count per source (for diagnostics)
        self._chunk_counts: dict[str, int] = {}

    def _load_source(self, source: str) -> None:
        """Load a sparse JSON file and build the inverted index for one source.

        Reads {source}_sparse.json, which has the shape:
            {chunk_id: {"indices": [token_id, ...], "values": [weight, ...]}}

        Converts it into an inverted index:
            {token_id: [(chunk_id, weight), ...]}

        Args:
            source: One of 'quran', 'hadith', 'tafsir_ar', 'tafsir_en'.
        """
        # Skip if already loaded
        if source in self._indexes:
            return

        file_path = self.sparse_path / f"{source}_sparse.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Sparse index not found: {file_path}. "
                f"Run the ingestion pipeline (scripts/colab_indexer.py) first."
            )

        # Load the raw JSON. These files can be 100MB+, so this takes a few
        # seconds for hadith (33K chunks) but is instant for quran (6K).
        with file_path.open("r", encoding="utf-8") as f:
            raw: dict[str, dict[str, list[int] | list[float]]] = json.load(f)

        # Build the inverted index in one pass over all chunks.
        inverted: dict[int, list[tuple[str, float]]] = {}
        for chunk_id, vec in raw.items():
            indices = vec["indices"]
            values = vec["values"]
            # indices and values are parallel arrays of the same length
            for token_id, weight in zip(indices, values):
                # token_id comes from JSON as int (we stored them as int(k))
                inverted.setdefault(int(token_id), []).append((chunk_id, float(weight)))

        self._indexes[source] = inverted
        self._chunk_counts[source] = len(raw)
        print(f"✅ SparseRetriever loaded '{source}': {len(raw):,} chunks, "
              f"{len(inverted):,} unique tokens.")

    def search(
        self,
        query_sparse: dict[int, float],
        source: str,
        top_k: int = 30,
    ) -> list[dict[str, Any]]:
        """Score all chunks in a collection against the query sparse vector.

        The score is the dot product of the query sparse vector and each
        chunk's sparse vector. Only chunks that share at least one token with
        the query receive a non-zero score.

        Args:
            query_sparse: The query's sparse representation from BGE-M3, as a
                          {token_id: weight} dict. This is exactly what
                          model.encode(..., return_sparse=True)['lexical_weights'][0]
                          returns.
            source: The collection to search ('quran', 'hadith', 'tafsir_ar',
                    'tafsir_en').
            top_k: Maximum number of results to return.

        Returns:
            A list of dicts, each with keys 'id' (chunk_id) and 'score' (float),
            sorted by score descending. Chunks with zero score are excluded.
        """
        self._load_source(source)
        inverted = self._indexes[source]

        # Accumulate dot-product scores using the inverted index.
        # For each query token, walk its posting list and add
        # query_weight * chunk_weight to the running score for that chunk.
        scores: dict[str, float] = {}
        for token_id, query_weight in query_sparse.items():
            postings = inverted.get(int(token_id))
            if not postings:
                continue
            for chunk_id, chunk_weight in postings:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + query_weight * chunk_weight

        # Sort by score descending and truncate to top_k.
        # Chunks that shared no tokens with the query never entered `scores`
        # and are therefore excluded — which is the correct sparse behavior.
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [{"id": chunk_id, "score": score} for chunk_id, score in ranked]

    def get_chunk_count(self, source: str) -> int:
        """Return the number of chunks indexed for a given source.

        Triggers a lazy load if the source has not been queried yet.
        """
        self._load_source(source)
        return self._chunk_counts[source]
