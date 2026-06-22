"""
NUR V3 — Runtime pipeline package.

Implements the V3 retrieval-augmented generation pipeline per docs/v3/.

Modules:
  architect        — Groq query decomposition (Step 1)
  retriever_quran  — Phase A: dense+sparse+RRF+rerank on quran_v3
  retriever_hadith — Phase B: dense+sparse+RRF+rerank on hadith_v3 + authenticity weighting
  cross_refs       — Auto-pull hadiths from Quran chunk metadata (Canal 1)
  reranker         — bge-reranker-v2-m3 cross-encoder + authenticity weighting
  reporter         — Groq structured JSON generation (Step 5)
  verifier         — NLI + Quran char-by-char + source ID validation (Step 6)
  pipeline         — Orchestrator with confidence gates (Step 2-4)
"""
from __future__ import annotations

__version__ = "3.0.0"
