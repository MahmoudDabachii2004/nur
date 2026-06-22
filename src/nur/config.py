"""
config.py

This file holds every tunable knob in the NUR pipeline. It loads values from a
local `.env` file (gitignored) so that secrets like API keys never end up in the
repository. The two source-of-truth documents that dictate the defaults here are:
  - docs/RAG_PIPELINE_ARCHITECTURE.md  (the LLM lineup and pipeline steps)
  - docs/PILLARS.md                    (theological rules: weighting, languages)
If either document changes, this file MUST be updated to match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project paths — resolved relative to THIS file so CWD never matters.
# config.py lives at src/nur/config.py, so parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
QURAN_DIR = DATA_DIR / "quran"
HADITH_DIR = DATA_DIR / "hadith"
TAFSIR_DIR = DATA_DIR / "tafsir"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_PATH = DATA_DIR / "chroma_db"
SPARSE_PATH = DATA_DIR / "sparse"


class Settings(BaseSettings):
    """NUR settings, loaded from `.env` file or environment variables.

    Most fields use the `NUR_` prefix (e.g. `NUR_LLM_PRIMARY`). The three
    API-key / SDK-standard fields below ALSO accept the unprefixed form so we
    stay compatible with the official Groq and OpenAI client conventions
    (`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`).
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="NUR_",
        extra="ignore",
    )

    # ----- LLM providers -----
    # Groq is the primary cloud host. Free tier, fast inference.
    # Accepts both `NUR_GROQ_API_KEY` and the SDK-standard `GROQ_API_KEY`.
    groq_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("NUR_GROQ_API_KEY", "GROQ_API_KEY"),
    )
    # OpenRouter is kept as a secondary cloud fallback (Phase 9+). Not wired yet.
    openrouter_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("NUR_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
    )
    # Ollama runs on the local PC and is used when Groq returns 429 (rate limit).
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("NUR_OLLAMA_BASE_URL", "OLLAMA_BASE_URL"),
    )

    # ----- Model selection (see docs/RAG_PIPELINE_ARCHITECTURE.md Section 3) -----
    # Architect  — Step 1, fast query decomposition. Cheap, high-volume.
    llm_architect: str = "llama-3.1-8b-instant"
    # Reporter   — Step 4, primary structured generation. 30K TPM on Groq free tier.
    llm_primary: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    # Reporter fallback for extreme Ikhtilaf cases. 12K TPM, deeper reasoning.
    llm_reasoning: str = "llama-3.3-70b-versatile"
    # Offline fallback when Groq 429s. Runs on local Ollama (RX 5700 XT).
    llm_local: str = "llama3.1:8b"

    # ----- Embeddings -----
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ----- Retrieval -----
    # RRF fusion parameters (see docs/RAG_PIPELINE_ARCHITECTURE.md Step 2)
    rrf_k: int = 25
    rrf_alpha_dense: float = 0.4

    # Pool sizes:
    # - top_k_initial: how many chunks to retrieve per (query, source) pair
    #   before dedup. Set to 400 (DEC-034) based on empirical recall testing:
    #     pool=100 → 29% recall, pool=200 → 29%, pool=300 → 36%,
    #     pool=400 → 50%, pool=500 → 50% (no improvement past 400).
    #   The reranker then scores all 400 chunks (~20-40s on M4 MPS) and keeps
    #   the top 10. Key verses like 2:43 ("establish prayer") are found at
    #   rank 349 — only reachable with pool >= 400.
    # - top_k_rerank: how many chunks to send to the LLM after reranking.
    #   Capped at 10 by Groq's 30K TPM limit (10 chunks × 1,660 tokens = 16,600
    #   tokens = 59% of TPM per query).
    top_k_initial: int = 400
    top_k_rerank: int = 10
    top_k_final: int = 5

    # ----- Authenticity weighting -----
    grade_weight_sahih: float = 1.30
    grade_weight_hasan: float = 1.10
    grade_weight_daif: float = 0.50
    grade_weight_mawdu: float = 0.00

    # ----- Generation -----
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_frequency_penalty: float = 0.0
    llm_repetition_penalty: float = 1.1

    # ----- Verification -----
    nli_model: str = "cross-encoder/nli-deberta-v3-large"
    nli_threshold: float = 0.95

    # ----- Misc -----
    log_level: str = "INFO"
    # Synthesis language for the Reporter LLM. Arabic text is ALWAYS displayed
    # alongside (Pillar 10) — this toggle only controls the FR/EN explanation.
    # 'ar' is intentionally excluded: Arabic is the source of truth, not a
    # synthesis language.
    default_lang: Literal["en", "fr"] = "en"


# Singleton — import this everywhere
settings = Settings()
