"""
NUR configuration — loaded from environment variables.

All cloud APIs are FREE. Settings are documented in `.env.example`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
QURAN_DIR = DATA_DIR / "quran"
HADITH_DIR = DATA_DIR / "hadith"
TAFSIR_DIR = DATA_DIR / "tafsir"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_PATH = DATA_DIR / "chroma_db"
SPARSE_PATH = DATA_DIR / "sparse"


class Settings(BaseSettings):
    """NUR settings, loaded from `.env` file or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NUR_",
        extra="ignore",
    )

    # ----- LLM providers -----
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # ----- Model selection -----
    llm_primary: str = "qwen/qwen3-32b"
    llm_reasoning: str = "qwen/qwen3.6-27b"
    llm_local: str = "qwen2.5:7b"

    # ----- Embeddings -----
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # ----- Retrieval -----
    rrf_k: int = 25
    rrf_alpha_dense: float = 0.4

    top_k_initial: int = 30
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
    default_lang: Literal["en", "ar"] = "en"


# Singleton — import this everywhere
settings = Settings()
