
# NUR (نور) — Project Context & Master Overview

> **نور** — "Light" in Arabic. A profound Quranic concept (Surah An-Nur, 24:35).
> This document provides the high-level vision, ethical boundaries, and technical context of the NUR project. 
> It is the entry point for any AI agent or engineer working on the codebase.

---

## 1. Project Vision

NUR is an Islamic Retrieval-Augmented Generation (RAG) chatbot designed to answer any question about Islam based **exclusively** on the Quran, authentic Hadith, recognized Tafsir, and scholarly opinions. 

### Core Principles (Non-Negotiable)
- **$0 Cost**: Everything must run on free tiers (Groq, OpenRouter) or locally (Ollama).
- **Absolute Reliability**: A wrong religious answer is worse than no answer. The system must abstain if it lacks confidence.
- **Transparency**: Every claim must be traceable to a clickable source (quran.com, sunnah.com, etc.).
- **Arabic-First**: Arabic is the source of truth. Translations are comprehension aids.
- **Bilingual Interface**: The UI defaults to English. Users can toggle to French. Arabic text is *always* displayed alongside citations.
- **Scholar Grounded**: Laymen (and LLMs) cannot interpret scriptures alone. The system must report scholarly opinions, never generate its own.

---

## 2. The Problem with Existing Solutions

Existing Islamic AI projects suffer from critical flaws that NUR is engineered to solve:
1. **Destructive Chunking**: Blindly splitting Quranic verses or Hadith chains (isnad) by character count.
2. **Flawed Embeddings**: Using English-centric models for Arabic text, causing poor semantic matching.
3. **Ignored Hadith Grading**: Failing to filter or weight results by authenticity (Sahih vs. Da'if).
4. **Zero Verification**: Allowing LLMs to hallucinate Quranic verses without post-generation character checks.
5. **Mixed Indexes**: Storing the Word of Allah and human commentary in the same vector space without distinction.

---

## 3. High-Level Technical Architecture

### Data Ingestion (Local)
- **Database**: ChromaDB (4 separate collections: Quran, Hadith, Tafsir AR, Tafsir EN).
- **Sparse Index**: Custom JSON files mapping lexical weights for hybrid search.
- **Embedding Model**: `BAAI/bge-m3` running locally. It places Arabic, French, and English in the same vector space for cross-lingual matching.
- **Context Enrichment**: Chunks are prefixed with LLM-generated bilingual index cards (Theme, Rule, Keywords) to solve cross-lingual sparse match failures.

### RAG Pipeline (Local + Cloud)
- **Retriever**: Parallel Dense (ChromaDB) and Sparse (JSON dot-product) search, fused using Reciprocal Rank Fusion (RRF).
- **Reranker**: `bge-reranker-v2-m3` cross-encoder running locally on Apple Silicon (MPS) to refine top results.
- **Generator**: Cloud LLM (Groq - Llama 4 Scout / Llama 3.3 70B) with strict JSON Schema output. Falls back to OpenRouter, then to local Ollama for offline mode.
- **Verification**: Post-generation NLI (Natural Language Inference) check and character-by-character validation of Quranic citations against the Uthmani script.

---

## 4. Current Project State

- **Phase 1 (Data)**: ✅ Completed. The contextual ingestion pipeline (`nur_synthesizer_cloud.py`) was executed on Lightning AI using `Qwen2.5-14B-Instruct-AWQ`. The local database is verified and benchmarked successfully.
- **Phase 2 (RAG Pipeline)**: 🚧 In Progress. Building the local Python CLI to query the database and generate structured answers via Groq.

---

## 5. Guiding Documents (Mandatory Reading)

For exact implementation rules and step-by-step execution, refer strictly to the following documents:

- **`docs/PILLARS.md`**: Read this for the exact functional and theological rules the system must follow (e.g., Authenticity Weighting, Ikhtilaf Awareness, Structured Citation Protocol). Do not deviate from these pillars.
- **`docs/PHASES.md`**: Read this for the engineering roadmap, current task checklists, and technical key decisions (e.g., RRF parameters, LLM temperature constraints). 
- **`AGENTS.md`**: Read this for the Software Engineering Rules (KISS, SOLID, baby steps, no hallucinated APIs) that must govern how the code is written.