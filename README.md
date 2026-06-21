# نور NUR

> **NUR** (نور) — "Light" in Arabic. A Quranic concept (Surah An-Nur, 24:35).

An Islamic RAG chatbot that answers questions about Islam based **exclusively** on the Quran, authentic Hadith, Tafsir, and recognized scholarly opinions. Every answer must cite its sources with clickable links.

**Arabic-first · English-supported · 100% Free · Scholar-grounded**

---

## What is NUR?

A retrieval-augmented chatbot that:

- Treats the **Arabic Quran as the source of truth** (translations are for comprehension only)
- Indexes **Hadith with grading** (Sahih / Hasan / Daʿif / Mawḍūʿ) and weights retrieval by authenticity
- Surfaces **scholarly opinions** (never gives its own)
- Detects **ikhtilāf** (scholarly disagreement) and presents all views with absolute neutrality
- Verifies **every Quranic citation character-by-character** against the Uthmani text
- Cites **clickable sources** (quran.com / sunnah.com / islamqa.info)

## Principles (Non-Negotiable)

1. **$0 cost** — everything runs on free tiers (Groq, OpenRouter) or locally (Ollama)
2. **Reliability** — a wrong religious answer is worse than no answer
3. **Transparency** — every claim is traceable to its source
4. **Scholarly opinions required** — laymen cannot interpret verses/hadiths alone
5. **Arabic-first** — Arabic is the source of truth, English is a translation aid
6. **Hadith grading respected** — Daʿif flagged with warning, Mawḍūʿ detected as fabricated
7. **Hybrid search** — Dense (semantic) + Sparse (keyword) via BGE-M3 with RRF fusion
8. **Post-generation verification** — character-level check for Quran, NLI for Hadith

## Architecture

The complete architecture, decisions, research, and references are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (1,700+ lines). The 10 pillars are summarized in [`docs/PILLARS.md`](docs/PILLARS.md).

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM (cloud) | Groq — Qwen 3 32B | Best Arabic support on free tier (60 RPM, OALL top-10) |
| LLM (local) | Ollama — Qwen2.5-7B (Q4_K_M) | Offline / privacy fallback (~15 tok/s on M4) |
| Embeddings | BAAI/bge-m3 (FlagEmbedding) | Dense + sparse + ColBERT in one model, 8192 ctx, AR↔EN cross-lingual |
| Reranker | bge-reranker-v2-m3 | Best multilingual cross-encoder supporting Arabic |
| Vector DB | ChromaDB (dev) → Qdrant (prod) | Metadata filtering for type/source/grade |
| RAG Framework | LangGraph (orchestration) + Custom Islamic core + LangChain (plumbing) |
| Backend | FastAPI | Async, Python-native |
| Frontend | Next.js PWA | Installable on mobile without App Store |
| Embedding runtime | Google Colab T4 (free) | One-time heavy compute, ~40 min for 52K chunks |

## Data Sources (all free)

| Source | Content | Format |
|--------|---------|--------|
| alquran.cloud API | Quran — Arabic Uthmani + English Saheeh International | REST JSON |
| meeAtif/hadith_datasets | Kutub al-Sittah — 33,738 hadiths, AR+EN, grades, sunnah.com URLs | HuggingFace JSON |
| spa5k/tafsir_api | Tafsir Ibn Kathir (AR + EN) | CDN JSON via jsDelivr |

## Project Structure

```
nur/
├── docs/
│   ├── ARCHITECTURE.md     # Master document — all decisions & research
│   ├── PILLARS.md          # 10-pillar summary
│   ├── SETUP.md            # Mac M4 setup instructions
│   └── PHASES.md           # 12-phase roadmap
├── scripts/                # Data pipeline (Phase 1)
│   ├── 01_download_quran.py
│   ├── 02_download_hadith.py
│   ├── 03_download_tafsir.py
│   ├── 04_normalize_and_chunk.py
│   └── 05_verify_phase1.py
├── colab/
│   ├── README.md           # How to run the Colab embedding
│   └── embed_nur_colab.py  # BGE-M3 embedding script (T4 GPU, ~40 min)
├── data/                   # gitignored — downloaded at runtime
│   ├── quran/              # 6,236 ayahs (AR + EN)
│   ├── hadith/             # 33,738 hadiths (Kutub al-Sittah)
│   ├── tafsir/             # Ibn Kathir AR + EN
│   ├── processed/          # JSONL chunks ready for embedding
│   ├── chroma_db/          # ChromaDB persistent store (4 collections)
│   └── sparse/             # BGE-M3 sparse vectors (JSON)
├── src/nur/                # Main application source (Phase 2+)
│   ├── __init__.py
│   ├── config.py
│   ├── arabic.py           # Arabic text normalization
│   ├── sources.py          # Source ID protocol + clickable URLs
│   ├── retriever/          # Dense + Sparse + RRF fusion
│   ├── generator.py        # Groq + Ollama with fallback
│   └── verification/       # Post-generation anti-hallucination
└── tests/
```

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | 🚧 In Progress | Data download + normalization + chunking + Colab embedding |
| 2 | Planned | RAG pipeline (retrieval + Groq generation + source IDs) |
| 3 | Planned | Hybrid search (RRF) + bge-reranker-v2-m3 |
| 4 | Planned | Context-enriched chunks (Anthropic technique) |
| 5 | Planned | Authenticity weighting + Mawḍūʿ detection |
| 6 | Planned | Post-generation verification (NLI + char-by-char Quran) |
| 7 | Planned | Scholar opinions + Ikhtilāf awareness |
| 8 | Planned | Cross-references Quran ↔ Hadith |
| 9 | Planned | Next.js PWA frontend |
| 10 | Planned | Advanced RAG (CRAG, query decomposition) |
| 11 | Planned | Evaluation (RAGAS + Islamic metrics) |
| 12 | Planned | Voice I/O (Whisper) + Prompt Guard |

See [`docs/PHASES.md`](docs/PHASES.md) for detailed phase breakdown.

## Quick Start (Mac M4)

```bash
# 1. Clone
git clone https://github.com/MahmoudDabachii2004/nur.git
cd nur

# 2. Create venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env — add your Groq API key (https://console.groq.com)

# 5. Download data (Phase 1)
python scripts/01_download_quran.py
python scripts/02_download_hadith.py
python scripts/03_download_tafsir.py
python scripts/04_normalize_and_chunk.py

# 6. Embed (on Google Colab T4 — see colab/README.md)
# Upload data/processed/*.jsonl to Colab, run embed_nur_colab.py
# Download data/chroma_db/ and data/sparse/ back to your Mac
```

See [`docs/SETUP.md`](docs/SETUP.md) for detailed Mac M4 setup including Ollama, Groq API key, and Colab embedding workflow.

## License

MIT
