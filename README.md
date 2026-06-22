# نور NUR

> **NUR** (نور) — "Light" in Arabic. A Quranic concept (Surah An-Nur, 24:35).

An Islamic RAG chatbot that answers questions about Islam based **exclusively** on the Quran, authentic Hadith, Tafsir, and recognized scholarly opinions. Every answer must cite its sources with clickable links.

**Arabic-first · English-supported · 100% Free · Scholar-grounded**

---

## Current Status: V3 (2026-06-23)

V3 is a complete refoundation of the data ingestion + RAG pipeline. It fixes 5 structural flaws of V1/V2:

| Flaw V1/V2 | V3 Fix |
|------------|--------|
| Tafsir orphan (4 separate collections) | Tafsir integrated in Quran chunk (2 collections) |
| Global ayah numbering (broken URLs) | Standard surah:ayah (correct URLs) |
| Inter-collection competition | 2-phase sequential retrieval |
| No semantic bridge for modern questions | Tafsir IS the bridge ("smoking" → 2:195 via Ibn Kathir "self-harm") |
| No cross-lingual context card | LLM-generated trilingual Context Card (FR/EN/AR keywords) |

**V3 Build Complete** ✅
- 6,236 Quran chunks (3-layer: Context Card + Word of Allah + 4 Tafsirs)
- 33,738 Hadith chunks (2-layer: mini context + Hadith text)
- 6,236 Context Cards (100% coverage, generated via Qwen2.5-14B-AWQ on Lightning AI)
- ChromaDB: `quran_v3_dense` (6,236 docs) + `hadith_v3_dense` (33,738 docs)
- 197 Quran chunks with hadith cross-refs (257 total refs)

**V3 Runtime** 🚧
- `src/nur/v3/` — 9 modules (architect, retriever_quran, retriever_hadith, cross_refs, reranker, reporter, verifier, pipeline, cli)
- Pending end-to-end validation with 5 ground-truth examples

## What is NUR?

A retrieval-augmented chatbot that:

- Treats the **Arabic Quran as the source of truth** (translations are for comprehension only)
- Indexes **Hadith with grading** (Sahih / Hasan / Daʿif / Mawḍūʿ) and weights retrieval by authenticity
- Surfaces **scholarly opinions** (never gives its own)
- Detects **ikhtilāf** (scholarly disagreement) and presents all views with absolute neutrality
- Verifies **every Quranic citation character-by-character** against the Uthmani text
- Cites **clickable sources** (quran.com / sunnah.com / islamqa.info)

## Principles (Non-Negotiable)

1. **$0 cost** — everything runs on free tiers (Groq, Lightning AI, Kaggle) or locally (Ollama)
2. **Reliability** — a wrong religious answer is worse than no answer
3. **Transparency** — every claim is traceable to its source
4. **Scholarly opinions required** — laymen cannot interpret verses/hadiths alone
5. **Arabic-first** — Arabic is the source of truth, English/French are translation aids
6. **Hadith grading respected** — Daʿif flagged with warning, Mawḍūʿ detected as fabricated
7. **Hybrid search** — Dense (semantic) + Sparse (keyword) via BGE-M3 with RRF fusion
8. **Post-generation verification** — character-level check for Quran, NLI for Hadith

## V3 Architecture

The V3 architecture is documented in [`docs/v3/`](docs/v3/) (9 documents, ~3,700 lines):

| Doc | Subject |
|-----|---------|
| [`00_OVERVIEW.md`](docs/v3/00_OVERVIEW.md) | V3 vision + 5 V1/V2 failures + 4 V3 pillars |
| [`01_DATA_SOURCES.md`](docs/v3/01_DATA_SOURCES.md) | Quran (alquran.cloud) + Hadith (meeAtif) + 4 Tafsirs (spa5k) |
| [`02_CHUNK_SCHEMA.md`](docs/v3/02_CHUNK_SCHEMA.md) | 3-layer Quran chunks + 2-layer Hadith chunks + cross-refs |
| [`03_TAFSIR_STRATEGY.md`](docs/v3/03_TAFSIR_STRATEGY.md) | Multi-tafsir (Ibn Kathir EN+AR, Tabari, Sa'di) + Ikhtilaf detection |
| [`04_RETRIEVAL_PIPELINE.md`](docs/v3/04_RETRIEVAL_PIPELINE.md) | 2-phase sequential retrieval + confidence gate |
| [`05_EMBEDDING_DESIGN.md`](docs/v3/05_EMBEDDING_DESIGN.md) | Data-driven tafsir truncation analysis (600 chars) |
| [`06_GENERATION_VERIFICATION.md`](docs/v3/06_GENERATION_VERIFICATION.md) | Reporter LLM + 4 anti-hallucination checks |
| [`07_FAILURE_MODES.md`](docs/v3/07_FAILURE_MODES.md) | 8 V1/V2 flaws + V3 fixes |
| [`08_EXAMPLES.md`](docs/v3/08_EXAMPLES.md) | 5 ground-truth end-to-end examples |

The 10 theological Pillars remain in [`docs/PILLARS.md`](docs/PILLARS.md).

## Tech Stack (V3)

| Component | Technology | Role |
|-----------|-----------|------|
| **Context Card generation (build)** | Qwen2.5-14B-Instruct-AWQ via vLLM | Lightning AI L40S — one-time build |
| **Embeddings (build)** | BAAI/bge-m3 (1024 dim, multilingual) | Lightning AI L40S + Kaggle T4×2 |
| **Architect (runtime)** | Groq llama-3.1-8b-instant | Query decomposition |
| **Reporter (runtime)** | Groq llama-4-scout-17b-16e-instruct | Structured JSON answer |
| **Offline fallback** | Ollama llama3.1:8b | When Groq 429s |
| Reranker | bge-reranker-v2-m3 | Cross-encoder for top-5 |
| Vector DB | ChromaDB | 2 collections (quran_v3, hadith_v3) |
| Sparse index | JSON (BGE-M3 lexical weights) | BM25-like dot product |

## Data Sources (all free)

| Source | Content | V3 Usage |
|--------|---------|----------|
| alquran.cloud API | Quran — Arabic Uthmani + English Saheeh International | Layer 2 (Word of Allah) |
| meeAtif/hadith_datasets (HuggingFace) | Kutub al-Sittah — 33,738 hadiths, AR+EN, grades, sunnah.com URLs | hadith_v3 collection |
| spa5k/tafsir_api (GitHub) | 4 tafsirs: Ibn Kathir EN+AR, Al-Tabari AR, As-Sa'di AR | Layer 3 (Human Commentary) |

## Project Structure (V3)

```
nur/
├── docs/
│   ├── v3/                    # V3 architecture (current)
│   │   ├── 00_OVERVIEW.md
│   │   ├── 01_DATA_SOURCES.md
│   │   ├── 02_CHUNK_SCHEMA.md
│   │   ├── 03_TAFSIR_STRATEGY.md
│   │   ├── 04_RETRIEVAL_PIPELINE.md
│   │   ├── 05_EMBEDDING_DESIGN.md
│   │   ├── 06_GENERATION_VERIFICATION.md
│   │   ├── 07_FAILURE_MODES.md
│   │   ├── 08_EXAMPLES.md
│   │   └── README.md
│   ├── PILLARS.md             # 10 theological pillars (unchanged)
│   ├── CONTEXT.md             # V1/V2 reference (V3 supersede notice)
│   ├── PHASES.md              # V1/V2 phases (V3 supersede notice)
│   └── brains.md              # Decision log (DEC-001 to DEC-042)
├── scripts/v3/                # V3 build pipeline (8 steps)
│   ├── 01_download_quran.py
│   ├── 02_download_hadith.py
│   ├── 03_download_tafsirs.py
│   ├── 04_generate_context_cards.py    # Lightning AI L40S
│   ├── 05_build_chunks.py
│   ├── 06_compute_cross_refs.py
│   ├── 07_embed_and_index.py           # Lightning AI / Kaggle T4×2
│   ├── 08_verify_pipeline.py
│   ├── run_all.sh
│   └── prepare_lightning_upload.sh
├── src/nur/
│   ├── v3/                    # V3 runtime pipeline
│   │   ├── architect.py       # Groq query decomposition
│   │   ├── retriever_quran.py # Phase A
│   │   ├── retriever_hadith.py# Phase B + authenticity weighting
│   │   ├── cross_refs.py      # Auto-pull hadiths from Quran chunks
│   │   ├── reranker.py        # bge-reranker-v2-m3
│   │   ├── reporter.py        # Groq structured JSON
│   │   ├── verifier.py        # NLI + Quran char-by-char
│   │   ├── pipeline.py        # Orchestrator
│   │   └── cli.py             # Typer CLI
│   ├── config.py              # Pydantic settings
│   ├── sources.py             # SourceRef + HADITH_COLLECTION_SLUGS
│   └── arabic.py              # Arabic normalization
├── data/                      # gitignored — built via scripts/v3/
│   ├── quran/                 # Raw Quran JSON
│   ├── hadith/meetif/         # Raw Hadith JSON (6 collections)
│   ├── tafsir/v3/             # Raw Tafsir JSON (4 editions × 114 surahs)
│   ├── processed/             # V3 chunks (quran_v3.jsonl, hadith_v3.jsonl, context_cards.jsonl)
│   ├── chroma_db_v3/          # V3 ChromaDB (quran_v3_dense, hadith_v3_dense)
│   └── sparse_v3/             # V3 sparse indices
└── tests/
```

## V3 Build Pipeline

| Step | Script | Where | GPU? | Time |
|------|--------|-------|------|------|
| 1 | `01_download_quran.py` | Local | No | 5 min |
| 2 | `02_download_hadith.py` | Local | No | 10 min |
| 3 | `03_download_tafsirs.py` | Local | No | 15 min |
| 4 | `04_generate_context_cards.py` | Lightning AI L40S | vLLM Qwen2.5-14B-AWQ | 30 min |
| 5 | `05_build_chunks.py` | Local | No | 1 min |
| 6 | `06_compute_cross_refs.py` | Local | No | 30 sec |
| 7 | `07_embed_and_index.py` | Lightning AI + Kaggle T4×2 | BGE-M3 | 25 min |
| 8 | `08_verify_pipeline.py` | Local | BGE-M3 (CPU OK) | 10 min |

See [`scripts/v3/LIGHTNING_AI.md`](scripts/v3/LIGHTNING_AI.md) for the Lightning AI workflow.

## Quick Start (V3 runtime)

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

# 5. Get the V3 indexed database
# Download nur_v3_indexed.zip from GitHub Releases (or build via scripts/v3/)
unzip nur_v3_indexed.zip

# 6. Verify
python scripts/v3/08_verify_pipeline.py

# 7. Run
export GROQ_API_KEY="your_key"
python -m nur.v3.cli "Est-ce que fumer est haram ?"
```

## License

MIT
