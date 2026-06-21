# NUR — Setup Guide (Mac M4)

Complete setup instructions for macOS on Apple Silicon (M1/M2/M3/M4). Tested on M4 16GB.

## Prerequisites

### 1. Python 3.10+

macOS comes with Python 3.9 — you need 3.10+ for modern type hints.

```bash
# Install via Homebrew (recommended)
brew install python@3.11

# Verify
python3.11 --version
# Python 3.11.x
```

### 2. Git

macOS comes with Git. Verify:

```bash
git --version
# git version 2.47+
```

### 3. Ollama (for local LLM fallback)

Download from https://ollama.com/download/mac

```bash
# After install, pull the local fallback model (~4.4GB download)
ollama pull qwen2.5:7b

# Verify it runs
ollama run qwen2.5:7b "Say hello in Arabic"
```

### 4. Groq API key (free)

1. Go to https://console.groq.com
2. Sign in with Google or GitHub
3. API Keys → Create new key
4. Copy the key (starts with `gsk_...`)

The free tier gives you:
- 60 requests/minute on Qwen3-32B
- 1,000 requests/day
- 6,000 tokens/minute

That's plenty for development and personal use.

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/MahmoudDabachii2004/nur.git
cd nur
```

### 2. Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Verify
which python
# /Users/.../nur/.venv/bin/python
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- FlagEmbedding + torch (for BGE-M3, used during Colab embedding)
- ChromaDB (vector DB)
- Groq SDK, Ollama client, OpenAI (for OpenRouter fallback)
- LangChain + LangGraph (RAG orchestration)
- FastAPI + Uvicorn (backend, Phase 2+)
- Rich + Typer (CLI)
- pyarabic (Arabic text processing)

> **Note**: `torch` is large (~700MB). On M4, it uses the MPS backend for Apple Silicon GPU.

### 4. Set up environment

```bash
cp .env.example .env

# Edit .env with your editor of choice
nano .env
# Set: GROQ_API_KEY=gsk_your_key_here
```

### 5. Verify installation

```bash
# Test the Arabic normalization module
python -c "
from nur.arabic import normalize_arabic
print(normalize_arabic('أَوْلَٰئِكَ عَلَيْهِمْ صَلَوَاتٌ'))
# Should print: اولئك عليهم صلوات
"

# Test source ID protocol
python -c "
from nur.sources import SourceRef
ref = SourceRef(kind='quran', surah=2, ayah=255, text_ar='...', text_en='...')
print(ref.source_id)  # SRC-QURAN-2-255
print(ref.url)        # https://quran.com/2/255
"
```

## Phase 1 — Data Pipeline

### Step 1: Download Quran (~30 sec)

```bash
python scripts/01_download_quran.py
```

Downloads:
- `data/quran/quran-uthmani.json` (Arabic — source of truth)
- `data/quran/en.sahih.json` (English Saheeh International)
- `data/quran/quran-meta.json` (surah metadata)

Expected: 114 surahs, 6,236 ayahs.

### Step 2: Download Hadith (~3 min)

```bash
python scripts/02_download_hadith.py
```

Downloads the 6 canonical collections (Kutub al-Sittah):
- Sahih al-Bukhari (~7,008 hadiths)
- Sahih Muslim (~5,362 hadiths)
- Sunan Abi Dawud (~4,590 hadiths)
- Jami` at-Tirmidhi (~3,956 hadiths)
- Sunan an-Nasa'i (~5,662 hadiths)
- Sunan Ibn Majah (~4,341 hadiths)

Total: ~33,738 hadiths with Arabic + English + grades.

### Step 3: Download Tafsir Ibn Kathir (~5 min)

```bash
python scripts/03_download_tafsir.py
```

Downloads 228 files (114 surahs × 2 languages) from spa5k/tafsir_api via jsDelivr CDN.

### Step 4: Normalize + Chunk (~1 min)

```bash
python scripts/04_normalize_and_chunk.py
```

Produces:
- `data/processed/quran.jsonl` (~6,236 chunks)
- `data/processed/hadith.jsonl` (~33,738 chunks)
- `data/processed/tafsir_ar.jsonl`
- `data/processed/tafsir_en.jsonl`

Each chunk is one JSON line with:
- Original Arabic + normalized Arabic (diacritics stripped, alef variants normalized)
- Original English + normalized English
- Source ID (e.g. `SRC-QURAN-2-255`)
- Grade + weight (for hadith)
- Clickable URL

### Step 5: Verify

```bash
python scripts/05_verify_phase1.py
```

### Step 6: Embed on Google Colab T4 (~45 min)

See [`colab/README.md`](../colab/README.md) for detailed instructions.

After embedding, you'll have:
- `data/chroma_db/` — ChromaDB persistent store (4 collections)
- `data/sparse/quran_sparse.json` etc. — BGE-M3 sparse vectors

## Memory Budget on M4 16GB

| Component | Memory | Notes |
|-----------|--------|-------|
| macOS | ~3-4 GB | Non-negotiable |
| Ollama (Qwen2.5-7B Q4) | ~4-5 GB | Only when in use |
| ChromaDB | ~0.5-1 GB | Index + cache |
| FastAPI + Python | ~0.5-1 GB | Runtime overhead |
| **Free for browser + other apps** | ~5-7 GB | Comfortable |

The embeddings model (BGE-M3) is loaded on-demand for re-embedding only — not during normal chatbot use.

## Tips

### Speed up re-embedding on Mac (no Colab)

If you want to iterate quickly and don't mind waiting ~3 hours:

```bash
# Use MPS (Metal Performance Shaders) for Apple Silicon GPU acceleration
PYTORCH_ENABLE_MPS_FALLBACK=1 python -c "
import torch
print(torch.backends.mps.is_available())  # Should print True
"
```

### Keep Ollama model in memory

By default, Ollama unloads models after 5 min of inactivity. For development:

```bash
# Set keep-alive to 30 minutes
export OLLAMA_KEEP_ALIVE=30m
ollama serve
```

### Use rich logging

The CLI uses `rich` for colored output. For verbose debug logs:

```bash
export NUR_LOG_LEVEL=DEBUG
```

## Next Steps

Once Phase 1 is complete:

- **Phase 2**: Build the RAG pipeline (retrieval + Groq generation + source IDs) — see `docs/PHASES.md`
- **Phase 3**: Hybrid search with RRF fusion + bge-reranker-v2-m3
- **Phase 4**: Context-enriched chunks (Anthropic technique)

See `docs/PHASES.md` for the full 12-phase roadmap.
