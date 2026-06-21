# NUR — Phase Roadmap

> Adapted from `ARCHITECTURE.md` Section 24. Each phase is independently shippable.

## Phase 1 — Data + Embedding 🚧 In Progress

**Goal**: Build the vector database with all Islamic texts.

- [x] Repo created (https://github.com/MahmoudDabachii2004/nur)
- [x] Arabic normalization module (`src/nur/arabic.py`)
- [x] Source ID protocol + clickable URLs (`src/nur/sources.py`)
- [x] Quran downloader (`scripts/01_download_quran.py`)
- [x] Hadith downloader (`scripts/02_download_hadith.py`)
- [x] Tafsir downloader (`scripts/03_download_tafsir.py`)
- [x] Normalize + chunk (`scripts/04_normalize_and_chunk.py`)
- [x] Verification script (`scripts/05_verify_phase1.py`)
- [x] Colab embedding script (`colab/embed_nur_colab.py`)
- [ ] **RUN** the pipeline locally + on Colab
- [ ] Verify 52K chunks embedded correctly

**Deliverable**: ChromaDB with 4 collections + 4 sparse JSON files, ready for retrieval.

---

## Phase 2 — RAG Pipeline + Structured Output

**Goal**: Working CLI chatbot that answers with cited sources.

Code structure:
```
src/nur/
├── pipeline.py        # NURPipeline orchestrator
├── retriever/
│   ├── dense.py       # ChromaDB queries (4 collections parallel)
│   ├── sparse.py      # Sparse JSON dot-product scoring
│   └── fusion.py      # RRF fusion (k=25, α=0.4 dense / 0.6 sparse)
├── generator.py       # Groq API + Instructor + JSON Schema
└── cli.py             # Rich + Typer interactive CLI
```

Key decisions:
- Groq Qwen3-32B at `temperature=0.0`, `frequency_penalty=0.0` (non-zero corrupts Arabic formulas)
- **In-Context Isolation** prompt — formal prohibition on using pretrained knowledge
- **XML format** for chunks in prompt (−30% hallucinations vs raw text)
- **Source IDs in UPPERCASE** (`SRC-QURAN-2-255`) — strong attention anchor
- **Chain-of-Thought Citation** via JSON Schema: `{thought_process, valid_source_ids, synthesis}`
- **JSON Schema (not Tool Calling)** via `response_format={"type": "json_schema", ...}` — native constrained decoding on Groq LPUs
- Language detection via `lingua-py` → force response language
- Retry via `tenacity` with exponential backoff + Ollama fallback
- **Latency target**: ~1.5 sec end-to-end (50ms BGE-M3 + 40ms ChromaDB + ~1s Groq)

**Deliverable**: Terminal chatbot that answers English/Arabic questions with clickable sources.

---

## Phase 3 — Reranking + Hybrid Search Fusion

- Implement Reciprocal Rank Fusion (RRF) — `k=25`, `α=0.4 dense / 0.6 sparse`
- Sparse dominates for Islamic text (exact references like "البخاري 1234" must match)
- Add `bge-reranker-v2-m3` cross-encoder reranker (~1.2GB RAM, ~200ms/50 docs on M4 MPS)
- Top-30 retrieval → reranker → top-10 → LLM
- **Divergence Detection**: if reranker returns opposing chunks with similar scores → flag `Divergence=True` (forces multi-perspective display, prerequisite for Ikhtilaf in Phase 7)
- **No ColBERT** — redundant with the reranker cross-encoder, <2% gain, too costly

**Deliverable**: Significantly improved retrieval quality + automatic ikhtilaf detection.

---

## Phase 4 — Context-Enriched Chunks (Anthropic)

- For each chunk, LLM-generate a contextual prefix (document type, theme, cross-refs)
- Re-embed the enriched chunks
- Anthropic technique: −67% retrieval failures
- Critical for Islamic text where verses depend on Asbab al-Nuzul (revelation context)
- ⚠️ The context LLM must NOT add dogmatic interpretation — purely factual prefix

**Deliverable**: Better semantic matching via rich context.

---

## Phase 5 — Authenticity Weighting + Mawdu' Detection

- Implement weighted scoring by grade (Sahih +30%, Hasan +10%, Da'if −50%, Mawdu weight=0)
- **Negative database**: index known fabricated hadiths (ADAM-HA dataset — to be verified)
- **Double-RAG strategy**: authentic base + fabricated base → automatic warning
- **Detection threshold**: cosine < 0.45 → "This text does not appear in authentic reference corpora"
- Mandatory grade display in every hadith citation
- Warning for hasan/da'if hadiths

**Deliverable**: Ethically responsible answers, weak hadiths flagged, fabricated hadiths detected.

---

## Phase 6 — Post-Generation Verification

Multi-layer anti-hallucination:

- **Module T4 — NLI Verification**: `DeBERTa-v3-large` or `Ayn-NLI` verifies each sentence (entailment < 0.95 = reject)
- **Module T5 — Bi-level Character Matching** for Arabic:
  - Level 1: Smith-Waterman / Levenshtein after diacritics strip + normalization
  - Level 2: Strict comparison with Uthmani Quran — 100% match on base letters
- **Module T6 — Decoupled Grounding**: `exact_citations` vs `synthesis` separated, parser rejects if citation isn't word-for-word
- **Constrained Decoding** (Outlines / Instructor): force LLM to produce only tokens present in retrieved chunks for Arabic citation fields

**Deliverable**: Zero hallucination on Quranic verses, sources verified by 3 independent layers.

---

## Phase 7 — Scholar Opinions + Ikhtilaf

- Build scholar index: collect fatwas from IslamQA, Islamweb, Dar al-Ifta
- Structure: scholar name, madhhab, opinion, evidence (Quran+Hadith), book source
- Detect ikhtilaf and present all views with 5 consensus levels (see PILLARS.md)
- **Absolute neutrality**: never encourage, never discourage
- Mandatory disclaimer: "For specific application, consult a qualified scholar of your school"

**Deliverable**: Answers with scholarly opinions, ikhtilaf correctly presented.

---

## Phase 8 — Cross-References Quran ↔ Hadith

- Use Tafsir Ibn Kathir as a bridge: for each ayah, extract hadiths referenced
- Build cross-reference DB: ayah → hadiths, hadith → ayahs
- In answers, when citing a verse → suggest explanatory hadiths, and vice versa

**Deliverable**: Rich answers with Quran ↔ Hadith ↔ Tafsir connections.

---

## Phase 9 — Next.js PWA Frontend

- Next.js PWA — installable on mobile without App Store
- Rich display: Arabic text (large, RTL) + English translation + grade + clickable link
- Language toggle (AR/EN)
- Source accordion
- Light/dark mode
- Free Vercel deployment

**Deliverable**: Web app usable in browser + installable on mobile.

---

## Phase 10 — Advanced RAG + CRAG

- **Corrective RAG**: retrieval evaluator grades documents before generation
- **Query decomposition**: complex questions → sub-questions
- **Adaptive routing**: classify question (factual, fiqh, aqeedah) and adapt pipeline
- **Fallback**: Groq → OpenRouter → Ollama local

**Deliverable**: Robust, intelligent RAG pipeline.

---

## Phase 11 — Evaluation + Benchmarking

- Build evaluation dataset: 50-100 questions with expected answers and sources
- Implement RAGAS (faithfulness, answer relevancy, context precision/recall)
- Islamic metrics:
  - Source Attribution Accuracy
  - Madhhab Consistency
  - Ikhtilaf Awareness
  - Quranic Accuracy (verbatim citation)
  - Hadith Grading Accuracy
  - Abstention Rate (correctly refuses out-of-scope)
- Compare phases: baseline (Phase 2) vs each improvement

**Deliverable**: Objective quality metrics, weak points identified.

---

## Phase 12 — Advanced Features

- Voice input via Groq Whisper Large v3
- TTS for summaries (Orpheus Arabic Saudi for interface only — NOT for Quran, no tajweed support)
- Prompt Guard 2-86M against prompt injection
- Quran ontology / Graph RAG (if Phase 8 justifies the need)
- Full offline mode via Ollama

**Deliverable**: Complete, robust application.
