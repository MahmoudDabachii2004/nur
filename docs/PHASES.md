# NUR — Phase Roadmap

> Each phase is independently shippable and validated before moving to the next. 
> **Engineering Rule**: Fail small, validate between steps, no lazy stubbing. 
> All code artifacts must be written in English, with clear doc-comments explaining *why*, not *what*.

## Phase 1 — Data Ingestion + Contextual Embedding ✅ Completed

**Goal**: Build the local vector database with LLM-enriched context for all Islamic texts.

- [x] Repo created (https://github.com/MahmoudDabachii2004/nur)
- [x] Arabic normalization module (`src/nur/arabic.py`)
- [x] Source ID protocol + clickable URLs (`src/nur/sources.py`)
- [x] Raw data download scripts (Quran, Hadith, Tafsir)
- [x] Normalize + chunk with Layer 1 Structural Context (`scripts/04_normalize_and_chunk.py` → `DEC-001`)
- [x] Cloud Layer 2 LLM Contextual Synthesis (`nur_synthesizer_cloud.py`)
  - Upgraded to `Qwen2.5-14B-Instruct-AWQ` for superior theological extraction.
  - Executed on Lightning AI (L40S GPU) with native FlashAttention-2.
  - Pre-tokenization with hard truncation (8192 tokens) to prevent validation crashes.
- [x] **RUN** the Cloud pipeline to generate `nur_indexed_data_contextual.zip`
- [x] Unzip and verify 52K chunks embedded correctly in local ChromaDB + Sparse JSONs.

**Validation Results**: 
Dense search benchmark passed. The Qwen 14B contextual prefix successfully resolved the French sparse/dense match failure (`DEC-004`). French queries (e.g., "L'importance de la recherche de la science") now return highly relevant Hadiths with positive similarity scores, whereas the previous baseline returned negative scores and irrelevant results.

**Deliverable**: Local ChromaDB (4 collections) + 4 sparse JSON files, enriched with dual-layer bilingual context, ready for retrieval.

---

## Phase 2 — Local RAG Pipeline + Structured Output (The "Smart Archivist") 🚧 In Progress

**Goal**: Working CLI chatbot that answers with cited sources, using the local database and a 2-step Groq API pipeline.

**Code structure to build**:
```
src/nur/
├── pipeline.py        # NURPipeline orchestrator ✅ DEC-025
├── retriever/
│   ├── dense.py       # ChromaDB queries (4 collections parallel) ✅ DEC-015
│   ├── sparse.py      # Sparse JSON dot-product scoring ✅ DEC-018
│   └── fusion.py      # RRF fusion (k=25, α=0.4 dense / 0.6 sparse) ✅ DEC-021
├── generator/         # Groq API (Architect + Reporter) + Instructor ✅ DEC-022
│   └── __init__.py    # Architect, Reporter, Generator classes + Pydantic schemas
├── grades.py          # Hadith grade education + two-tier warning ✅ DEC-027
└── cli.py             # Rich + Typer interactive CLI ✅ DEC-028
```

**Progress checklist**:
- [x] `dense.py` — DenseRetriever connected to ChromaDB. Tested via `scripts/benchmark_dense.py` with 6 queries (EN/FR/AR) across quran + hadith. All return on-topic results with healthy similarity scores (0.65+).
- [x] `sparse.py` — SparseRetriever with inverted index. Tested via `scripts/test_sparse_search.py` (model-free, 3 self-similarity cases pass) + 6 stress tests (empty query, non-existent token, top_k edges, determinism, dot-product math). All pass.
- [x] `fusion.py` — RRFFuser implementing Reciprocal Rank Fusion. Tested via `scripts/test_fusion.py` (pure-math, 9 tests pass: hand-computed scores, both-lists boost, single-list cases, top_k truncation, empty inputs, determinism, config defaults, invalid param validation). All pass.
- [x] `generator/` — Architect (Step 1: query decomposition, `Mode.TOOLS`) + Reporter (Step 4: structured JSON report, `Mode.JSON_SCHEMA`) + Generator facade. Both tests PASSED on user's Mac (DEC-022 + DEC-024): Architect returns 4 well-decomposed sub-questions, Reporter returns valid structured report with all source IDs cited, Arabic preserved verbatim, "Strict Archivist" persona working.
- [x] `pipeline.py` — NURPipeline orchestrator wiring Architect → multi-query hybrid retrieval (4 sources × N queries) → RRF fusion → dedup → Reporter. Lazy-loads BGE-M3 on first query. Handles all 4 source metadata schemas (quran, hadith, tafsir_ar, tafsir_en). Returns `PipelineResult` with all intermediate steps for transparency. **Live integration test PASSED on user's Mac** (DEC-026): 12.2s total, 4 sub-questions generated, 30 chunks retrieved, 4 direct reports with valid source IDs, synthesis cites all 4 IDs. Retrieved exactly the right verses for "charity and zakat" (9:60 the 8 categories, 2:273 the restricted poor, 70:25 the deprived, 19:31 prayer + zakah). **Grade enrichment added** (DEC-027): pipeline now attaches hadith grade explanations + computes answer-level warnings (two-tier: CRITICAL for Mawdu, BIG for only-Da'if, SMALL for Da'if-alongside-Sahih/Hasan).
- [x] `cli.py` — Rich + Typer interactive terminal chatbot. Two modes: single-query (`python -m nur "question"`) and interactive REPL (`python -m nur`). Displays: answer warning FIRST (Pillar 4), conflict detection (Pillar 9), synthesis, direct reports with Arabic (large, RTL — Pillar 10), grade explanations (Pillar 3), clickable URLs (Pillar 7). Flags: `--lang fr`, `--force-reasoning`, `--no-arabic`, `--verbose`. Entry point added to pyproject.toml (`nur = "nur.cli:app"`). **Live test deferred to user** — needs BGE-M3 + ChromaDB + Groq.
- [ ] End-to-end validation: real user question → cited, structured answer. **This is the final Phase 2 deliverable.**

**Key Decisions (The Smart Archivist Pipeline)**:
- **Task 1 (Architect)**: `llama-3.1-8b-instant` (Groq). Takes raw query, outputs JSON array of 1 to N sub-questions in FR/EN.
- **Task 2 (Reporter)**: `meta-llama/llama-4-scout-17b-16e-instruct` (Groq). Takes Top-10 chunks + raw query. Outputs strict JSON schema: `{conflict_detection, direct_reports, synthesis}`.
- **In-Context Isolation**: LLM is forced to act as an Archivist. No use of pretrained knowledge.
- **Language Strategy**: LLM analyzes FR/EN context to resolve dilemmas. Copies Arabic text exactly into `arabic_text` field.
- **Retry/Fallback**: If Groq returns 429 (Rate Limit), silently reroute Task 2 to local Ollama (`llama-3.1-8b` on RX 5700 XT).

**Deliverable**: Terminal chatbot that answers complex questions with structured, grounded reports.

---

## Phase 3 — Mathematical Reranking + Abstention Gate

**Goal**: Eliminate hallucination by ensuring the LLM only receives high-quality context.

- [ ] Add `bge-reranker-v2-m3` cross-encoder reranker locally on M4 (MPS).
- [ ] Pipeline: Fetch 30 chunks (from multi-query) -> Reranker scores all 30 -> Keep Top 10.
- [ ] **Abstention Rule**: If Top 1 chunk score < 0.35, abort LLM generation. Return "Insufficient sources".
- [ ] Implement Authenticity Weighting (Sahih +30%, Hasan +10%, Da'if -50%, Mawdu weight=0).
- [ ] **Mawdu' Detection**: Parallel check against a negative index of fabricated hadiths.

**Deliverable**: Highly relevant retrieval, ethically weighted by authenticity, with mathematical abstention.

---

## Phase 4 — Post-Generation Verification (Zero Tolerance)

**Goal**: Mathematically verify the LLM's output before showing it to the user.

- [ ] **Module T4 — NLI Verification**: `DeBERTa-v3-large` verifies each sentence (entailment < 0.95 = reject).
- [ ] **Module T5 — Bi-level Character Matching** for Arabic:
  - Level 1: Smith-Waterman / Levenshtein after diacritics strip + normalization.
  - Level 2: Strict comparison with Uthmani Quran — 100% match on base letters. If LLM output mismatches, replace with local authentic text.
- [ ] **Module T6 — Decoupled Grounding**: Parser rejects if `exact_citations` doesn't match `direct_reports`.

**Deliverable**: Mathematically verified responses, zero tolerance for Quranic alteration.

---

## Phase 5 — Scholar Opinions + Ikhtilaf

**Goal**: Integrate scholarly consensus and disagreement ethically.

- [ ] Build scholar index: collect fatwas from IslamQA, Islamweb, Dar al-Ifta.
- [ ] Structure metadata: scholar name, madhhab, opinion, evidence, book source.
- [ ] Detect Ikhtilaf and present all views with 5 consensus levels (Ijma' to Isolated Opinion).
- [ ] Absolute neutrality enforced in system prompt.
- [ ] Mandatory disclaimer: "For specific application, consult a qualified scholar of your school."

**Deliverable**: Answers with scholarly opinions, Ikhtilaf correctly presented and neutral.

---

## Phase 6 — Cross-References Quran ↔ Hadith

**Goal**: Rich answers showing interconnections between scriptures.

- [ ] Use Tafsir Ibn Kathir as a bridge: for each ayah, extract referenced hadiths.
- [ ] Build cross-reference DB: ayah -> hadiths, hadith -> ayahs.
- [ ] In answers, when citing a verse -> suggest explanatory hadiths, and vice versa.

**Deliverable**: Deeply interconnected scriptural answers.

---

## Phase 7 — Next.js PWA Frontend

**Goal**: Accessible, installable web application.

- [ ] Next.js PWA — installable on mobile without App Store.
- [ ] Rich display: Arabic text (large, RTL) *always visible* + translation + grade + clickable link.
- [ ] Language toggle (FR/EN) for the UI and LLM synthesis language.
- [ ] Source accordion and dark/light mode.
- [ ] Free Vercel deployment.

**Deliverable**: Web app usable in browser + installable on mobile.

---

## Phase 8 — Evaluation + Benchmarking

**Goal**: Objective quality metrics.

- [ ] Build evaluation dataset: 50-100 questions with expected answers and sources.
- [ ] Implement RAGAS (faithfulness, answer relevancy, context precision/recall).
- [ ] Islamic metrics: Source Attribution Accuracy, Madhhab Consistency, Ikhtilaf Awareness, Quranic Accuracy, Hadith Grading Accuracy, Abstention Rate.

**Deliverable**: Objective quality metrics, weak points identified.

---

## Phase 9 — Advanced RAG + CRAG (Future)

- [ ] Corrective RAG: retrieval evaluator grades documents before generation.
- [ ] Adaptive routing: classify question (factual, fiqh, aqeedah) and adapt pipeline.
- [ ] Voice input/output (Groq Whisper / Orpheus TTS for interface only, not Quran).
- [ ] Full offline mode via Ollama.

**Deliverable**: Complete, robust, production-ready application.
