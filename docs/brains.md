# NUR (نور) — Brains & Change Log Database

This document serves as the project's living memory. It chronicles every architectural decision, major refactoring, model switch, and implementation milestone. Each entry is timestamped and logs the "Before" state, the "After" state, the "Why" (engineering rationale), and validation metrics.

---

## Change Log

### [2026-06-21T12:30:00-04:00] — Implementing Multilingual Contextual Retrieval
* **Decision ID:** `DEC-001`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
Our initial hybrid search (RRF) audit showed that queries in French and English returned `N/A` for their sparse (lexical keyword) ranks. This occurred because the indexed text was purely Arabic, preventing exact keyword match for non-Arabic queries. We needed a way to make the lexical matching engine function across French, English, and Arabic without losing the Arabic text as the primary source of truth.

#### 2. Before vs. After
* **Before:**
  * Chunks contained only raw Arabic text.
  * French/English queries relied 100% on the dense vector model's translation capacity, losing half of the hybrid search benefit (no sparse rank contribution).
  * No metadata (like Surah names, Hadith narrators, or grading) was embedded inside the chunk text, making searches for "Hadith narrated by Abu Hurairah" highly inaccurate.
* **After:**
  * Created a contextual template inside `scripts/03_normalize_and_chunk.py`:
    * *Quran:* Chunks are prefixed with Surah name (AR/EN), Revelation type, English & French translations, and a 200-character snippet of Tafsir Ibn Kathir.
    * *Hadith:* Chunks are prefixed with narrator, chapter titles (AR/EN), grade, and 300 characters of English translation.
  * Lexical index (sparse) now contains English and French terms, enabling full hybrid keyword search in all three languages.

#### 3. Impacted Files
* [03_normalize_and_chunk.py](file:///Users/mahmoud/Documents/nur/nur/scripts/03_normalize_and_chunk.py) — Updated chunking script to prepend structural templates.

#### 4. Validation
* Regenerated 6,236 Quran chunks, 33,738 Hadith chunks, and 12,472 Tafsir chunks in `data/processed/`.

---

### [2026-06-21T12:45:00-04:00] — Optimizing Indexer for Apple Silicon (MPS) & Fixing ChromaDB Sparse Storage
* **Decision ID:** `DEC-002`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
Passing sparse vectors directly to local ChromaDB's `Collection.add()` raised a `TypeError` because ChromaDB does not natively support sparse vector storage. Additionally, indexing on macOS Apple Silicon (MPS) was extremely slow due to PyTorch compilation overhead.

#### 2. Before vs. After
* **Before:**
  * The ingestion script crashed trying to push sparse vectors to ChromaDB.
  * High latency (~3-5 chunks/sec) on macOS MPS backend due to dynamic tensor shapes recompiling on every batch.
* **After:**
  * Modified the indexing script `05_fix_sparse.py`:
    * Separated storage: Dense vectors go into ChromaDB `{source}_dense` collections; Sparse vectors are saved as JSON weight dictionaries under `data/sparse/{source}_sparse.json`.
    * Implemented static batch size (`batch_size=16` inside PyTorch) to prevent MPS dynamic graph shape compilation overhead, boosting speed to ~9.5 chunks/sec.
    * Silenced third-party libraries' `tqdm` progress bars to keep terminal outputs clean.

#### 3. Impacted Files
* [05_fix_sparse.py](file:///Users/mahmoud/Documents/nur/nur/scripts/05_fix_sparse.py) — Rewritten to decouple dense/sparse storage and optimize for MPS.

#### 4. Validation
* Indexing task verified to run without crashes on MPS, outputting correct dense embeddings to ChromaDB and sparse dictionaries to JSON files.

---

### [2026-06-21T13:10:00-04:00] — Deploying GPU-Accelerated Google Colab Ingestion
* **Decision ID:** `DEC-003`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
Even with MPS optimizations, indexing the 52k chunks on a MacBook M4 CPU/GPU takes ~1.5 to 2 hours. We needed server-grade GPU hardware to accelerate the ingestion process.

#### 2. Before vs. After
* **Before:**
  * Indexing run bound to local MacBook, taking ~1.5 hours at ~9 chunks/sec.
* **After:**
  * Created `scripts/colab_indexer.py`, a self-contained Google Colab script.
  * Automates installation of dependencies, presents browser upload buttons for chunk files, leverages a T4 GPU (CUDA) with a larger batch size (`batch_size=128`), and zips/downloads the final database folder (`nur_indexed_data.zip`).

#### 3. Impacted Files
* [colab_indexer.py](file:///Users/mahmoud/Documents/nur/nur/scripts/colab_indexer.py) — Created self-contained Colab helper script.

#### 4. Validation
* Executed on Google Colab (T4 GPU), increasing indexing speed from ~9 chunks/sec to **~23 chunks/sec**, cutting total execution time to less than 20 minutes.

---

### [2026-06-21T14:22:00-04:00] — First Retrieval Audit (Baseline Benchmark)
* **Decision ID:** `DEC-004`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
With the database unzipped and verified, we needed to establish our baseline search pertinence scores. We ran `python3 scratch/test_retrieval_audit.py` to evaluate 13 test queries (Arabic, English, French) across Quran and Hadiths using Hybrid Search (RRF) + BGE-Reranker-v2-M3.

#### 2. Before vs. After
* **Before:**
  * No structured evaluation report existed to compare different chunk contexts.
* **After:**
  * Generated [retrieval_audit_results.md](file:///Users/mahmoud/Documents/nur/nur/scratch/retrieval_audit_results.md) detailing the ranks and scores for all 13 queries.
  * *Arabic Queries:* Perform exceptionally well. Sparse index matches perfectly, and Reranker scores are highly positive (e.g. `2.87` for patience hadiths).
  * *English Queries:* Perform well on both Quran and Hadiths (Sparse ranks active because chunk text contains English translations).
  * *French Queries on Hadiths:* The weak link. Sparse rank is systematically `N/A` (except for transliterated words like "Riba"), and Reranker scores are highly negative (e.g. `-8.64` for seeking knowledge). This occurs because Hadith chunks contain no French translation.

#### 3. Impacted Files
* [retrieval_audit_results.md](file:///Users/mahmoud/Documents/nur/nur/scratch/retrieval_audit_results.md) — Created search evaluation report.

#### 4. Validation
* Verified that the hybrid retrieval and local reranker run successfully on macOS MPS backend.

---

### [2026-06-21T14:33:00-04:00] — Dynamically Rescaling LLM Context Summary Constraints for Kaggle Ingestion
* **Decision ID:** `DEC-005`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
Our initial design clamped the LLM-generated summaries to a strict 45-word constraint to prevent chunk truncation within BGE-M3's 512-token context window. However, this hard limit risked constraining context synthesis for long/complex Hadiths and Quranic verses (which contain multiple jurisprudential topics). We needed an adaptive approach to capture multi-topic detail without causing index truncation. Additionally, we fixed a critical bug in the vLLM stop tokens configuration where a newline character `\n` incorrectly aborted bilingual generation.

#### 2. Before vs. After
* **Before:**
  * LLM-generated summaries were restricted to under 45 words, risking information loss in long/rich text chunks.
  * `vllm.SamplingParams` included `\n` in its `stop` list, causing the generator to abort after the first line (generating only the French block and completely skipping the English block).
  * `max_tokens` was capped at 60.
* **After:**
  * Expanded summary limits to 80 words dynamically scale for long texts, instructing the model to remain brief for short texts but fully capture rulings for long texts.
  * Removed `\n` from the stop tokens list in `vllm.SamplingParams` to allow generating the full multi-line bilingual text (`[FR]` and `[EN]`).
  * Increased `max_tokens` to 120.

#### 3. Impacted Files
* [kaggle_context_synthesizer.py](file:///Users/mahmoud/Documents/nur/nur/scripts/kaggle_context_synthesizer.py) — Modified system prompt constraints and vLLM sampling parameters.
* [architecture_audit_report.md](file:///Users/mahmoud/.gemini/antigravity-ide/brain/7809d477-b30a-47bf-a633-60bacc549d28/architecture_audit_report.md) — Documented architectural analysis and safeguards.

#### 4. Validation
* Inspected generated prompts and parameters locally. The updated script is prepared for execution on Kaggle.

---

### [2026-06-21T15:00:00-04:00] — Switching to Qwen2.5-14B-Instruct-AWQ (Quantized) to Resolve CUDA OOM on Kaggle
* **Decision ID:** `DEC-006`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
When running the baseline FP16 model `Qwen/Qwen2.5-14B-Instruct` on Kaggle's dual Tesla T4 GPUs (15GB VRAM each), the model weights alone consumed ~14.0 GB of VRAM per GPU. This left only ~0.56 GB of free VRAM, causing a CUDA Out Of Memory (OOM) error during vLLM's internal autotuning and memory allocation phase. We needed to transition to a quantized model to reduce weights overhead while retaining 14B logical quality.

#### 2. Before vs. After
* **Before:**
  * Used `Qwen/Qwen2.5-14B-Instruct` (FP16), consuming 14.0 GB of VRAM per GPU (total weight size ~28GB).
  * vLLM initialization failed with a CUDA Out of Memory error during the autotuning process.
* **After:**
  * Switched to `Qwen/Qwen2.5-14B-Instruct-AWQ` (4-bit quantized), reducing model weight size to ~8GB total (~4GB per GPU).
  * This leaves ~11GB of free VRAM per GPU for KV cache, workspace activations, and the subsequent BGE-M3 execution.

#### 3. Impacted Files
* [kaggle_context_synthesizer.py](file:///Users/mahmoud/Documents/nur/nur/scripts/kaggle_context_synthesizer.py) — Updated model name to AWQ version.

#### 4. Validation
* Switched the model in the synthesizer script and prepared for execution on Kaggle.

---

### [2026-06-21T15:09:00-04:00] — Disabling FlashInfer to Bypass Compiler Linker Errors on Kaggle
* **Decision ID:** `DEC-007`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
During vLLM initialization of the AWQ model, the `FlashInfer` sampling backend attempted to perform runtime JIT compilation using `ninja`. However, the compilation failed with `/usr/bin/ld: cannot find -lcuda` because the Kaggle Docker container does not map the CUDA driver shared library (`libcuda.so`) inside the standard linker library path. We needed to bypass FlashInfer to prevent compilation failures.

#### 2. Before vs. After
* **Before:**
  * vLLM used the `FlashInfer` sampler and backend, which failed at compile time with a `CalledProcessError` on `ninja`.
* **After:**
  * Configured `VLLM_USE_FLASHINFER_SAMPLER="0"` and `VLLM_DISABLE_FLASHINFER="1"` environment variables at the top of the script.
  * Forces vLLM to fall back to its pre-compiled **Triton/SDPA** attention backend and sampler, which runs out-of-the-box on Kaggle T4 GPUs without compilation.

#### 3. Impacted Files
* [kaggle_context_synthesizer.py](file:///Users/mahmoud/Documents/nur/nur/scripts/kaggle_context_synthesizer.py) — Set environment variables to disable FlashInfer.

#### 4. Validation
* Prepared the updated script for copy-pasting into Kaggle.

---

### [2026-06-21T15:13:00-04:00] — Switching to Single GPU (TP=1) to Bypass Kaggle IPC Shared Memory Limits
* **Decision ID:** `DEC-008`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
When running vLLM with `tensor_parallel_size=2`, the engine spawned multiple processes that communicate using shared memory and NCCL. However, containerized environments like Kaggle restrict `/dev/shm` (shared memory IPC blocks) by default, causing worker communication to freeze with the error `No available shared memory broadcast block found in 60 seconds`. Since the 4-bit AWQ model weighs only ~8.3 GB, it fits easily within a single 15GB T4 GPU's VRAM. We needed to run the model on a single GPU to bypass all multi-processing communication bottlenecks.

#### 2. Before vs. After
* **Before:**
  * Used `tensor_parallel_size=2` (distributed over two T4 GPUs), causing vLLM's multi-process orchestrator to hang on shared memory broadcast timeouts.
* **After:**
  * Configured `tensor_parallel_size=1` to run exclusively on one GPU.
  * Completely eliminates the need for NCCL, multiprocessing, and shared memory IPC blocks, ensuring instant, single-process initialization.
  * Set `gpu_memory_utilization=0.80` for safe single-GPU VRAM usage.

#### 3. Impacted Files
* [kaggle_context_synthesizer.py](file:///Users/mahmoud/Documents/nur/nur/scripts/kaggle_context_synthesizer.py) — Switched tensor_parallel_size to 1.

#### 4. Validation
* Switched parameter and updated script ready for execution on Kaggle.

---

### [2026-06-21T15:18:00-04:00] — Enforcing Triton Attention Backend to Prevent FlashInfer Prefill Compiler Fails
* **Decision ID:** `DEC-009`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
Although we disabled the FlashInfer sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`), the vLLM engine still fell back to the `FLASHINFER` attention backend for prefill computations. This triggered another JIT compilation of CUDA kernels for batch prefill (`batch_prefill_with_kv_cache...`), causing compilation to crash with the same linker error (missing `libcuda.so`). We needed to explicitly force vLLM to use the pre-compiled Triton attention backend to completely bypass FlashInfer compilation steps.

#### 2. Before vs. After
* **Before:**
  * vLLM compiled prefill kernels with FlashInfer/Ninja at runtime, resulting in a linker crash on Kaggle.
* **After:**
  * Configured `VLLM_ATTENTION_BACKEND="TRITON_ATTN"` in the script.
  * Bypasses the FlashInfer compilation path entirely, directing all attention and prefill computations to use pre-compiled Triton kernels.

#### 3. Impacted Files
* [kaggle_context_synthesizer.py](file:///Users/mahmoud/Documents/nur/nur/scripts/kaggle_context_synthesizer.py) — Set VLLM_ATTENTION_BACKEND to TRITON_ATTN.

#### 4. Validation
* Prepared the updated script for copy-pasting to Kaggle.

---

### [2026-06-21T15:31:00-04:00] — Force-Uninstalling FlashInfer to Bypass Kernel JIT Compiler Failure
* **Decision ID:** `DEC-010`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
Even after setting `VLLM_ATTENTION_BACKEND="TRITON_ATTN"`, the version of vLLM in the environment still attempted to initialize the `FLASHINFER` attention backend for prefill operations, failing on compilation because of Kaggle's missing `libcuda.so`. This happened because vLLM's internal backend selector checks for the *presence* of the `flashinfer` package and overrides the backend selection. To force an absolute fallback, we decided to uninstall the `flashinfer-python` and `flashinfer-cubin` packages from the environment immediately after dependency installation.

#### 2. Before vs. After
* **Before:**
  * The `flashinfer` packages were present in the environment, causing vLLM to automatically import and try to compile JIT prefill kernels, leading to a Ninja crash.
* **After:**
  * Added `pip uninstall -y flashinfer-python flashinfer-cubin` to the script's initialization stage.
  * Without the package files present, vLLM's soft-import checks fail, forcing it to fall back to the pre-compiled **Triton/SDPA** backend with zero JIT compilation steps.

#### 3. Impacted Files
* [kaggle_context_synthesizer.py](file:///Users/mahmoud/Documents/nur/nur/scripts/kaggle_context_synthesizer.py) — Added pip uninstall command.

#### 4. Validation
* Prepared the updated script for copy-pasting to Kaggle.

### [2026-06-21T15:38:00-04:00] — Overriding Tokenizer model_max_length to Prevent 512-Token Context Limit Validation Fails
* **Decision ID:** `DEC-011`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
During contextual generation on Kaggle, the vLLM engine crashed with a `VLLMValidationError` asserting that the model's maximum context length is 512 tokens when processing prompts exceeding 512 tokens (such as Al-Baqarah 282). This happens because Hugging Face tokenizers in certain environments default `model_max_length` to 512, which overrides the engine's `max_model_len` configuration inside vLLM's request validation (`_token_len_check`).

#### 2. Before vs. After
* **Before:**
  * Tokenizer's `model_max_length` defaulted to 512, rejecting any input prompt longer than 512 tokens (such as long Quran verses and translations combined).
* **After:**
  * Explicitly retrieve the tokenizer object via `llm.get_tokenizer()` and override its `model_max_length` property to 2048 to align with the engine's `max_model_len` limits.

#### 3. Impacted Files
* [kaggle_context_synthesizer.py](file:///Users/mahmoud/Documents/nur/nur/scripts/kaggle_context_synthesizer.py) — Added tokenizer model_max_length override logic.

#### 4. Validation
* Updated the script and prepared it for paste-and-run on Kaggle.

### [2026-06-21T15:52:00-04:00] — Increasing gpu_memory_utilization to 0.90 to Prevent Cache Block OOM Fails
* **Decision ID:** `DEC-012`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
During startup of the single-GPU AWQ model on Kaggle, the compilation process (`torch.compile`) consumed substantial GPU memory, leaving only `-0.1 GiB` of VRAM for the KV Cache blocks, which caused vLLM to crash with a `ValueError: No available memory for the cache blocks.`

#### 2. Before vs. After
* **Before:**
  * `gpu_memory_utilization` was set to `0.80`, leaving too little headroom for KV cache blocks after graph compilation overhead.
* **After:**
  * Increased `gpu_memory_utilization` to `0.90` (90%) to allocate more of the T4 GPU's VRAM for vLLM's cache blocks, which safely bypasses the compiler-induced VRAM deficit.

#### 3. Impacted Files
* [kaggle_context_synthesizer.py](file:///Users/mahmoud/Documents/nur/nur/scripts/kaggle_context_synthesizer.py) — Increased `gpu_memory_utilization` to `0.90`.

#### 4. Validation
* Updated the script and prepared it for paste-and-run on Kaggle.

### [2026-06-21T16:02:00-04:00] — Implementing Spawned Multiprocessing Parallel Context Generation over Dual GPUs
* **Decision ID:** `DEC-013`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
In dual GPU settings on Kaggle, our previous single-process sequential pipeline left one T4 GPU idle (0% utilization) while processing context synthesis on a single GPU (`tensor_parallel_size=1`). To cut context generation runtime in half, we needed to parallelize execution using both available T4 GPUs concurrently.

#### 2. Before vs. After
* **Before:**
  * Context generation ran on a single GPU sequentially, reloading the Qwen model three separate times for Quran, Hadith, and Tafsir chunks, taking twice as long.
* **After:**
  * Modified `kaggle_context_synthesizer.py` to support multi-GPU parallelization using Python's `spawn` multiprocessing context:
    * Merges all chunk files (Quran, Hadith, Tafsir) into a single unified sequence to run Qwen generation in a single pass, completely eliminating redundant model load cycles.
    * Automatically queries device count and divides the list equally into parallel GPU partitions.
    * Spawns worker subprocesses concurrently with isolated `CUDA_VISIBLE_DEVICES` environments (`0` and `1` respectively).
    * Implements temporary JSONL files for robust, zero-IPC data reassembly, preventing memory-bound serialization OOM blocks.
    * Automatically destroys worker processes to release 100% of GPU VRAM prior to loading the BGE-M3 embedding model on GPU 0.

#### 3. Impacted Files
* [kaggle_context_synthesizer.py](file:///Users/mahmoud/Documents/nur/nur/scripts/kaggle_context_synthesizer.py) — Implemented spawned multiprocessing and dataset merging.

#### 4. Validation
* Updated the script and prepared it for paste-and-run on Kaggle.

---

### [2026-06-21T21:05:00-04:00] — Verifying Local Database Sanity for Phase 2 Retrieval Pipeline
* **Decision ID:** `DEC-014`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
Before building the retrieval pipeline, we must verify that the local Python environment can connect to ChromaDB and read the 4 expected collections: `quran_dense`, `hadith_dense`, `tafsir_ar_dense`, and `tafsir_en_dense`.

#### 2. Before vs. After
* **Before:**
  * Local ChromaDB database contents were unzipped but not verified for availability, structure, and collection document counts.
* **After:**
  * Created `scripts/verify_db.py` to connect via `chromadb.PersistentClient` pointing to `./data/chroma_db` and count the documents in the collections.
  * Verified all 4 expected collections are fully loaded and accessible.

#### 3. Impacted Files
* [verify_db.py](file:///Users/mahmoud/Documents/nur/nur/scripts/verify_db.py) — Created script to check ChromaDB collections sanity.

#### 4. Validation
* Ran the script successfully, yielding:
  * `quran_dense`: 6,236 documents
  * `hadith_dense`: 33,738 documents
  * `tafsir_ar_dense`: 6,236 documents
  * `tafsir_en_dense`: 6,236 documents

---

### [2026-06-21T21:12:00-04:00] — Implementing and Testing Semantic Dense Retriever
* **Decision ID:** `DEC-015`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
With the ChromaDB database verified, we need to implement the first component of the retrieval pipeline: semantic (dense) search. This component queries specific collections using dense embedding vectors.

#### 2. Before vs. After
* **Before:**
  * No dedicated retriever module existed to connect to ChromaDB and run semantic searches.
* **After:**
  * Implemented `src/nur/retriever/dense.py` defining the `DenseRetriever` class.
  * Created `scripts/test_dense_search.py` to encode queries using a local `BGEM3FlagModel` on the `mps` device (Apple Silicon) and verify search results.

#### 3. Impacted Files
* [dense.py](file:///Users/mahmoud/Documents/nur/nur/src/nur/retriever/dense.py) — Created dense semantic retriever module.
* [test_dense_search.py](file:///Users/mahmoud/Documents/nur/nur/scripts/test_dense_search.py) — Created test script for dense search.

#### 4. Validation
* Handed off to user to execute and inspect local search outputs.

---

### [2026-06-21T22:05:00-04:00] — Cleaning Up RAG_PIPELINE_ARCHITECTURE.md Preamble and Unclosed Fence
* **Decision ID:** `DEC-016`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
The architecture doc had been updated via a chat-paste: a French preamble ("Oui, il y a une modification cruciale à faire...") and an orphan opening ` ```markdown ` fence with no matching close were left at the top of the file. This broke the document as a clean spec — agents reading it would treat conversational meta-text as part of the architecture.

#### 2. Before vs. After
* **Before:**
  * File started with French chat preamble explaining the change.
  * An opening ` ```markdown ` fence existed at line 5 with no closing fence.
  * File did not end with a trailing newline.
* **After:**
  * File starts directly with the `# NUR — RAG Pipeline Architecture` H1 title.
  * Only the legitimate JSON schema fences remain (lines 52 and 65).
  * Trailing newline added for POSIX compliance.

#### 3. Impacted Files
* [RAG_PIPELINE_ARCHITECTURE.md](file:///home/z/my-project/repos/nur/docs/RAG_PIPELINE_ARCHITECTURE.md) — Removed preamble and orphan fence.

#### 4. Validation
* `grep '```' docs/RAG_PIPELINE_ARCHITECTURE.md` returns only the two legitimate JSON schema fences.
* `head -5` confirms the file now starts with the H1 title.

---

### [2026-06-21T22:10:00-04:00] — Syncing config.py and .env.example with Llama 4 Scout Architecture
* **Decision ID:** `DEC-017`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
The architecture doc (`DEC-016` cleanup confirmed) specifies a Llama 4 Scout primary + Llama 3.3 70B reasoning fallback + Llama 3.1 8B local Ollama fallback lineup. But `src/nur/config.py` still held stale model names (`qwen/qwen3-32b`, `qwen/qwen3.6-27b` — the latter doesn't even exist as a real model). The `.env.example` had the same staleness plus a broken doc reference. Continuing into Phase 2 with these mismatches would have caused silent runtime failures when the generator module tried to call a non-existent model.

#### 2. Before vs. After
* **Before:**
  * `llm_primary = "qwen/qwen3-32b"` (no longer in the docs).
  * `llm_reasoning = "qwen/qwen3.6-27b"` (hallucinated model name — does not exist).
  * `llm_local = "qwen2.5:7b"` (docs specify `llama3.1:8b`).
  * No `llm_architect` field (the Step-1 Architect model was not configurable).
  * `default_lang: Literal["en", "ar"]` — Arabic was incorrectly listed as a synthesis language, violating Pillar 10 (Arabic is the source of truth, always displayed; not a synthesis toggle).
  * `.env` was loaded via relative path `"env_file='.env'"`, so `Settings()` only worked when the process CWD was the repo root.
  * API keys required the `NUR_` prefix (`NUR_GROQ_API_KEY`), incompatible with the official Groq SDK convention which reads `GROQ_API_KEY`.
  * `.env.example` referenced `docs/ARCHITECTURE.md` which does not exist (the real file is `docs/RAG_PIPELINE_ARCHITECTURE.md`).
* **After:**
  * Model lineup aligned 1:1 with `docs/RAG_PIPELINE_ARCHITECTURE.md` Section 3:
    * `llm_architect = "llama-3.1-8b-instant"` (Step 1).
    * `llm_primary = "meta-llama/llama-4-scout-17b-16e-instruct"` (Step 4, 30K TPM).
    * `llm_reasoning = "llama-3.3-70b-versatile"` (extreme Ikhtilaf fallback, 12K TPM).
    * `llm_local = "llama3.1:8b"` (offline Ollama fallback).
  * `default_lang: Literal["en", "fr"] = "en"` — Arabic correctly excluded.
  * `.env` loaded via absolute path `PROJECT_ROOT / ".env"`, so settings work from any CWD.
  * API-key fields (`groq_api_key`, `openrouter_api_key`, `ollama_base_url`) use `AliasChoices` to accept BOTH `NUR_GROQ_API_KEY` and the SDK-standard `GROQ_API_KEY`. This keeps compatibility with the official Groq Python SDK.
  * `.env.example` fully rewritten with current model names, clear pointers to source-of-truth docs, and a header explaining the `NUR_` prefix convention.
  * `.gitignore` extended with `*.zip` rule to prevent the 500MB indexed DB download from being accidentally staged.

#### 3. Impacted Files
* [config.py](file:///home/z/my-project/repos/nur/src/nur/config.py) — Updated model lineup, absolute `.env` path, AliasChoices for SDK compatibility, default_lang type fix.
* [.env.example](file:///home/z/my-project/repos/nur/.env.example) — Full rewrite to match current architecture.
* [.gitignore](file:///home/z/my-project/repos/nur/.gitignore) — Added `*.zip` rule for the Lightning DB download.

#### 4. Validation
* Ran `python -c "from src.nur.config import settings"` from `/tmp` (non-repo CWD) — confirmed settings load with the correct model names and the real `GROQ_API_KEY` (length 56) is picked up.
* `git status` confirms `.env` and `*.zip` files are NOT staged — only `.env.example`, `config.py`, and `.gitignore` are committed.

---

### [2026-06-21T22:30:00-04:00] — Implementing SparseRetriever with Inverted Index (Phase 2)
* **Decision ID:** `DEC-018`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
Phase 2 requires a SparseRetriever to complement the existing DenseRetriever (DEC-015). Per `docs/RAG_PIPELINE_ARCHITECTURE.md` Step 2, the two retrievers run in parallel and their results are fused via Reciprocal Rank Fusion (RRF). The sparse index files (`data/sparse/{source}_sparse.json`) already exist from Phase 1 ingestion — they store BGE-M3 lexical weights per chunk in the format `{chunk_id: {"indices": [...], "values": [...]}}`.

#### 2. Before vs. After
* **Before:**
  * No sparse retrieval module existed.
  * The Phase 1 retrieval audit (`DEC-004`) showed French queries had systematically `N/A` sparse ranks because the indexed text was purely Arabic. The LLM-synthesized bilingual context (DEC-005 through DEC-013) solved this at the data level — chunks now contain French and English keywords. But there was no code to actually query the sparse index at runtime.
* **After:**
  * Created `src/nur/retriever/sparse.py` defining the `SparseRetriever` class.
  * The class builds an **inverted index** on first load: `token_id → [(chunk_id, weight), ...]`. This is the standard sparse-retrieval data structure (same one BM25 uses). Query scoring iterates only over posting lists for tokens present in the query, making it O(sum of posting-list sizes) instead of O(all_chunks × avg_tokens_per_chunk).
  * Sparse JSON files are loaded **lazily** — only when a query targets that specific source. This avoids loading 100MB+ files for sources that aren't queried.
  * Created `scripts/test_sparse_search.py` — a model-free test that verifies the math via the self-similarity property (a chunk must rank #1 when its own vector is the query, because the dot product of a vector with itself = sum of squared weights = maximum possible score). This avoids the 2.3GB BGE-M3 download during testing; live query encoding is deferred to `benchmark_sparse.py` (to be run on the user's Mac where the model is cached).

#### 3. Impacted Files
* [sparse.py](file:///home/z/my-project/repos/nur/src/nur/retriever/sparse.py) — Created sparse retriever module with inverted index and lazy loading.
* [test_sparse_search.py](file:///home/z/my-project/repos/nur/scripts/test_sparse_search.py) — Created model-free test suite using self-similarity property.

#### 4. Validation
* Ran `python scripts/test_sparse_search.py` successfully. All 3 self-similarity tests passed:
  * `quran_1_1` (Bismillah) → rank 1, score 1.903861. Neighbors: other Al-Fatihah verses (shared vocabulary).
  * `quran_2_255` (Ayat al-Kursi) → rank 1, score 1.523489. Neighbors: other long theological verses.
  * `hadith_tirmidhi_1` (purification) → rank 1, score 1.742256. Neighbors: other Tirmidhi hadiths in the same chapter.
* The neighboring results are not just mathematically correct but semantically sensible — chunks sharing more tokens rank higher, which is exactly the lexical matching behavior we want.

---

## Future Architectural Plans

### [Phase 2] — LLM-Synthesized Contextual Retrieval via Kaggle GPUs
* **Goal:** Run `kaggle_context_synthesizer.py` on Kaggle with the AWQ model, single-GPU configuration, and force-uninstalled FlashInfer backend to generate and index enriched bilingual contexts.
* **Rationale:** Maximizes retrieval scores across all 3 target languages while maintaining zero local CPU runtime cost.





