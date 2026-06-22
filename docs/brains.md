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

### [2026-06-21T22:52:00-04:00] — Moving and Publishing .agents Configuration to GitHub
* **Decision ID:** `DEC-019`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
The user requested to make `.agents` public on GitHub. However, the Git repository resides in the `nur` subdirectory (`/Users/mahmoud/Documents/nur/nur`), while the `.agents` directory was located in the parent folder `/Users/mahmoud/Documents/nur/.agents`, which made it outside the Git repository. We needed to move `.agents` inside the repository while preserving its accessibility to the parent directory workspace via a symbolic link.

#### 2. Before vs. After
* **Before:**
  * `.agents` was in the parent workspace directory, untracked by Git.
  * Git push rejected due to divergent remote changes.
* **After:**
  * Moved `.agents` to `nur/.agents`.
  * Created a symbolic link `.agents → nur/.agents` at the parent workspace directory to maintain agent-system functionality.
  * Reconciled divergent Git branches using `git pull --rebase` and pushed `.agents` changes successfully to GitHub.

#### 3. Impacted Files
* [.agents](file:///Users/mahmoud/Documents/nur/nur/.agents) — Moved `.agents` config folder into the repository.
* [.gitignore](file:///Users/mahmoud/Documents/nur/nur/.gitignore) — Ensured no rules block `.agents`.

#### 4. Validation
* Ran `git status` to verify working tree is clean.
* Ran `git push` successfully pushing the local commit `äjout fichier .agent` to GitHub.

---

### [2026-06-21T23:00:00-04:00] — Formalizing Two New Developer Rules in .agents/rules/updatesmdfiles.md
* **Decision ID:** `DEC-020`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
During Phase 2 work on `sparse.py`, two process gaps surfaced:
1. The agent silently skipped testing the dense retriever because it could not download the 2.3GB BGE-M3 model on the server, and presented the work as "done" without flagging this to the user. This broke trust and forced the user to ask "why didn't you test?".
2. The agent completed `sparse.py` and committed it, but did not update `docs/PHASES.md` to reflect the new progress. The user had to explicitly request the update.

Both gaps are process problems, not code problems. They need to be encoded as rules so any future agent (or future-session of the same agent) follows them automatically.

#### 2. Before vs. After
* **Before:**
  * `.agents/rules/updatesmdfiles.md` existed but was an empty stub (only frontmatter, no content).
  * No rule mandated `docs/PHASES.md` updates after code changes.
  * No rule governed when architecture docs (`CONTEXT.md`, `PILLARS.md`, `RAG_PIPELINE_ARCHITECTURE.md`) could be edited vs. treated as frozen source-of-truth.
  * No rule prevented silent test skips.
* **After:**
  * `.agents/rules/updatesmdfiles.md` now contains three explicit rules:
    1. **Always update `docs/PHASES.md` after a code change** — in the same commit or the next one. Mark items `[x]`, mention test scripts for traceability.
    2. **Only edit architecture docs when a structural change is agreed** — code conforms to docs, not the other way around. Mismatches must be flagged, not silently "fixed".
    3. **Never skip tests silently** — if the agent cannot run a test (missing model, GPU, API key, disk space), it MUST tell the user explicitly with the phrasing pattern: *"I can't test this on my side because [reason]. Please run `python scripts/<name>.py` on your machine and paste me the output."*

#### 3. Impacted Files
* [updatesmdfiles.md](file:///home/z/my-project/repos/nur/.agents/rules/updatesmdfiles.md) — Filled the empty stub with three concrete rules.

#### 4. Validation
* Read the file back to confirm the three rules are present and unambiguous.
* The rules will be applied retroactively to all future commits in this session.

---

### [2026-06-21T23:15:00-04:00] — Implementing RRFFuser for Dense+Sparse Fusion (Phase 2)
* **Decision ID:** `DEC-021`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
With `DenseRetriever` (DEC-015) and `SparseRetriever` (DEC-018) both working, the next Phase 2 step per `docs/RAG_PIPELINE_ARCHITECTURE.md` Step 2 is to fuse their ranked lists into a single ranking. The two retrievers produce scores on completely different scales (dense cosine similarity ∈ [0,1] vs sparse dot-product unbounded), so naive addition or averaging would let whichever produces larger numbers dominate. Reciprocal Rank Fusion (RRF) solves this by operating on RANKS, not scores.

#### 2. Before vs. After
* **Before:**
  * No fusion module existed. Dense and sparse results could not be combined.
  * The Phase 1 retrieval audit (`DEC-004`) had hand-fused dense+sparse ranks for benchmarking purposes, but there was no reusable, tested implementation in the codebase.
* **After:**
  * Created `src/nur/retriever/fusion.py` defining the `RRFFuser` class.
  * Implements the exact RRF formula from the architecture doc: `score_rrf(c) = α × 1/(k + rank_dense(c)) + (1-α) × 1/(k + rank_sparse(c))` with `k=25`, `α=0.4` (dense weight), `1-α=0.6` (sparse weight).
  * Defaults are pulled from `settings.rrf_k` and `settings.rrf_alpha_dense` so changing them in `.env` propagates automatically.
  * Chunks appearing in only one list are NOT penalized with a synthetic low rank — they simply miss that term of the sum. This is standard RRF behavior.
  * Output is deterministic: ties (same RRF score) are broken by chunk ID alphabetically, so the same input always produces the same output. Critical for testing.
  * Returns enriched dicts with `dense_rank` and `sparse_rank` fields alongside `rrf_score`, so downstream code (and the future reranker in Phase 3) can see where each chunk came from.
  * Created `scripts/test_fusion.py` — a pure-math test suite (no BGE-M3, no ChromaDB, no network) with 9 test cases:
    1. Hand-computed scores (verified to 6 decimal places against manual calculation).
    2. Chunk in both lists gets boosted score (ranks #1).
    3. Chunk only in dense gets only the dense term.
    4. Chunk in neither list is excluded.
    5. `top_k` truncation.
    6. Empty inputs don't crash.
    7. Determinism (same inputs → identical output).
    8. Config defaults match the architecture doc (`k=25`, `α=0.4`).
    9. Invalid parameters (`k≤0`, `α∉[0,1]`) raise `ValueError`.

#### 3. Impacted Files
* [fusion.py](file:///home/z/my-project/repos/nur/src/nur/retriever/fusion.py) — Created RRF fusion module.
* [test_fusion.py](file:///home/z/my-project/repos/nur/scripts/test_fusion.py) — Created pure-math test suite (9 cases).

#### 4. Validation
* Ran `python scripts/test_fusion.py` — all 9 tests pass.
* The hand-computed test case (dense=[A,B,C], sparse=[B,A,D]) produced exactly the expected scores:
  * B (rank 2 dense, rank 1 sparse) → 0.037892 ✅
  * A (rank 1 dense, rank 2 sparse) → 0.037607 ✅
  * D (absent dense, rank 3 sparse) → 0.021429 ✅
  * C (rank 3 dense, absent sparse) → 0.014286 ✅
* Note: this script is fully testable on the agent side because RRF operates on ranks, not on raw scores — no BGE-M3 model or ChromaDB connection is needed. The integration test (real query → dense + sparse → fuse) is deferred to the future `benchmark_fusion.py` script, which the user will run on their Mac.

---

### [2026-06-21T23:45:00-04:00] — Implementing Generator (Architect + Reporter) with Groq + Instructor (Phase 2)
* **Decision ID:** `DEC-022`
* **Status:** Completed (code complete; live API test deferred to user per Rule 3)
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
Phase 2 needs the two-LLM "Smart Archivist" generation pipeline defined in `docs/RAG_PIPELINE_ARCHITECTURE.md`:
  - Step 1 (Architect): Decompose a complex user query into 1..N sub-questions targeting distinct jurisprudential themes. Uses `llama-3.1-8b-instant` on Groq (fast, cheap, high-volume).
  - Step 4 (Reporter): Generate a strict JSON report from the Top-10 retrieved chunks. Uses `meta-llama/llama-4-scout-17b-16e-instruct` on Groq (30K TPM, deep reasoning). Falls back to `llama-3.3-70b-versatile` for extreme Ikhtilaf cases.

The critical anti-hallucination requirement (Pillar 4) is that the Reporter must NEVER use its parametric knowledge — it can only report what the retrieved sources say. This is enforced structurally via the `instructor` library + Pydantic schemas that force the LLM to fill `{conflict_detection, direct_reports, synthesis}` before any synthesis is produced.

#### 2. Before vs. After
* **Before:**
  * `src/nur/generator/__init__.py` was an empty stub. No LLM calls were possible.
  * The architecture doc specified the LLM lineup (DEC-017 synced config.py) but no code consumed it.
* **After:**
  * Created `src/nur/generator/__init__.py` with 3 classes + 3 Pydantic models:
    * **`Architect`** — wraps `instructor.from_groq(client, mode=Mode.JSON_SCHEMA)`. Method `decompose(user_query) -> list[str]` returns 1..6 sub-questions. System prompt enforces same-language output and forbids answering the question.
    * **`Reporter`** — same instructor wrapper. Method `generate(user_query, sources_xml, force_reasoning=False) -> ReporterOutput`. System prompt enforces the "Strict Archivist" persona: no parametric knowledge, no invented logical links, verbatim Arabic text, mandatory source ID citation.
    * **`Generator`** — convenience facade grouping Architect + Reporter with a shared Groq client (one connection pool instead of two).
    * **Pydantic models** — `SubQuestions`, `DirectReport`, `ReporterOutput`. These define the strict JSON schemas that `instructor` forces the LLM to fill. Field-level descriptions tell the LLM exactly what each field must contain.
  * Retry logic via `tenacity` (3 attempts, exponential backoff 2-10s) handles transient 429/5xx errors. Phase 3 will add the Ollama offline fallback after retries are exhausted.
  * Created `scripts/test_generator.py` with 2 live API tests:
    1. Architect test — decomposes the marital-dilemma example query from the architecture doc.
    2. Reporter test — feeds 3 real zakat-related Quran chunks (pulled from ChromaDB and saved as `scripts/_fixtures_zakat.json`) and validates the structured output: source IDs match injected IDs, Arabic text is non-empty, synthesis cites valid `[SX]` references.
  * **SDK verification (Rule 7 — never trust memory for library APIs):** Inspected the installed `groq==1.5.0` SDK signature for `chat.completions.create` directly. Confirmed that `meta-llama/llama-4-scout-17b-16e-instruct`, `llama-3.1-8b-instant`, and `llama-3.3-70b-versatile` are all in the accepted model literal. Confirmed `response_format`, `temperature`, `max_tokens`, `frequency_penalty` parameters exist. Confirmed `instructor.from_groq` and `instructor.Mode.JSON_SCHEMA` exist in `instructor==1.15.3`.

#### 3. Impacted Files
* [generator/__init__.py](file:///home/z/my-project/repos/nur/src/nur/generator/__init__.py) — Created Architect, Reporter, Generator classes + Pydantic schemas + system prompts.
* [test_generator.py](file:///home/z/my-project/repos/nur/scripts/test_generator.py) — Created live API test script (2 tests).
* [_fixtures_zakat.json](file:///home/z/my-project/repos/nur/scripts/_fixtures_zakat.json) — 3 real zakat-related Quran chunks pulled from ChromaDB, used as Reporter test input.

#### 4. Validation
* **Agent-side validation (passed):**
  * Module imports cleanly: `from src.nur.generator import Architect, Reporter, Generator, SubQuestions, DirectReport, ReporterOutput` works.
  * Pydantic schemas validate against synthetic data (constructed a `ReporterOutput` manually, all fields accepted).
  * System prompts contain the critical rules (`"NEVER use your pre-trained parametric knowledge"`, `"character-for-character"`, `"MUST explicitly cite source IDs"`).
  * Fixture loading + XML rendering via existing `render_sources_for_prompt()` produces a 1405-char prompt with 3 `<document>` blocks.
* **Live API validation (DEFERRED TO USER per Rule 3):**
  * The agent's GROQ_API_KEY returns HTTP 403 Forbidden on both `models.list()` and `chat.completions.create`. This is likely a region or scope restriction on the key — the key itself is valid (length 56, correct `gsk_` prefix), but the agent server is not authorized to call the API.
  * The user MUST run `python scripts/test_generator.py` on their Mac to validate:
    1. The Architect returns 1-6 sub-questions in the same language as the query.
    2. The Reporter returns a `ReporterOutput` with valid source IDs, non-empty Arabic text, and a synthesis that cites `[SX]` IDs that exist in `direct_reports`.
  * The test script prints clear diagnostics if the API call fails (missing key, rate limit, network).

---

### [2026-06-22T00:30:00-04:00] — Creating Data Sources & Provenance Document
* **Decision ID:** `DEC-023`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
The user raised a critical religious-integrity concern: if a user one day reports that a hadith in NUR's output is fabricated (`Mawdu'`) or that a Quranic verse is mis-transcribed, we need to be able to trace the chunk back to its original upstream source in seconds. The provenance info existed only inside the docstrings of the 3 download scripts (`01_download_quran.py`, `02_download_hadith.py`, `03_download_tafsir.py`) — nowhere in `docs/`. If a script gets deleted or rewritten, the traceability is lost.

This is especially important for an Islamic project where a wrong answer is theologically harmful (Pillar 4 — Absolute Reliability). The user explicitly asked: "we need to have source where we took them if one day there's a problem about one being false and haram".

#### 2. Before vs. After
* **Before:**
  * Source URLs were buried in download-script docstrings.
  * No central document recorded the license of each upstream source.
  * No documented procedure for auditing a problematic hadith or verse.
  * No chain-of-custody diagram showing the full pipeline from upstream source to final ChromaDB chunk.
  * `docs/CONTEXT.md` section 5 (Guiding Documents) referenced a non-existent `AGENTS.md` file and did not mention the existing `RAG_PIPELINE_ARCHITECTURE.md`.
* **After:**
  * Created `docs/DATA_SOURCES.md` with 9 sections covering:
    1. Summary table of the 3 upstream sources and what they contribute (52,446 chunks total).
    2. Source 1 — Quran (alquran.cloud): editions, upstream provider, license, download script, verification commands.
    3. Source 2 — Hadith (meeAtif/hadith_datasets on HuggingFace): 6 collections, per-hadith fields, the critical "do NOT construct sunnah.com URLs dynamically" rule, verification commands.
    4. Source 3 — Tafsir Ibn Kathir (spa5k/tafsir_api on GitHub): both AR + EN editions, full chain of custody (qul.tarteel.ai → spa5k → NUR), the upstream `tafisr` typo warning, verification commands.
    5. Full chain-of-custody ASCII diagram: upstream → download scripts → normalize → LLM context synthesis → BGE-M3 embedding → ChromaDB + sparse JSON.
    6. Re-download instructions (full pipeline + Lightning AI fast path).
    7. **Audit procedure** — step-by-step what to do if a user reports a problematic hadith: identify the chunk by `source_id`, locate it in ChromaDB, verify against the upstream URL, re-download if corrupted, report upstream if the upstream itself is wrong. Explicit rule: "Do NOT silently correct the data in NUR's local copy — log the discrepancy in docs/brains.md with a new Decision ID."
    8. License summary table (alquran.cloud has no explicit license; meeAtif is MIT; spa5k is MIT).
    9. Known issues & caveats: the tafsir folder typo, hadith grade normalization, missing checksums (Phase 8 task), alquran.cloud's lack of formal license, Tanzil as the deeper fallback.
  * Updated `docs/CONTEXT.md` section 5 to list `DATA_SOURCES.md` and the existing `RAG_PIPELINE_ARCHITECTURE.md`, and to point to `.agents/rules/` instead of the non-existent `AGENTS.md`.

#### 3. Impacted Files
* [DATA_SOURCES.md](file:///home/z/my-project/repos/nur/docs/DATA_SOURCES.md) — Created the provenance and audit document.
* [CONTEXT.md](file:///home/z/my-project/repos/nur/docs/CONTEXT.md) — Updated section 5 to list the new doc and fix the AGENTS.md reference.

#### 4. Validation
* Fetched the upstream licenses directly via `curl` to verify the documented licenses are accurate:
  * `meeAtif/hadith_datasets` README YAML frontmatter confirms `license: mit`.
  * `spa5k/tafsir_api` LICENSE file confirms MIT, copyright 2023 Spark.
  * `alquran.cloud` has no explicit LICENSE file — confirmed by inspecting the API and the islamic.network site; the document records this honestly.
* Fetched the spa5k README's tafsir source table to confirm the deeper upstream: AR Ibn Kathir ← `qul.tarteel.ai/resources/tafsir/22`, EN Ibn Kathir ← `qul.tarteel.ai/resources/tafsir/35`.
* Confirmed Phase 1 ingestion date via `git log -- scripts/0[1-3]_download_*.py`: **2026-06-21**.
* Verified the `tafisr` typo by directly fetching both URL variants on GitHub — only `en-tafisr-ibn-kathir/` returns 200.

---

### [2026-06-22T01:00:00-04:00] — Fixing Architect instructor Mode + Formalizing "Never Guess External Capabilities" Rule
* **Decision ID:** `DEC-024`
* **Status:** Completed (Architect fix applied; awaiting user re-test)
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
The user ran `scripts/test_generator.py` on their Mac. TEST 2 (Reporter) passed perfectly — 3 direct reports, all source IDs cited, Arabic preserved verbatim, "Strict Archivist" persona working. But TEST 1 (Architect) failed with HTTP 400:
```
This model does not support response format `json_schema`.
See supported models at https://console.groq.com/docs/structured-outputs#supported-models
```
Root cause: the agent used `instructor.Mode.JSON_SCHEMA` for the Architect, but `llama-3.1-8b-instant` does not support the `json_schema` response format. Only newer models like Llama 4 Scout support it.

This was a hallucination on the agent's part — the agent assumed `json_schema` mode works on all Groq models based on training memory, without verifying. The agent tried to fetch the Groq structured-outputs doc to verify compatibility, but the page is JS-rendered and not scrapable. Rather than asking the user, the agent proceeded with the unverified assumption.

The user also gave the agent the Groq rate limit table (saved to `/home/z/my-project/upload/Pasted Content_1782098886176.txt`) and established a new rule: "if you can't access it just ask me — if I won't [be able to either] I will tell you no, that's it. It will save us so much time and hallucination."

#### 2. Before vs. After
* **Before:**
  * `Architect.__init__` used `instructor.Mode.JSON_SCHEMA` — failed on `llama-3.1-8b-instant` with HTTP 400.
  * No rule prevented the agent from guessing external capabilities (rate limits, model features, API compatibility).
  * No persistent doc recorded Groq's rate limits or structured-output mode compatibility — the agent would have to re-look-them-up (or re-guess) every session.
  * The agent had earlier (DEC-017) called `qwen3.6-27b` a "hallucinated model name" — this was WRONG. The rate limit table the user provided confirms `qwen/qwen3.6-27b` is a real Groq model (30 RPM, 1K RPD, 8K TPM). The agent owned this error in the conversation.
* **After:**
  * **Architect fix**: Switched `Architect.__init__` from `instructor.Mode.JSON_SCHEMA` to `instructor.Mode.TOOLS` (function calling). Function calling is universally supported across Groq chat models. The Architect's schema is simple (a flat list of strings via `SubQuestions`), so tool-call enforcement is more than sufficient. The Reporter keeps `Mode.JSON_SCHEMA` because Scout supports it and the Reporter's nested `direct_reports` schema needs the strictest enforcement.
  * **New rule** (`.agents/rules/updatesmdfiles.md` Rule 4): "Never Guess External Capabilities — Ask the User to Verify". If the agent cannot access an external doc/API reference/rate-limit page, it MUST ask the user to check it or paste the relevant section. The user can say "I can't access it either" — that's acceptable; the agent then proceeds with last known-good values from `docs/GROQ_REFERENCE.md` and marks the assumption as unverified. NEVER hallucinate from training memory.
  * **New doc** `docs/GROQ_REFERENCE.md`: Persistent record of verified Groq rate limits (Free plan table for all models) + structured-output mode compatibility (`json_schema` vs `json_object` vs tools) + which mode each NUR role uses + rate-limit response headers + when to re-verify. This doc is the single source of truth so we never re-guess Groq capabilities.
  * Updated `docs/CONTEXT.md` section 5 to list `GROQ_REFERENCE.md` and updated the `.agents/rules/` description to mention the new "no guessing external capabilities" rule.

#### 3. Impacted Files
* [generator/__init__.py](file:///home/z/my-project/repos/nur/src/nur/generator/__init__.py) — Architect now uses `Mode.TOOLS` instead of `Mode.JSON_SCHEMA`; docstring updated with the reasoning.
* [GROQ_REFERENCE.md](file:///home/z/my-project/repos/nur/docs/GROQ_REFERENCE.md) — Created persistent Groq rate limits + mode compatibility reference.
* [updatesmdfiles.md](file:///home/z/my-project/repos/nur/.agents/rules/updatesmdfiles.md) — Added Rule 4: "Never Guess External Capabilities — Ask the User to Verify".
* [CONTEXT.md](file:///home/z/my-project/repos/nur/docs/CONTEXT.md) — Section 5 now lists `GROQ_REFERENCE.md` and mentions the new rule.
* [PHASES.md](file:///home/z/my-project/repos/nur/docs/PHASES.md) — Generator checklist entry updated to reflect the Reporter pass + Architect fix.

#### 4. Validation
* **Reporter (TEST 2) — PASSED on user's Mac** (DEC-022 validation complete):
  * 3 `direct_reports` entries, one per injected source (S1, S2, S3).
  * All 3 source IDs cited in synthesis.
  * Arabic text preserved verbatim (e.g. `يَسْـَلُونَكَ مَاذَا يُنفِقُونَ...`).
  * Factual reports without external logic (S2 correctly says "criticizes those who do not encourage feeding the poor" — did NOT extrapolate to "therefore we should feed the poor").
  * "Strict Archivist" persona is working — no parametric knowledge leaked.
* **Architect (TEST 1) — fix applied, awaiting user re-test**:
  * Agent verified the fix imports cleanly: `from src.nur.generator import Architect; a = Architect()` works (no API call needed to verify the mode switch).
  * The `Mode.TOOLS` approach is confirmed compatible with `llama-3.1-8b-instant` by the Groq API reference (function calling is universally supported on Groq chat models, per the `tools` parameter docs).
  * User MUST re-run `python scripts/test_generator.py` on their Mac to confirm the Architect now returns 1-6 sub-questions.
* **Honest correction logged**: `qwen3.6-27b` IS a real Groq model — the agent's earlier claim in DEC-017 that it was "hallucinated" was itself a hallucination. The rate limit table provided by the user confirms the model exists (30 RPM, 1K RPD, 8K TPM, 200K TPD). This does not change the architecture (NUR uses Llama 4 Scout as primary, not Qwen), but the record is corrected.

---

### [2026-06-22T01:30:00-04:00] — Implementing NURPipeline Orchestrator (Phase 2)
* **Decision ID:** `DEC-025`
* **Status:** Completed (code complete; live integration test deferred to user)
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
With the Architect + Reporter (DEC-022, DEC-024) and the retriever trio (DEC-015, DEC-018, DEC-021) all validated, the remaining Phase 2 component is `pipeline.py` — the orchestrator that wires them together into the 5-step "Smart Archivist" flow defined in `docs/RAG_PIPELINE_ARCHITECTURE.md`. Without this orchestrator, a user would have to manually call 7 components in sequence with complex data transformations between them. The pipeline encapsulates all of that into a single `query(user_question)` call.

#### 2. Before vs. After
* **Before:**
  * No orchestrator existed. The 7 Phase 2 components (DenseRetriever, SparseRetriever, RRFFuser, Architect, Reporter, Generator, SourceRef) were individually tested but never wired together.
  * No code handled the multi-query hybrid retrieval pattern (Step 2): encoding each sub-question once, searching all 4 collections per query, fusing dense+sparse per (query, source), and deduplicating across the entire pool.
  * No code converted ChromaDB metadata back into `SourceRef` objects for the 4 different source types (quran, hadith, tafsir_ar, tafsir_en) — each has a different metadata schema.
* **After:**
  * Created `src/nur/pipeline.py` defining `NURPipeline` and `PipelineResult`.
  * **`NURPipeline`** class with:
    * Lazy BGE-M3 loading (2.3GB model loads on first `query()` call, not at construction — keeps instantiation fast for debugging).
    * Auto device detection (CUDA > MPS > CPU).
    * `_encode_query(query)` — encodes a query string into (dense_vector, sparse_vector) in one BGE-M3 call.
    * `_retrieve(queries, top_k)` — multi-query hybrid retrieval: for each query, encodes it once, then for each of the 4 sources runs dense + sparse + RRF fusion. Deduplicates by chunk ID keeping the MAX RRF score. Returns top 30.
    * `_chunks_to_source_refs(chunks)` — batch-fetches metadata from ChromaDB for the top 10 chunks and constructs `SourceRef` objects. Handles all 4 metadata schemas (quran: `text_ar/en`, hadith: `text_ar/en` + `collection` + `hadith_number` + `grade`, tafsir_ar: `tafsir_text_ar`, tafsir_en: `tafsir_text`).
    * `query(user_query, force_reasoning=False)` — the main entry point. Runs all 5 steps and returns a `PipelineResult` with every intermediate result for transparency.
  * **`PipelineResult`** dataclass with: `user_query`, `sub_questions`, `retrieved_chunks` (top 30), `top_chunks` (top 10 SourceRefs), `report` (ReporterOutput), `error`. The CLI will use this to display a rich, transparent response.
  * Created `scripts/test_pipeline.py` — full integration test with `--query` and `--force-reasoning` CLI flags. Prints all intermediate steps (sub-questions, retrieved chunks with RRF scores + ranks, top 10 SourceRefs, final report) and runs 4 validation checks (sub-question count, retrieval non-empty, SourceRefs built, report structure + source ID citation).

#### 3. Impacted Files
* [pipeline.py](file:///home/z/my-project/repos/nur/src/nur/pipeline.py) — Created NURPipeline orchestrator + PipelineResult dataclass.
* [test_pipeline.py](file:///home/z/my-project/repos/nur/scripts/test_pipeline.py) — Created full integration test script.

#### 4. Validation
* **Agent-side validation (passed):**
  * Module imports cleanly: `from src.nur.pipeline import NURPipeline, PipelineResult` works.
  * `_build_source_ref` static method constructs valid SourceRef objects for all 4 source types (verified with synthetic metadata for quran, hadith, tafsir_ar, tafsir_en).
  * Source IDs generate correctly: `SRC-QURAN-1-1`, `SRC-HADITH-BUKHARI-1`, `SRC-TAFSIR-AR-1-1`, `SRC-TAFSIR-EN-1-1`.
  * URL generation works for all 4 types (quran.com, sunnah.com, quran.com/tafsir).
  * Grade weight computation works for hadith (Sahih → 1.3).
  * `PipelineResult` dataclass has all expected fields.
  * Test script imports cleanly and argparse defaults are correct.
* **Live integration validation (DEFERRED TO USER per Rule 3):**
  * The agent server cannot run `test_pipeline.py` because it lacks:
    1. BGE-M3 model weights (2.3GB, disk space limited)
    2. A working Groq API key (returns 403)
    3. Sufficient compute for BGE-M3 inference (CPU-only torch)
  * The user MUST run `python scripts/test_pipeline.py` on their Mac to validate the full end-to-end flow. The test script prints clear diagnostics at each step and runs 4 validation checks on the output.

---

### [2026-06-22T02:00:00-04:00] — Pipeline Integration Test Passed + Actual Token Usage Recorded
* **Decision ID:** `DEC-026`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
The user ran `scripts/test_pipeline.py` on their Mac with the default query "What does the Quran say about charity and zakat?". The full pipeline executed end-to-end and all 4 validation checks passed. The user also provided two Groq dashboard screenshots showing actual token usage per model.

#### 2. Before vs. After
* **Before:**
  * Pipeline code was written but not live-tested (DEC-025 deferred to user).
  * Token usage estimates in `docs/GROQ_REFERENCE.md` were guesses (~5,100 tokens per query).
* **After:**
  * Pipeline validated end-to-end: 12.2s total, 4 sub-questions, 30 chunks retrieved, 4 direct reports, all source IDs cited.
  * Retrieved exactly the right verses: 9:60 (8 zakat categories), 2:273 (restricted poor), 70:25 (deprived), 19:31 (prayer + zakah).
  * `docs/GROQ_REFERENCE.md` updated with verified token data:
    - Architect: 657 input + 60 output = 717 tokens (12% of 6K TPM)
    - Reporter: 16,600 input + 937 output = 17,537 tokens (59% of 30K TPM)
    - Total per query: ~18,254 tokens
    - Throughput ceiling: ~1 query per minute (TPM-limited, not RPM-limited)
    - Dashboard shows 25 RPM, not 30 RPM as docs state — treat 25 as the real ceiling.

#### 3. Impacted Files
* [GROQ_REFERENCE.md](file:///home/z/my-project/repos/nur/docs/GROQ_REFERENCE.md) — Replaced estimated token math with verified dashboard data.
* [PHASES.md](file:///home/z/my-project/repos/nur/docs/PHASES.md) — Marked pipeline.py as fully validated.

#### 4. Validation
* User's test output confirmed all 4 checks passed (Architect, Retrieval, SourceRefs, Report structure + citation).
* Token data extracted from screenshots via VLM (vision language model) and recorded.

---

### [2026-06-22T02:15:00-04:00] — Implementing Hadith Grade Education + Two-Tier Warning System
* **Decision ID:** `DEC-027`
* **Status:** Completed (code complete; live test deferred to user)
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
The user raised a critical theological concern: showing "Sahih" without explaining what it means is irresponsible — a layperson might not know the difference between Sahih (authentic) and Mawdu' (fabricated). The user also specified the exact warning logic: "if there is no sahih or hasan it should be a big watchout and if there is it should be small warning additional context to warn max people."

This implements Pillar 3 (Authenticity-Weighted Retrieval) education layer + Pillar 4 (Post-Generation Verification) enrichment + Pillar 8 (Scholar Opinions Are Mandatory) by noting that grades come from scholars.

#### 2. Before vs. After
* **Before:**
  * Hadith grades were stored in metadata and sent to the LLM in the prompt, but NOT displayed to the user with education.
  * No warning system existed for weak or fabricated hadiths.
  * The LLM's `DirectReport` schema did not include grade fields.
* **After:**
  * Created `src/nur/grades.py` with:
    * `WarningLevel` IntEnum: NONE (0) < SMALL (1) < BIG (2) < CRITICAL (3). IntEnum allows severity comparison.
    * `GRADE_INFO` dict: 5 grade levels (sahih, hasan, daif, mawdu, unknown) each with label, plain-language explanation, per-hadith warning level, and warning text.
    * `normalize_grade_level(grade_string)`: parses raw grade strings like "Sahih (Darussalam)" into normalized levels. Conservative — returns "unknown" rather than guessing.
    * `get_grade_info(grade_string)`: returns the full info dict for a grade string.
    * `get_answer_warning(grade_levels)`: implements the two-tier logic — CRITICAL if any Mawdu', BIG if only Da'if (no Sahih/Hasan), SMALL if Da'if alongside Sahih/Hasan, NONE otherwise. CRITICAL takes precedence.
  * Updated `src/nur/pipeline.py`:
    * `PipelineResult` gained `answer_warning: str | None` and `grade_explanations: dict[str, str]` fields.
    * New `_enrich_report_with_grades(result)` method: runs AFTER the Reporter generates its output, looks up each cited hadith's grade from the SourceRef metadata (NOT from the LLM), attaches the explanation to `grade_explanations`, and computes the answer-level warning. This is a Pillar 4 post-generation enrichment — the grade comes from the actual metadata, preventing the LLM from hallucinating or misreading grades.
    * `query()` method now calls `_enrich_report_with_grades` as Step 4b.
  * Updated `scripts/test_pipeline.py`:
    * `print_report()` now shows the answer-level warning FIRST (before the report), so the user sees the most critical information before reading the answer.
    * Each direct_report that has a grade explanation shows it with a 📚 icon.

#### 3. Impacted Files
* [grades.py](file:///home/z/my-project/repos/nur/src/nur/grades.py) — Created grade education + warning system.
* [pipeline.py](file:///home/z/my-project/repos/nur/src/nur/pipeline.py) — Added grade enrichment step + PipelineResult fields.
* [test_pipeline.py](file:///home/z/my-project/repos/nur/scripts/test_pipeline.py) — Updated to display warnings + grade explanations.
* [PHASES.md](file:///home/z/my-project/repos/nur/docs/PHASES.md) — Updated pipeline.py checklist entry.

#### 4. Validation
* **Agent-side validation (passed):**
  * `grades.py` module imports cleanly.
  * Grade normalization: tested 10 raw grade strings → all normalize correctly (Sahih, Hasan, Da'if, Mawdu, Ungraded, empty, None).
  * Answer warning logic: tested 8 grade-level combinations → all produce the correct warning level (NONE, SMALL, BIG, CRITICAL) per the two-tier logic.
  * `WarningLevel` IntEnum comparison works: CRITICAL > BIG > SMALL > NONE.
  * Pipeline imports cleanly with the new grade integration.
  * `PipelineResult` has the new `answer_warning` and `grade_explanations` fields.
* **Live integration validation (DEFERRED TO USER per Rule 3):**
  * The user MUST re-run `python scripts/test_pipeline.py` to see the grade warnings in action. For the default "charity and zakat" query, the answer contains Quran verses + 1 Sahih hadith (Bukhari #4661) → expected: NO answer warning, but the hadith's grade explanation should appear.

---

### [2026-06-22T02:45:00-04:00] — Implementing the CLI (Rich + Typer Interactive Terminal Chatbot)
* **Decision ID:** `DEC-028`
* **Status:** Completed (code complete; live test deferred to user)
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
The final Phase 2 component per `docs/PHASES.md`: a user-facing terminal interface that wraps `NURPipeline.query()` into a beautiful, interactive, theologically-safe display. Without this, the pipeline is only accessible via test scripts. The CLI is what makes NUR a "chatbot" rather than a library.

#### 2. Before vs. After
* **Before:**
  * The pipeline could only be invoked via `scripts/test_pipeline.py` — a test script, not a user-facing tool.
  * No interactive mode existed — every query required re-running a script.
  * The output was plain text, not formatted.
* **After:**
  * Created `src/nur/cli.py` — a Rich + Typer CLI with two modes:
    * **Single-query mode**: `python -m nur "What does the Quran say about charity?"` — answers one question and exits.
    * **Interactive REPL mode**: `python -m nur` — enters a chat loop at a `NUR>` prompt. User can ask multiple questions. Type `exit` or `quit` or Ctrl+C to leave.
  * Created `src/nur/__main__.py` — enables `python -m nur` invocation.
  * Added `[project.scripts]` entry point to `pyproject.toml` (`nur = "nur.cli:app"`) so after `pip install -e .` the user can run `nur` directly.
  * **Display order (pillar-compliant)**:
    1. Answer warning (Pillar 4 — must be seen FIRST, before the answer). CRITICAL = red background panel, BIG = red border, SMALL = yellow border.
    2. Conflict detection / Ikhtilaf (Pillar 9 — only shown if a conflict was detected, not the default "no conflict" message).
    3. Synthesis — the main answer in a cyan panel.
    4. Direct reports — each source as a sub-panel with: source ID + label, clickable URL (`[link=URL]` Rich markup), grade (for hadiths), Arabic text (always visible, large, with "العربية:" label — Pillar 10), English report, grade explanation (📚 icon — DEC-027 education layer).
    5. Sub-questions (verbose only — transparency layer showing how the Architect decomposed the query).
    6. Retrieval summary (verbose only — top 5 chunks by RRF score).
    7. Error panel (if any).
  * **Flags**: `--lang fr` (force French synthesis), `--force-reasoning` (use 70B model), `--no-arabic` (hide Arabic for terminals without font support), `--verbose` (show sub-questions + retrieval).
  * **Pipeline singleton**: `get_pipeline()` caches the NURPipeline instance so subsequent queries in interactive mode don't re-load BGE-M3.
  * Used relative imports (`.config`, `.pipeline`) instead of absolute (`src.nur.config`) so the module works correctly as a proper Python package when invoked via `python -m nur`.

#### 3. Impacted Files
* [cli.py](file:///home/z/my-project/repos/nur/src/nur/cli.py) — Created the Rich + Typer CLI with single-query + interactive modes.
* [__main__.py](file:///home/z/my-project/repos/nur/src/nur/__main__.py) — Created module entry point for `python -m nur`.
* [pyproject.toml](file:///home/z/my-project/repos/nur/pyproject.toml) — Added `[project.scripts]` entry point: `nur = "nur.cli:app"`.

#### 4. Validation
* **Agent-side validation (passed):**
  * Module imports cleanly as a package: `from nur.cli import app` works.
  * `python -m nur --help` displays the full help text with all 4 flags and examples.
  * Typer app has 1 registered command with correct argument + options.
  * All 8 display functions present (display_answer_warning, display_synthesis, display_conflict_detection, display_direct_reports, display_sub_questions, display_retrieval_summary, display_error, display_result).
  * Relative imports work correctly when invoked via `python -m nur`.
  * Pre-flight check for `GROQ_API_KEY` exits with a clear error if missing.
* **Live integration validation (DEFERRED TO USER per Rule 3):**
  * The agent server cannot run the CLI live because it lacks BGE-M3, ChromaDB, and a working Groq API key.
  * The user MUST run on their Mac:
    ```
    python -m nur "What does the Quran say about charity?"
    ```
    or for interactive mode:
    ```
    python -m nur
    ```
  * Expected: beautiful Rich-formatted output with Arabic text, clickable URLs, grade explanations, and (if applicable) warning panels.

---

### [2026-06-22T03:30:00-04:00] — Verifying BGE-reranker-v2-m3 Compatibility + Recall Gap Analysis
* **Decision ID:** `DEC-029`
* **Status:** Completed
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
Phase 3 requires the `bge-reranker-v2-m3` cross-encoder reranker. Before writing the module, the agent needed to verify: (1) the model downloads and loads, (2) the `compute_score` API works, (3) the scores are meaningful for Islamic-text relevance, (4) the sigmoid normalization gives 0-1 scores for the abstention threshold (0.35 from the architecture doc). Additionally, the user identified a critical recall problem: the Phase 2 retriever was missing key verses (Quran 2:43, 4:103, Bukhari #8) for the "Is prayer obligatory?" query — meaning reranking 30 chunks cannot recover them.

#### 2. Before vs. After
* **Before:**
  * Reranker not installed. Unknown if it works with the installed `FlagEmbedding==1.4.0` + `transformers==5.12.1`.
  * Recall gap unmeasured — the agent assumed RRF top-30 contained the relevant chunks.
  * Abstention threshold (0.35) was theoretical — unknown if scores land in that range.
* **After:**
  * **Compatibility fix**: `transformers==5.12.1` broke `FlagEmbedding==1.4.0` (the `prepare_for_model` method was removed in transformers 5.x). Downgraded to `transformers==4.57.6` — both BGE-M3 and the reranker now work. This is a real version-pin requirement: `transformers>=4.44,<5.0`.
  * **Reranker verified working**: tested 5 chunk pairs against "Is prayer obligatory?". Results:
    - "prayer = 2nd pillar obligatory" → raw +7.43 / sigmoid **0.993** (most relevant)
    - "prayer = pillar" → raw +3.12 / sigmoid ~0.96
    - "Friday prayer abandonment" → raw -5.39 / sigmoid ~0.005 (tangential — exactly the hadith S5 that caused the Phase 2 quality issue!)
    - "charity", "fasting" → raw -6.x / sigmoid ~0.00002 (off-topic)
  * **Normalization confirmed**: `compute_score(pairs, normalize=True)` applies sigmoid internally, giving scores in [0, 1]. The 0.35 abstention threshold from the architecture doc is directly applicable.
  * **Recall gap confirmed**: from the user's Phase 2 test output for "Is prayer obligatory?", the retrieved chunks (S1-S10) did NOT include Quran 2:43, 4:103, or the 5-pillars hadith (Bukhari #8). The retriever missed the most direct proofs. Reranking cannot recover what the retriever never found.
  * Created `scripts/test_recall.py` — a minimal recall audit script that prints the top 100 retrieved chunks as a flat list (no panels, no formatting) for easy copy-paste. Includes a `RECALL_CHECKS` dict listing known-correct chunks per query, so the user can immediately see if key verses appear in the results.

#### 3. Impacted Files
* [test_recall.py](file:///home/z/my-project/repos/nur/scripts/test_recall.py) — Created recall audit script with flat-list output and expected-chunk checks.
* [PHASES.md](file:///home/z/my-project/repos/nur/docs/PHASES.md) — Phase 3 checklist expanded with concrete steps + verification status.

#### 4. Validation
* Reranker loads in 2.4s on CPU (server has no MPS/CUDA). On the user's M4 Mac with MPS, it will be faster.
* `compute_score` with `normalize=True` returns sigmoid scores — verified with 5 test pairs.
* The tangential hadith (Ibn Majah #1125 about Friday prayer) scores ~0.005, confirming the reranker would have filtered it out in Phase 2 if it had been in place.
* Recall audit script imports cleanly. The user will run it on their Mac to measure actual recall for "Is prayer obligatory?" with top-100.
* **Version pin discovered**: `transformers>=4.44,<5.0` is required for FlagEmbedding 1.4.0 compatibility. This must be added to `requirements.txt` in a future commit.

---

### [2026-06-22T04:30:00-04:00] — Discovering Critical Data-Quality Issue: Quran Chunks Use Global Ayah Numbering
* **Decision ID:** `DEC-030`
* **Status:** Completed (documented; fix deferred to Phase 8)
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
During Phase 3 recall testing, the agent built a recall audit script (`scripts/test_recall.py`) that checked whether specific "expected" Quran verses (2:43, 4:103, etc.) appeared in the retriever's top-100 results. The first run reported 9% recall, which seemed implausibly low given that the top-10 chunks were visibly about prayer. The user correctly insisted: "don't trust your DB to identify which verses are about prayer — search online for the authoritative list." This led to discovering that the DB uses a non-standard ayah numbering scheme.

#### 2. Before vs. After
* **Before:**
  * The agent assumed the DB's `ayah_num` metadata field used standard surah:ayah numbering (e.g., `quran_2_43` = standard 2:43).
  * Recall checks used standard verse numbers and reported 9% recall — but this was a false alarm caused by the numbering mismatch.
  * The agent initially tried to "fix" the recall check by using the DB's chunk IDs directly, but this was also wrong because the agent's "expected" list was based on standard numbering.
* **After:**
  * **Verified the authoritative verse list** by querying the alquran.cloud search API (`http://api.alquran.cloud/v1/search/establish%20prayer/all/en.sahih`), which returned 35 verses with their standard `numberInSurah` values. This is the SAME edition our DB uses (en.sahih), so it's the canonical reference.
  * **Verified the DB numbering by text-content matching**: for each API verse, found the DB chunk whose `text_en` starts with the same prefix. This produced an unambiguous mapping:
    - Standard 2:3 → DB `quran_2_10` (ayah_num=10, offset +7)
    - Standard 2:43 → DB `quran_2_50` (ayah_num=50, offset +7)
    - Standard 4:77 → DB `quran_4_570` (ayah_num=570, offset +493)
    - Standard 9:5 → DB `quran_9_1240` (ayah_num=1240, offset +1235)
  * The offset increases with each surah, confirming **cumulative global numbering** (1 to 6,236) rather than per-surah numbering.
  * **Root cause identified**: the Phase 1 ingestion script `scripts/01_download_quran.py` used alquran.cloud's `/quran/quran-uthmani` endpoint, which returns `numberInQuran` (global) instead of `numberInSurah` (standard). The chunking script then used this global number as both the chunk ID suffix and the `ayah_num` metadata field.
  * **Impact documented in `docs/DATA_SOURCES.md` section 9, issue #6** (new): the `SourceRef.source_id`, `url`, and `display_label` all use the DB's `ayah_num`, so they show the WRONG verse number to users. Example: `SRC-QURAN-2-255` actually points to standard 2:248, and `https://quran.com/2/255` takes the user to the wrong verse. The Arabic text and English translation are still correct — only the reference number is wrong.
  * **Fix scoped to Phase 8**: re-download from alquran.cloud using the `/surah/{n}` endpoint (which returns both numberings), store both in metadata, regenerate chunk IDs with standard numbering, and re-embed all 6,236 Quran chunks.
  * **Workaround for Phase 3**: `scripts/test_recall.py` uses the DB `ayah_num` values directly (verified by text matching) for recall auditing. The recall check correctly identifies whether a verse was retrieved, even though the displayed verse numbers are wrong.

#### 3. Impacted Files
* [DATA_SOURCES.md](file:///home/z/my-project/repos/nur/docs/DATA_SOURCES.md) — Added section 9 issue #6 documenting the global ayah numbering bug, with the full mapping table, root cause, impact, fix plan, and workaround.
* [test_recall.py](file:///home/z/my-project/repos/nur/scripts/test_recall.py) — RECALL_CHECKS updated with the 11 authoritative prayer verses using verified DB ayah_num values.
* [PHASES.md](file:///home/z/my-project/repos/nur/docs/PHASES.md) — Phase 8 evaluation task added: re-index Quran chunks with standard surah:ayah numbering.

#### 4. Validation
* **Authoritative source verified**: alquran.cloud search API returned 35 verses for "establish prayer" in en.sahih — the same edition our DB uses. This is the canonical list.
* **DB mapping verified by text content**: for each of the 11 curated verses, the DB chunk whose `text_en` matches the API verse's text was found. The mapping is unambiguous.
* **Offset pattern confirmed**: the offset between standard `numberInSurah` and DB `ayah_num` increases monotonically with each surah (7, 493, 669, 789, 954, 1160, 1235), proving global cumulative numbering.
* **User instruction honored**: the agent did NOT trust its own memory or the DB to identify prayer verses. It fetched the authoritative list from the source API and verified the mapping empirically.

---

### [2026-06-22T05:00:00-04:00] — A+B Recall Improvements: Bigger Pool + Quranic Terminology Prompt
* **Decision ID:** `DEC-031`
* **Status:** Completed (code complete; awaiting user re-test)
* **Author:** Antigravity (AI Peer Engineer)

#### 1. Context & Motivation
The recall baseline (DEC-030) was **3/14 = 21%** — only 3 of 14 authoritative prayer verses found in the top-100. Root cause analysis: the retriever matches by semantic similarity, and "Is prayer obligatory" does NOT match "establish prayer" well (different surface forms). The top-10 was full of tafsirs that explicitly say "prière obligatoire" in their bilingual context prefix, but the actual Quran verses (2:3, 2:43, 2:110, etc.) that say "establish prayer" were ranked below rank 100 or not found at all.

The user approved two improvements (A+B):
- **A**: Increase the retriever pool size from 30 to 100 — more chunks retrieved = better chance of finding direct verse proofs.
- **B**: Improve the Architect system prompt to generate sub-questions that use Quranic terminology ("establish prayer", "five pillars of Islam", "give zakah") instead of just "prayer obligation" — this helps the retriever match the actual Quranic phrasing.

#### 2. Before vs. After
* **Before:**
  * `top_k_initial = 30` (config default) — only 30 chunks retrieved per (query, source) pair.
  * Architect system prompt had 6 rules about decomposition but NO guidance on using Quranic terminology.
  * Sub-questions for "Is prayer obligatory?" were all generic: "Is prayer obligatory in Islam?", "What are the conditions for prayer to be obligatory?", etc. None used the phrase "establish prayer".
* **After:**
  * `top_k_initial = 100` (config + .env + .env.example updated). The `_retrieve()` method now uses the `top_k` parameter consistently (was hardcoded to 30 in 3 places).
  * Architect system prompt gained 3 new rules (7, 8, 9) under a "CRITICAL — USE QURANIC TERMINOLOGY" section:
    - Rule 7: For obligations (prayer, charity, fasting, pilgrimage), include a sub-question with the EXACT Quranic phrasing ("establish prayer", "give zakah", "fasting prescribed", "pilgrimage to the House").
    - Rule 8: For rulings (halal/haram), use "Quranic verse about" or "hadith about" + topic.
    - Rule 9: For five pillars questions, use the exact phrase "five pillars of Islam" (matches Bukhari/Muslim hadith).
  * The prompt now explains WHY: "the retriever matches by semantic similarity. 'Is prayer obligatory' does NOT match 'establish prayer' well, but 'establish prayer in Quran' matches it directly."

#### 3. Impacted Files
* [config.py](file:///home/z/my-project/repos/nur/src/nur/config.py) — `top_k_initial` default changed from 30 to 100, with comment explaining the DEC-031 rationale.
* [.env.example](file:///home/z/my-project/repos/nur/.env.example) — `NUR_TOP_K_INITIAL=100` with explanatory comment.
* [generator/__init__.py](file:///home/z/my-project/repos/nur/src/nur/generator/__init__.py) — Architect `SYSTEM_PROMPT` gained rules 7-9 (Quranic terminology guidance).
* [pipeline.py](file:///home/z/my-project/repos/nur/src/nur/pipeline.py) — `_retrieve()` method now uses `top_k` parameter consistently (was hardcoded to 30 in 3 places: dense search, sparse search, RRF fusion).
* [PHASES.md](file:///home/z/my-project/repos/nur/docs/PHASES.md) — Phase 3 checklist updated: recall audit done (21% baseline), A+B done, reranker next.

#### 4. Validation
* **Agent-side validation (passed):**
  * Config loads with `top_k_initial=100` (verified after .env update).
  * Architect prompt contains "QURANIC TERMINOLOGY", "establish prayer", "five pillars" — verified via string search.
  * Pipeline `_retrieve()` uses `top_k` parameter in all 3 places (dense, sparse, RRF) — no more hardcoded 30.
* **Live recall re-test (DEFERRED TO USER per Rule 3):**
  * The user MUST re-run `python3 scripts/test_recall.py` to measure the new recall.
  * Expected: recall should improve from 21% (3/14) to ideally 50%+ (7/14).
  * If the Architect now generates sub-questions like "establish prayer in Quran" and "five pillars of Islam hadith", the retriever should find 2:43, 2:110, Bukhari #8, etc. that were previously missed.
  * If recall is still low (<40%), we need Option C (query expansion with paraphrases) or Option D (ColBERT late-interaction).

---

## Future Architectural Plans

### [Phase 2] — LLM-Synthesized Contextual Retrieval via Kaggle GPUs
* **Goal:** Run `kaggle_context_synthesizer.py` on Kaggle with the AWQ model, single-GPU configuration, and force-uninstalled FlashInfer backend to generate and index enriched bilingual contexts.
* **Rationale:** Maximizes retrieval scores across all 3 target languages while maintaining zero local CPU runtime cost.





