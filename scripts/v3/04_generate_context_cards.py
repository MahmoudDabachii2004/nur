"""
NUR V3 — Step 4: Generate Context Cards via vLLM on Lightning AI (L40S)

⚠️ RUN ON LIGHTNING AI L40S INSTANCE — NOT local.
   (Same setup as scripts/kaggle_context_synthesizer.py from Phase 1.)

Generates a multilingual Context Card (Layer 1 of V3 chunk) for each of the
6,236 Quran ayahs. The card provides cross-lingual semantic bridge so
"prière" FR matches "salah" AR via shared keywords.

Model: Qwen2.5-14B-Instruct-AWQ via vLLM (tensor parallelism on L40S)
Input:  data/quran/*.json + data/tafsir/v3/ibn_kathir_en/*.json
Output: data/processed/context_cards.jsonl (1 line per ayah)

Each output line:
  {"surah": 2, "ayah": 195, "context_card": {"fr": {...}, "en": {...}, "ar": {...}}}

Context Card schema:
  {
    "fr": {"theme": "...", "rule": "...", "keywords": [...]},
    "en": {"topic": "...", "rule": "...", "keywords": [...]},
    "ar": {"theme": "...", "keywords": [...]}
  }

Strict rules (docs/v3/02_CHUNK_SCHEMA.md):
  - keywords = INTERSECTION of terms in verse + translation + Ibn Kathir first paragraph
  - NEVER extrapolated
  - theme ≤ 5 words, rule ≤ 15 words
  - At least 1 AR + 1 EN + 1 FR keyword

Usage on Lightning AI:
  # 1. Upload the repo (or git clone) to the Lightning AI studio
  # 2. Open a terminal in the L40S studio
  # 3. Install deps:
  pip install vllm FlagEmbedding chromadb pyarrow
  # 4. Run:
  python scripts/v3/04_generate_context_cards.py
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

# When run on Lightning AI, the repo root is the cwd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
QURAN_DIR = DATA_DIR / "quran"
TAFSIR_DIR = DATA_DIR / "tafsir" / "v3"
OUTPUT_PATH = DATA_DIR / "processed" / "context_cards.jsonl"

# Qwen2.5-14B-Instruct-AWQ — same model as Phase 1
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.90

SYSTEM_PROMPT = """You are an expert Islamic theologian and database indexer.
Your task is to generate a multilingual Context Card for a Quran verse, to be used
as cross-lingual semantic bridge in a RAG system.

Output STRICT JSON ONLY (no markdown, no explanation):
{
  "fr": {"theme": "...", "rule": "...", "keywords": [...]},
  "en": {"topic": "...", "rule": "...", "keywords": [...]},
  "ar": {"theme": "...", "keywords": [...]}
}

STRICT RULES:
1. "keywords" MUST be terms actually present in the verse (AR), the English
   translation, or the Ibn Kathir excerpt. NEVER invent keywords.
2. "keywords" must include AT LEAST 1 Arabic term + 1 English term + 1 French term
   (the French term may be a translation of an Arabic/English keyword).
3. "theme" / "topic" / "الموضوع" must be ≤ 5 words.
4. "rule" (FR/EN only) must be ≤ 15 words. Empty string "" if verse is purely
   narrative (no ruling).
5. Output ONLY the JSON object."""


def load_quran_data() -> tuple[list[dict], dict[tuple[int, int], str]]:
    """Load Arabic + English Quran. Returns (list_of_ayah_records, en_lookup)."""
    ar_path = QURAN_DIR / "quran-uthmani.json"
    en_path = QURAN_DIR / "en.sahih.json"

    if not ar_path.exists():
        raise FileNotFoundError(f"Run 01_download_quran.py first. Missing: {ar_path}")

    with ar_path.open("r", encoding="utf-8") as f:
        ar_data = json.load(f)

    en_lookup: dict[tuple[int, int], str] = {}
    if en_path.exists():
        with en_path.open("r", encoding="utf-8") as f:
            en_data = json.load(f)
        for surah in en_data.get("surahs", []):
            for ayah in surah.get("ayahs", []):
                en_lookup[(surah["number"], ayah["number"])] = ayah.get("text", "")

    records = []
    for surah in ar_data.get("surahs", []):
        surah_num = surah["number"]
        surah_name_ar = surah.get("name", "")
        surah_name_en = surah.get("englishName", "")
        revelation_type = surah.get("revelationType", "")
        for ayah in surah.get("ayahs", []):
            ayah_num = ayah["numberInSurah"]
            text_ar = ayah.get("text", "").lstrip("\ufeff").strip()
            text_en = en_lookup.get((surah_num, ayah_num), "")
            records.append({
                "surah": surah_num,
                "ayah": ayah_num,
                "surah_name_ar": surah_name_ar,
                "surah_name_en": surah_name_en,
                "revelation_type": revelation_type,
                "text_ar": text_ar,
                "text_en": text_en,
            })
    return records, en_lookup


def load_tafsir_excerpts() -> dict[tuple[int, int], str]:
    """Load Ibn Kathir EN tafsir, first 1500 chars per ayah."""
    tafsir_dir = TAFSIR_DIR / "ibn_kathir_en"
    excerpts: dict[tuple[int, int], str] = {}

    if not tafsir_dir.exists():
        print(f"  [WARN] Tafsir dir not found: {tafsir_dir}")
        return excerpts

    for surah_file in sorted(tafsir_dir.glob("*.json")):
        try:
            surah_num = int(surah_file.stem)
        except ValueError:
            continue
        with surah_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            ayah = entry.get("ayah")
            text = str(entry.get("text", "")).strip()
            if isinstance(ayah, int) and text:
                excerpts[(surah_num, ayah)] = text[:1500]
    return excerpts


def build_qwen_prompt(record: dict, tafsir_excerpt: str) -> str:
    """Build Qwen chat template prompt."""
    user_msg = (
        f"Surah {record['surah']}: {record['surah_name_en']} ({record['surah_name_ar']})\n"
        f"Ayah: {record['ayah']}\n"
        f"Revelation: {record['revelation_type']}\n\n"
        f"VERSE TEXT (Arabic Uthmani):\n{record['text_ar']}\n\n"
        f"VERSE TEXT (English - Saheeh International):\n{record['text_en'] or '(not available)'}\n\n"
        f"IBN KATHIR TAFSIR (first 1500 chars):\n{tafsir_excerpt or '(no tafsir available for this ayah)'}\n\n"
        f"Output the Context Card JSON now."
    )
    # Qwen chat template
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def parse_card_response(text: str) -> dict | None:
    """Parse the LLM output as JSON, tolerant of surrounding text and truncated output."""
    text = text.strip()
    # Strip trailing stop tokens
    for stop in ("<|im_end|>", "<|im_start|>"):
        if text.endswith(stop):
            text = text[: -len(stop)].strip()
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        return None
    if end == -1 or end <= start:
        # JSON truncated before closing brace — try to repair
        return _repair_truncated_json(text[start:])
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        # Try repair
        return _repair_truncated_json(text[start:])


def _repair_truncated_json(text: str) -> dict | None:
    """Attempt to repair truncated JSON by closing open brackets/braces.

    Common failure: max_tokens hit mid-keyword, JSON looks like:
      {"fr": {"theme": "...", "keywords": ["الذي
    We try to close: ["الذي"]}]} }
    """
    if not text.startswith("{"):
        return None

    # Track open brackets/braces
    open_stack = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            open_stack.append(ch)
        elif ch == "}" and open_stack and open_stack[-1] == "{":
            open_stack.pop()
        elif ch == "]" and open_stack and open_stack[-1] == "[":
            open_stack.pop()

    # If we're inside a string, close it
    repaired = text
    if in_string:
        repaired += '"'
    # Close any open arrays/objects
    for opener in reversed(open_stack):
        if opener == "[":
            repaired += "]"
        elif opener == "{":
            repaired += "}"

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def validate_card(card: dict) -> bool:
    if not isinstance(card, dict):
        return False
    for lang in ("fr", "en", "ar"):
        if lang not in card or not isinstance(card[lang], dict):
            return False
        kw = card[lang].get("keywords")
        if not isinstance(kw, list) or len(kw) == 0:
            return False
    if not card["fr"].get("theme") or not card["en"].get("topic"):
        return False
    return True


def load_completed() -> dict[tuple[int, int], dict]:
    """Load existing context_cards.jsonl for resume."""
    completed = {}
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    completed[(obj["surah"], obj["ayah"])] = obj
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def main() -> int:
    print("=" * 60)
    print("NUR V3 — Step 4: Generate Context Cards (Lightning AI L40S)")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}\n")

    # 1. Load data
    print("[1/4] Loading Quran data...")
    records, _ = load_quran_data()
    print(f"  {len(records):,} ayahs loaded")

    print("[2/4] Loading Ibn Kathir tafsir excerpts...")
    tafsir_excerpts = load_tafsir_excerpts()
    print(f"  {len(tafsir_excerpts):,} tafsir excerpts loaded")

    print("[3/4] Checking resume state...")
    completed = load_completed()
    print(f"  {len(completed):,} ayahs already have context cards")

    pending = [r for r in records if (r["surah"], r["ayah"]) not in completed]
    print(f"  {len(pending):,} ayahs to process\n")

    if not pending:
        print("All context cards already generated. Nothing to do.")
        return 0

    # 2. Setup vLLM
    print("[4/4] Initializing vLLM engine...")
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[FATAL] No GPU detected. Run on Lightning AI L40S.")
        return 1
    # Cap at 2 GPUs (L40S studios usually have 1-2)
    tensor_parallel = min(num_gpus, 2)
    print(f"  Using {tensor_parallel} GPU(s) via tensor parallelism")

    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=tensor_parallel,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
    )

    try:
        tokenizer = llm.get_tokenizer()
        if hasattr(tokenizer, "model_max_length"):
            tokenizer.model_max_length = MAX_MODEL_LEN
    except Exception as e:
        print(f"  Warning: could not override tokenizer length: {e}")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=800,  # bumped from 400 — fixes JSON truncation on keyword-rich AR verses
        stop=["<|im_end|>", "<|im_start|>"],
    )

    # 3. Build prompts in batches (avoid OOM on huge prompt lists)
    BATCH_SIZE = 256
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    total = len(pending)

    with OUTPUT_PATH.open("a", encoding="utf-8") as out_f:
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch = pending[batch_start:batch_end]

            prompts = [
                build_qwen_prompt(r, tafsir_excerpts.get((r["surah"], r["ayah"]), ""))
                for r in batch
            ]

            outputs = llm.generate(prompts, sampling_params)

            for record, out in zip(batch, outputs):
                generated = out.outputs[0].text.strip()
                card = parse_card_response(generated)

                if card and validate_card(card):
                    record_out = {
                        "surah": record["surah"],
                        "ayah": record["ayah"],
                        "context_card": card,
                    }
                    out_f.write(json.dumps(record_out, ensure_ascii=False) + "\n")
                    out_f.flush()
                    success += 1
                else:
                    failed += 1
                    print(f"  [FAIL] {record['surah']}:{record['ayah']} — output: {generated[:120]!r}")

            elapsed = batch_end
            pct = elapsed / total * 100
            print(f"  [{elapsed:,}/{total:,}] {pct:.1f}% — success={success:,}, failed={failed:,}")

    # Cleanup vLLM
    print("\nCleaning up vLLM...")
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    print("\n" + "=" * 60)
    print("CONTEXT CARD GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Success: {success:,}")
    print(f"  Failed:  {failed:,}")
    print(f"  Total:   {success + failed:,}")
    print(f"  Output:  {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"\nNext step: python scripts/v3/05_build_chunks.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
