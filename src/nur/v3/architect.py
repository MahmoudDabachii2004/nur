"""
NUR V3 Runtime — Step 1: Architect (query decomposition)

Sends user question to Groq llama-3.1-8b-instant, gets back 3-7 sub-questions
in FR/EN/AR that target distinct aspects of the query.

Per docs/v3/04_RETRIEVAL_PIPELINE.md Step 1.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Project root for config import
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from nur.config import settings  # noqa: E402

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
ARCHITECT_MODEL = settings.llm_architect  # llama-3.1-8b-instant

ARCHITECT_PROMPT = """You are an Islamic search query decomposer for a RAG system.

Decompose the user's question into 3-7 sub-questions that will be used to search
an Islamic database (Quran + Hadith + Tafsir).

Rules:
- Output: JSON object with "sub_questions" array of strings
- Each sub-question ≤ 10 words
- Mix FR/EN/AR keywords when relevant (the database is trilingual)
- Include synonyms and root words
- Include the modern topic AND the Quranic equivalent (e.g. "smoking" → "self-harm forbidden quran")
- Detect the user's language and include sub-questions in that language primarily

User question: {question}

Output JSON now:"""


def call_groq(prompt: str, api_key: str, max_retries: int = 3) -> dict | None:
    """Call Groq chat API. Returns parsed JSON dict or None on failure."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": ARCHITECT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"  [Architect] 429 rate limit, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            # Tolerant JSON parse
            content = content.strip()
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1:
                return None
            return json.loads(content[start : end + 1])
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"  [Architect] Attempt {attempt+1} failed: {e}")
            time.sleep(3)
    return None


def decompose_query(user_question: str) -> list[str]:
    """Decompose user question into sub-questions for retrieval.

    Returns list of 3-7 sub-questions, including the original question
    prepended (raw user question is always used as first query).
    """
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("NUR_GROQ_API_KEY") or settings.groq_api_key
    if not api_key:
        print("[Architect] No GROQ_API_KEY — returning only the raw user question")
        return [user_question]

    prompt = ARCHITECT_PROMPT.format(question=user_question)
    result = call_groq(prompt, api_key)

    if not result or "sub_questions" not in result:
        print("[Architect] LLM returned invalid response — using raw question only")
        return [user_question]

    sub_qs = result["sub_questions"]
    if not isinstance(sub_qs, list):
        return [user_question]

    # Filter: keep only strings, dedupe, cap at 7
    seen = set()
    cleaned = []
    for q in sub_qs:
        if isinstance(q, str) and q.strip() and q.strip() not in seen:
            seen.add(q.strip())
            cleaned.append(q.strip())
            if len(cleaned) >= 7:
                break

    # Always include the raw user question as the first query (preserves keywords/tone)
    if user_question not in seen:
        cleaned.insert(0, user_question)

    return cleaned


if __name__ == "__main__":
    # Quick CLI test
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Est-ce que fumer est haram ?"
    print(f"User: {q}\n")
    subs = decompose_query(q)
    print(f"Architect output ({len(subs)} sub-questions):")
    for i, s in enumerate(subs, 1):
        print(f"  {i}. {s}")
