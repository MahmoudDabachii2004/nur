"""
NUR V3 Runtime — Step 5: Reporter (Groq structured JSON generation)

Sends top-5 Quran chunks + top-5 Hadith chunks + auto-pulled hadiths + confidence
flags to Groq llama-4-scout-17b-instruct, gets back a structured JSON response.

Per docs/v3/06_GENERATION_VERIFICATION.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from nur.config import settings  # noqa: E402

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are NUR, an Islamic RAG assistant. You answer questions about Islam
using ONLY the retrieved sources provided below.

ABSOLUTE RULES:
1. NEVER invent a verse, hadith, or tafsir.
2. Cite sources using [S1], [S2], ... format.
3. ONLY use sources explicitly provided in <sources>.
4. For Quran citations: copy the Arabic EXACTLY as provided (we verify char-by-char).
5. For tafsir content: ALWAYS label as "Tafsir {source} says:" — never as Quran.
6. If phase_a_status = EMPTY, start with: "Le Quran n'aborde pas directement ce sujet."
7. If <ikhtilaf detected="true">: present BOTH views, mention "Ikhtilaf", suggest consulting a scholar.
8. If a tafsir has contains_isra_iliyyat="true": add disclaimer about Isra'iliyyat.
9. Answer in the user's language.
10. Arabic text of Quran MUST be displayed alongside any answer (Pillar 10).

OUTPUT FORMAT — strict JSON:
{
  "answer": "...",  (in user's language, with [Sn] citations inline)
  "citations": [
    {
      "source_id": "SRC-QURAN-2-195",
      "label": "Quran 2:195",
      "type": "quran" | "hadith",
      "arabic": "...",  (for Quran/hadith, exact text)
      "english": "...",
      "tafsir_used": "Ibn Kathir" | null,
      "url": "https://..."
    }
  ],
  "ikhtilaf": {"detected": false, "summary": null},
  "confidence": "high" | "medium" | "low",
  "phase_a_status": "STRONG" | "WEAK" | "EMPTY",
  "phase_b_status": "STRONG" | "WEAK" | "EMPTY",
  "disclaimer": "..."  (optional)
}

Do not include any text outside the JSON."""


def _render_quran_chunk_xml(chunk: dict, index: int) -> str:
    """Render a Quran chunk as XML block for the Reporter prompt."""
    meta = chunk.get("metadata", {})
    sid = chunk["id"]
    label = f"Quran {meta.get('surah', '?')}:{meta.get('ayah', '?')}"
    text_ar = meta.get("text_ar", "")
    text_en = meta.get("text_en", "")
    surah_name = meta.get("surah_name_en", "")
    revelation = meta.get("revelation_type", "")
    url = meta.get("url", "")

    # Context card (flattened by ChromaDB)
    context_fr_theme = meta.get("context_card_fr_theme", "")
    context_en_topic = meta.get("context_card_en_topic", "")
    keywords = meta.get("context_card_fr_keywords", "") or meta.get("context_card_en_keywords", "")

    # Tafsirs (ChromaDB stores as flat strings)
    tafsir_blocks = []
    for source, lang in [("Ibn Kathir", "en"), ("Ibn Kathir", "ar"),
                          ("Al-Tabari", "ar"), ("As-Sa'di", "ar")]:
        # We can't recover individual tafsirs from flat metadata easily
        # The LLM gets the full embedding_text which contains them
        pass

    return f"""<document id="S{index}">
  <source_id>{sid}</source_id>
  <type>quran</type>
  <label>{label}</label>
  <surah_name>{surah_name}</surah_name>
  <revelation>{revelation}</revelation>
  <arabic>{text_ar}</arabic>
  <english>{text_en}</english>
  <context_card>
    [FR] Thème: {context_fr_theme}
    [EN] Topic: {context_en_topic}
    Keywords: {keywords}
  </context_card>
  <embedding_text>{chunk.get('embedding_text', '')[:2000]}</embedding_text>
  <url>{url}</url>
</document>"""


def _render_hadith_chunk_xml(chunk: dict, index: int, auto_pulled: bool = False) -> str:
    meta = chunk.get("metadata", {})
    sid = chunk["id"]
    collection = meta.get("collection", "")
    hadith_num = meta.get("hadith_number", "?")
    grade = meta.get("grade", "")
    narrator = meta.get("narrator", "")
    text_ar = meta.get("text_ar", "")
    text_en = meta.get("text_en", "")
    url = meta.get("url", "")

    auto_pulled_attr = ' auto_pulled="true"' if auto_pulled else ""
    note = ""
    if auto_pulled:
        source_quran = chunk.get("auto_pulled_from", "")
        note = f"<note>Auto-pulled because cited in tafsir for {source_quran}</note>"

    return f"""<document id="S{index}"{auto_pulled_attr}>
  <source_id>{sid}</source_id>
  <type>hadith</type>
  <label>{collection} #{hadith_num}</label>
  <grade>{grade}</grade>
  <narrator>{narrator}</narrator>
  <arabic>{text_ar}</arabic>
  <english>{text_en}</english>
  {note}
  <url>{url}</url>
</document>"""


def _build_reporter_prompt(
    user_question: str,
    detected_lang: str,
    phase_a_status: str,
    phase_a_confidence: float,
    phase_b_status: str,
    phase_b_confidence: float,
    quran_chunks: list[dict],
    hadith_chunks: list[dict],
    auto_pulled_hadiths: list[dict],
) -> str:
    """Build the full Reporter prompt with system + user messages."""
    # Number sources S1..Sn in order: Quran first, then Hadith, then auto-pulled
    all_sources_xml = []
    idx = 1
    for chunk in quran_chunks:
        all_sources_xml.append(_render_quran_chunk_xml(chunk, idx))
        idx += 1
    for chunk in hadith_chunks:
        all_sources_xml.append(_render_hadith_chunk_xml(chunk, idx, auto_pulled=False))
        idx += 1
    for chunk in auto_pulled_hadiths:
        all_sources_xml.append(_render_hadith_chunk_xml(chunk, idx, auto_pulled=True))
        idx += 1

    sources_xml = "\n".join(all_sources_xml)

    user_msg = f"""<question>{user_question}</question>
<detected_language>{detected_lang}</detected_language>

<phase_status>
  <phase_a status="{phase_a_status}" confidence="{phase_a_confidence:.3f}" />
  <phase_b status="{phase_b_status}" confidence="{phase_b_confidence:.3f}" />
</phase_status>

<sources>
{sources_xml}
</sources>

Generate the JSON response now."""

    return user_msg


def call_reporter(
    user_question: str,
    detected_lang: str,
    phase_a_status: str,
    phase_a_confidence: float,
    phase_b_status: str,
    phase_b_confidence: float,
    quran_chunks: list[dict],
    hadith_chunks: list[dict],
    auto_pulled_hadiths: list[dict],
    max_retries: int = 2,
) -> dict:
    """Call Groq Reporter LLM. Returns parsed JSON dict.

    Fallback chain: llama-4-scout-17b → llama-3.3-70b-versatile → Ollama local
    """
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("NUR_GROQ_API_KEY") or settings.groq_api_key

    user_msg = _build_reporter_prompt(
        user_question, detected_lang,
        phase_a_status, phase_a_confidence,
        phase_b_status, phase_b_confidence,
        quran_chunks, hadith_chunks, auto_pulled_hadiths,
    )

    # Try Groq primary first, then Groq reasoning, then Ollama
    if api_key:
        models_to_try = [
            ("groq", settings.llm_primary),       # llama-4-scout-17b
            ("groq", settings.llm_reasoning),     # llama-3.3-70b
        ]
    else:
        models_to_try = []

    # Always include Ollama fallback
    models_to_try.append(("ollama", settings.llm_local))

    for provider, model in models_to_try:
        print(f"[Reporter] Trying {provider}:{model}")
        try:
            response_text = _call_llm(provider, model, api_key, user_msg, max_retries)
            if response_text:
                parsed = _parse_json_response(response_text)
                if parsed and _validate_schema(parsed):
                    return parsed
                print(f"[Reporter] {model} returned invalid JSON, trying next model")
        except Exception as e:
            print(f"[Reporter] {model} failed: {e}")

    # All models failed
    return {
        "answer": "I apologize, but I'm unable to generate a reliable answer at this time. Please try again or consult a qualified scholar.",
        "citations": [],
        "ikhtilaf": {"detected": False, "summary": None},
        "confidence": "low",
        "phase_a_status": phase_a_status,
        "phase_b_status": phase_b_status,
        "error": "all_models_failed",
    }


def _call_llm(provider: str, model: str, api_key: str, user_msg: str, max_retries: int) -> str | None:
    """Call a single LLM provider. Returns raw response text or None."""
    if provider == "groq":
        return _call_groq(model, api_key, user_msg, max_retries)
    elif provider == "ollama":
        return _call_ollama(model, user_msg, max_retries)
    return None


def _call_groq(model: str, api_key: str, user_msg: str, max_retries: int) -> str | None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"  [Groq] 429, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError) as e:
            print(f"  [Groq] Attempt {attempt+1}: {e}")
            time.sleep(3)
    return None


def _call_ollama(model: str, user_msg: str, max_retries: int) -> str | None:
    """Call local Ollama server (offline fallback)."""
    import requests as ollama_requests
    base_url = settings.ollama_base_url  # http://localhost:11434
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    for attempt in range(max_retries):
        try:
            resp = ollama_requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except requests.RequestException as e:
            print(f"  [Ollama] Attempt {attempt+1}: {e}")
            time.sleep(5)
    return None


def _parse_json_response(text: str) -> dict | None:
    """Tolerant JSON parse — extracts the first {...} block from text."""
    if not text:
        return None
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text[3:] else text
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _validate_schema(parsed: dict) -> bool:
    """Basic structural validation of the Reporter response."""
    if not isinstance(parsed, dict):
        return False
    required = ["answer", "citations", "ikhtilaf", "confidence"]
    for k in required:
        if k not in parsed:
            return False
    if not isinstance(parsed["citations"], list):
        return False
    if not isinstance(parsed["ikhtilaf"], dict) or "detected" not in parsed["ikhtilaf"]:
        return False
    if parsed["confidence"] not in ("high", "medium", "low"):
        return False
    return True
