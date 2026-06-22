"""
generator.py

This file implements the two-LLM "Smart Archivist" generation pipeline for NUR,
following the architecture defined in docs/RAG_PIPELINE_ARCHITECTURE.md.

WHY THIS EXISTS:
  A user asking a complex Islamic question (e.g. "I committed adultery, repented,
  must I tell my wife?") cannot be answered by a single LLM call safely. The LLM
  might hallucinate, miss a relevant source, or invent a ruling. We split the
  work into two specialized roles:

  1. THE ARCHITECT (Step 1): A fast, cheap LLM (llama-3.1-8b-instant on Groq)
     that takes the raw user query and decomposes it into 1..N sub-questions,
     each targeting a distinct jurisprudential theme. Simple questions get 1
     sub-question; complex dilemmas get 3-4. This allows the retrieval pipeline
     to fetch sources for each angle separately, preserving nuance.

  2. THE REPORTER (Step 4): A deep-reasoning LLM (meta-llama/llama-4-scout-17b-
     16e-instruct on Groq, 30K TPM) that receives the Top-10 retrieved chunks
     and produces a STRICT JSON report with three sections:
       - conflict_detection: identifies scholarly disagreement (Ikhtilaf)
       - direct_reports: factual summaries of each source, no external logic
       - synthesis: the final answer, citing the source IDs

  The Reporter is FORCED into the archivist persona via the `instructor` library
  and Pydantic schema validation. It is forbidden from using its parametric
  knowledge — it can only report what the retrieved sources say.

FALLBACK CHAIN (docs/RAG_PIPELINE_ARCHITECTURE.md Section 3):
  1. Primary: Llama 4 Scout 17B (Groq, 30K TPM)
  2. Extreme Ikhtilaf fallback: Llama 3.3 70B (Groq, 12K TPM) — only if the
     Scout model struggles with a complex theological conflict. Triggered
     manually by passing force_reasoning=True to Reporter.generate().
  3. Offline fallback: Llama 3.1 8B via local Ollama — only if Groq returns
     429 Rate Limit Exceeded. Implemented in a future commit (needs the
     `ollama` Python client to be wired in).

CRITICAL RULES (docs/PILLARS.md):
  - Temperature MUST be 0.0 (no hallucination tolerance for sacred text).
  - Frequency penalty MUST be 0.0 (non-zero corrupts Arabic formulas).
  - The LLM never invents Source IDs — it can only cite IDs that were injected
    into its prompt via the <document id="S1"> XML blocks (Pillar 7).
"""

from __future__ import annotations

import json
from typing import Literal

import instructor
from groq import Groq
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.nur.config import settings


# ============================================================
# Pydantic Models — these define the strict JSON schemas
# the LLM is forced to fill via `instructor`.
# ============================================================


class SubQuestions(BaseModel):
    """Output schema for the Architect (Step 1).

    The LLM decomposes a complex user query into 1..N sub-questions, each
    targeting a distinct jurisprudential theme. The number is dynamic —
    simple questions get 1 sub-question, complex dilemmas get 3-4.
    """

    sub_questions: list[str] = Field(
        ...,
        description=(
            "An array of 1 to N sub-questions derived from the user's original "
            "query. Each sub-question targets a distinct jurisprudential theme. "
            "For simple questions (e.g. 'How many rakats for Fajr?'), return "
            "exactly 1 sub-question. For complex dilemmas, return 3-4."
        ),
        min_length=1,
        max_length=6,
    )


class DirectReport(BaseModel):
    """A single source report inside the Reporter's output (Step 4).

    The LLM fills one of these per retrieved source. The 'report' field is the
    critical anti-hallucination layer — it must summarize ONLY what that specific
    source states, with no external logic or cross-source inference.
    """

    source_id: str = Field(
        ...,
        description=(
            "The source ID as injected in the prompt, e.g. 'S1', 'S2', etc. "
            "Must match one of the <document id=\"...\"> blocks provided."
        ),
    )
    source_type: Literal["quran", "hadith", "tafsir_ar", "tafsir_en", "scholar"] = Field(
        ...,
        description="The type of source this report summarizes.",
    )
    arabic_text: str = Field(
        ...,
        description=(
            "The exact Arabic text from the chunk, copied character-for-character. "
            "For non-Arabic sources (e.g. Tafsir EN), use the original text. "
            "DO NOT paraphrase, translate, or modify in any way."
        ),
    )
    report: str = Field(
        ...,
        description=(
            "A factual summary of ONLY what this specific source states. "
            "No external logic. No cross-source inference. No opinions. "
            "If the source is silent on a sub-topic, say so explicitly."
        ),
    )


class ReporterOutput(BaseModel):
    """Output schema for the Reporter (Step 4).

    The strict JSON structure the LLM must fill. This is the anti-hallucination
    enforcement layer (Pillar 4 — Post-Generation Verification):
      - direct_reports forces the LLM to state what each source says BEFORE
        synthesizing, preventing it from inventing conclusions first and
        cherry-picking sources to justify them.
      - synthesis must cite source IDs that exist in direct_reports.
        (Phase 4 will add a parser to verify this.)
    """

    conflict_detection: str = Field(
        ...,
        description=(
            "Does the provided context present a dilemma or scholarly "
            "disagreement (Ikhtilaf)? Answer in one or two sentences. If there "
            "is no conflict, say 'No conflict detected — sources are consistent.'"
        ),
    )
    direct_reports: list[DirectReport] = Field(
        ...,
        description=(
            "A list of factual reports, one per retrieved source. Each report "
            "summarizes ONLY what that specific source states — no external "
            "logic, no cross-source inference."
        ),
        min_length=1,
    )
    synthesis: str = Field(
        ...,
        description=(
            "The final answer assembling the reports. MUST explicitly cite "
            "source IDs (e.g. [S1], [S3]). MUST NOT invent logical links not "
            "present in the reports. MUST NOT use any knowledge outside the "
            "provided context."
        ),
    )


# ============================================================
# Architect — Step 1: Query Decomposition
# ============================================================


class Architect:
    """Decomposes a user query into 1..N sub-questions (Step 1).

    Uses the fast llama-3.1-8b-instant model on Groq. The output is a list of
    sub-questions, each targeting a distinct jurisprudential theme. This allows
    the retrieval pipeline to fetch sources for each angle separately,
    preserving nuance that a single vector search would lose.
    """

    SYSTEM_PROMPT = (
        "You are the Architect of the NUR Islamic RAG system. Your job is to "
        "decompose a user's question into 1 to N sub-questions, each targeting "
        "a distinct jurisprudential theme.\n\n"
        "RULES:\n"
        "1. For simple factual questions (e.g. 'How many rakats for Fajr?'), "
        "return exactly 1 sub-question — the original question rephrased for "
        "search clarity.\n"
        "2. For complex dilemmas (e.g. 'I committed adultery, repented, must I "
        "tell my wife?'), return 2-4 sub-questions, each isolating one "
        "jurisprudential angle (the rule itself, the conditions, the scholarly "
        "opinion on the specific case).\n"
        "3. Sub-questions MUST be in the same language as the original query.\n"
        "4. Sub-questions MUST be self-contained — a search engine should be "
        "able to find relevant Islamic sources for each one independently.\n"
        "5. NEVER exceed 6 sub-questions.\n"
        "6. NEVER answer the question yourself — your only job is decomposition.\n\n"
        "CRITICAL — USE QURANIC TERMINOLOGY:\n"
        "7. When the question is about an Islamic obligation (prayer, charity, "
        "fasting, pilgrimage, etc.), include at least one sub-question that uses "
        "the EXACT Quranic phrasing. For example:\n"
        "   - For prayer: use 'establish prayer' (not just 'prayer obligation')\n"
        "   - For charity: use 'give zakah' or 'establish prayer and give zakah'\n"
        "   - For fasting: use 'fasting prescribed' or 'fast as prescribed'\n"
        "   - For pilgrimage: use 'pilgrimage to the House' or 'Hajj obligation'\n"
        "8. When the question is about a ruling (halal/haram), include a "
        "sub-question that uses the phrase 'Quranic verse about' or 'hadith "
        "about' followed by the topic — this matches how Islamic texts are "
        "indexed.\n"
        "9. When the question is about the five pillars of Islam, include a "
        "sub-question with the exact phrase 'five pillars of Islam' — this "
        "matches the famous Bukhari/Muslim hadith.\n"
        "These Quranic-terminology sub-questions are essential because the "
        "retriever matches by semantic similarity. 'Is prayer obligatory' does "
        "NOT match 'establish prayer' well, but 'establish prayer in Quran' "
        "matches it directly."
    )

    def __init__(self, client: Groq | None = None) -> None:
        """Initialize the Architect with a Groq client.

        Uses `instructor.Mode.TOOLS` (function calling) instead of
        `Mode.JSON_SCHEMA` because `llama-3.1-8b-instant` does NOT support
        the `json_schema` response format (Groq returns HTTP 400 with
        "This model does not support response format json_schema"). Function
        calling is universally supported across Groq chat models, and the
        Architect's schema is simple enough (a flat list of strings) that
        tool-call enforcement is more than sufficient.

        Args:
            client: A Groq client instance. If None, a new one is created from
                    the GROQ_API_KEY in settings.
        """
        if client is None:
            client = Groq(api_key=settings.groq_api_key)
        # Mode.TOOLS = function calling. Universally supported on Groq chat models.
        # Mode.JSON_SCHEMA would be stricter but is not supported on llama-3.1-8b-instant.
        self.client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)
        self.model = settings.llm_architect

    def decompose(self, user_query: str) -> list[str]:
        """Decompose a user query into 1..N sub-questions.

        Args:
            user_query: The raw user question, in any language (EN/FR/AR).

        Returns:
            A list of 1..6 sub-questions in the same language as the input.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            response_model=SubQuestions,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        # instructor returns the validated Pydantic model directly
        return response.sub_questions


# ============================================================
# Reporter — Step 4: Structured Generation
# ============================================================


class Reporter:
    """Generates the final structured JSON report from retrieved chunks (Step 4).

    Uses the deep-reasoning Llama 4 Scout 17B model on Groq as primary, with
    Llama 3.3 70B as an extreme-Ikhtilaf fallback. The LLM is forced into the
    'Strict Archivist' persona — it can only report what the retrieved sources
    say, never invent logical links or use parametric knowledge.
    """

    SYSTEM_PROMPT = (
        "You are the Reporter of the NUR Islamic RAG system. You act as a "
        "Strict Archivist — NOT a Mufti, NOT a conversational agent.\n\n"
        "ABSOLUTE RULES:\n"
        "1. NEVER use your pre-trained parametric knowledge to answer. You can "
        "ONLY report what the retrieved <document> blocks state.\n"
        "2. NEVER invent logical links between concepts. If a source does not "
        "explicitly connect two ideas, you MUST NOT connect them.\n"
        "3. For each <document>, write a 'direct_report' that summarizes ONLY "
        "what THAT specific source states. No external logic. No opinions.\n"
        "4. Copy the Arabic text from the chunk EXACTLY into the 'arabic_text' "
        "field — character-for-character. Do NOT paraphrase, translate, or "
        "modify diacritics.\n"
        "5. In the 'synthesis' field, cite source IDs explicitly (e.g. [S1], "
        "[S3]). The IDs MUST match the <document id=\"...\"> values provided.\n"
        "6. If the sources disagree, state the disagreement neutrally in "
        "'conflict_detection'. Do NOT take sides.\n"
        "7. If the sources are insufficient to answer, say so in 'synthesis'. "
        "Do NOT fabricate an answer.\n"
        "8. The synthesis language defaults to English. If the user query is "
        "in French, write the synthesis in French. Arabic text is ALWAYS "
        "preserved verbatim regardless of synthesis language."
    )

    def __init__(self, client: Groq | None = None) -> None:
        """Initialize the Reporter with a Groq client.

        Args:
            client: A Groq client instance. If None, a new one is created from
                    the GROQ_API_KEY in settings.
        """
        if client is None:
            client = Groq(api_key=settings.groq_api_key)
        self.client = instructor.from_groq(client, mode=instructor.Mode.JSON_SCHEMA)
        self.raw_groq_client = client  # kept for the 429 fallback logic
        self.model_primary = settings.llm_primary  # Llama 4 Scout
        self.model_reasoning = settings.llm_reasoning  # Llama 3.3 70B

    def _build_user_prompt(self, user_query: str, sources_xml: str) -> str:
        """Build the user message containing the query and the retrieved sources.

        Args:
            user_query: The original user question.
            sources_xml: The retrieved chunks formatted as <document> XML blocks.
                         Typically produced by render_sources_for_prompt() in
                         src/nur/sources.py.

        Returns:
            The complete user message string.
        """
        return (
            f"USER QUESTION:\n{user_query}\n\n"
            f"RETRIEVED SOURCES:\n{sources_xml}\n\n"
            f"Based ONLY on the sources above, produce the structured report."
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        # Note: we retry on ANY exception so transient 429s and 5xx get retried.
        # Phase 3 will add the Ollama offline fallback after retries are exhausted.
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        user_query: str,
        sources_xml: str,
        force_reasoning: bool = False,
    ) -> ReporterOutput:
        """Generate the structured report from retrieved sources.

        Args:
            user_query: The original user question.
            sources_xml: Retrieved chunks formatted as <document> XML blocks.
            force_reasoning: If True, use the Llama 3.3 70B reasoning model
                             instead of the default Scout. Use this only for
                             extreme Ikhtilaf (scholarly disagreement) cases
                             where Scout struggles.

        Returns:
            A validated ReporterOutput Pydantic model. instructor guarantees
            the structure matches the schema — no manual JSON parsing needed.
        """
        model = self.model_reasoning if force_reasoning else self.model_primary
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(user_query, sources_xml)},
        ]

        return self.client.chat.completions.create(
            model=model,
            response_model=ReporterOutput,
            messages=messages,
            temperature=settings.llm_temperature,  # 0.0 — no hallucination tolerance
            max_tokens=settings.llm_max_tokens,
            frequency_penalty=settings.llm_frequency_penalty,  # 0.0 — preserves Arabic
        )


# ============================================================
# Convenience: combined facade for the pipeline.py orchestrator
# ============================================================


class Generator:
    """Convenience facade exposing both Architect and Reporter.

    This is what pipeline.py will instantiate. Keeping Architect and Reporter
    as separate classes makes them independently testable; this facade just
    groups them for ergonomic use.
    """

    def __init__(self, client: Groq | None = None) -> None:
        """Initialize both the Architect and Reporter with a shared Groq client.

        Args:
            client: A Groq client instance. If None, a new one is created.
                    Sharing one client across both models is more efficient
                    than creating two separate connection pools.
        """
        if client is None:
            client = Groq(api_key=settings.groq_api_key)
        self.architect = Architect(client=client)
        self.reporter = Reporter(client=client)

    def decompose_query(self, user_query: str) -> list[str]:
        """Step 1: Decompose the user query into sub-questions."""
        return self.architect.decompose(user_query)

    def generate_report(
        self,
        user_query: str,
        sources_xml: str,
        force_reasoning: bool = False,
    ) -> ReporterOutput:
        """Step 4: Generate the structured report from retrieved sources."""
        return self.reporter.generate(user_query, sources_xml, force_reasoning)
