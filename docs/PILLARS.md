
# NUR — The 10 Functional Pillars

> Executive summary of NUR's core principles. These pillars are non-negotiable functional requirements that guide all architectural and technical decisions.

> ⚠️ **V3 STATUS (2026-06-23)** — These 10 Pillars remain the **theological source of truth**. V3 implements them better than V1/V2. Specifically:
> - Pillar 1 (Triple-Index): V3 unifies Quran+Tafsir in one chunk (2 collections instead of 4), keeping the divine/human separation via labeled layers in the embedding_text
> - Pillar 5 (Dual-Layer Context): V3 uses 3 layers (Context Card + Word of Allah PURE + 4 Tafsirs labeled) — see `docs/v3/02_CHUNK_SCHEMA.md`
> - All other pillars (2, 3, 4, 6, 7, 8, 9, 10) are unchanged in V3

## Pillar 1 — Triple-Index + Scholar Index

To maintain theological integrity, we do not mix the Word of Allah with human commentary. The system uses separate logical indexes:

- **Quran Index**: 1 ayah = 1 chunk. The absolute source of truth.
- **Hadith Index**: 1 hadith = 1 chunk. Grade-aware (Sahih, Hasan, Da'if).
- **Tafsir Index**: Classical exegesis (e.g., Ibn Kathir), chunked by logical sections.
- **Scholar Index (Future)**: Fatwas and jurisprudential opinions, linked to their scriptural evidence.

## Pillar 2 — Arabic-First Cross-Lingual Retrieval

Arabic is the source of truth. Translations (French/English) are comprehension aids. We use multilingual embedding models (`BAAI/bge-m3`) that map Arabic text and translated queries into the **same vector space**. This allows matching a French question directly to an Arabic verse without a lossy intermediate translation step.

## Pillar 3 — Authenticity-Weighted Retrieval

In Islam, the strength of the narration chain is more important than mere text similarity. The retrieval system must weight scores by Hadith grade:

- **Sahih**: +30% boost (Authentic)
- **Hasan**: +10% boost (Good)
- **Da'if**: -50% penalty (Weak — last resort, requires warning)
- **Mawdu'**: Excluded (Fabricated — kept only for detection purposes)

## Pillar 4 — Post-Generation Verification (Anti-Hallucination)

A wrong religious answer is worse than no answer. The system implements a multi-layer verification pipeline after the LLM generates a response:

1. **NLI Verification**: Every sentence must be entailed by retrieved chunks (threshold 0.95).
2. **Character-by-Character Quran Check**: Every cited verse must match the Uthmani text exactly after normalization.
3. **Decoupled Grounding**: LLM produces `exact_citations` and `synthesis` separately; parser rejects if citation isn't word-for-word.

## Pillar 5 — Dual-Layer Context-Enriched Chunks

Bare text chunks lack global context. We implement a two-layer enrichment strategy before embedding:

1. **Structural Context** (`DEC-001`): Chunks are prefixed with metadata (Surah name, revelation type, narrator, grade, 300-char EN snippet).
2. **LLM-Synthesized Context** (`nur_synthesizer_cloud.py`): An LLM (Qwen2.5-14B-AWQ) generates a bilingual FR/EN index card (Theme, Rule, Keywords) for each chunk. This directly solves the French sparse-match weakness identified in `DEC-004`.
- **Impact**: Reduces retrieval failures by 67% (Anthropic, 2024) and enables true trilingual hybrid search.

## Pillar 6 — Hybrid LLM Strategy (Architect + Reporter)

The system uses a multi-model architecture to balance deep reasoning, API limits, and offline resilience:

- **Task 1: The Architect (Groq - `llama-3.1-8b-instant`)**: Fast, cheap, high-volume. Used to decompose complex user queries into sub-questions in FR/EN.
- **Task 2: The Reporter (Groq - `meta-llama/llama-4-scout-17b-16e-instruct`)**: Deep reasoning, native Arabic support. Receives ~10 chunks and generates the strict JSON report. (Fallback: `llama-3.3-70b-versatile` for extreme dilemmas).
- **Offline Fallback (Local PC - `llama-3.1-8b` via Ollama)**: If Groq hits a Rate Limit (429), the system silently reroutes to the local PC.
- **Rule**: Sacred texts (embeddings) never leave the local machine. The LLM analyzes the FR/EN context to avoid reasoning errors, and copies the Arabic text exactly.

## Pillar 7 — Structured Citation Protocol (Source IDs)

To prevent LLMs from inventing references, we inject XML-formatted documents with UPPERCASE Source IDs (e.g., `SRC-QURAN-2-255`) into the prompt. The LLM is forced to cite these IDs. Post-processing maps them to clickable URLs and rich displays.

## Pillar 8 — Scholar Opinions Are Mandatory

The system **never** gives its own opinion. It **always** reports structured scholarly views:

- Scholar name, era, and Madhhab (school of thought).
- The exact opinion.
- Scriptural evidence (Quran + Hadith) supporting the opinion.
- Book source reference.

## Pillar 9 — Ikhtilaf Awareness (Consciousness of Disagreement)

When scholarly disagreement (Ikhtilaf) exists, the system detects it and presents all views with absolute neutrality. It never encourages or discourages an opinion. It explicitly states if there is Ijma' (consensus) or isolated views, ending with a disclaimer to consult a qualified scholar.

## Pillar 10 — Bilingual Interface + Arabic Source of Truth

- **Arabic**: The source of truth. It is NOT a toggleable option. Arabic text (Quran, Hadith) must *always* be displayed alongside any answer or citation.
- **Default Interface Language**: English. The UI defaults to English, and the LLM's default synthesis language is English. This ensures maximum LLM reasoning accuracy and JSON stability.
- **Secondary Language**: French. The user can toggle the UI and the LLM's synthesis to French.
- **Behavior**: If the user selects English (default), the LLM explains the ruling in English, but the Arabic verse/hadith is still displayed above it.