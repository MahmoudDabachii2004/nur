Oui, il y a une modification cruciale à faire. On avait décidé ensemble (suite à l'analyse des limites Groq) de passer au modèle **Llama 4 Scout** pour le "Reporter" (Task 2) afin d'éviter de saturer la limite de 12 000 TPM du 70B, tout en gardant le 70B comme fallback pour les cas extrêmes. Il faut aussi mentionner que l'interface est en anglais par défaut.

Voici le fichier complet et mis à jour. Remplace tout le contenu de ton fichier d'architecture par ceci :

```markdown
# NUR — RAG Pipeline Architecture (The "Smart Archivist" Model)

> **MANDATORY READ**: This document defines the exact, definitive sequence of operations for the NUR Retrieval-Augmented Generation pipeline. 
> It is designed to solve complex theological dilemmas (Ikhtilaf) while strictly adhering to `PILLARS.md` (Zero hallucination, Scholar-grounded, Strict abstention).

---

## 1. Core Philosophy: The "Reporter" Persona

NUR's LLM does not act as a Mufti (scholar giving rulings) or a conversational agent improvising answers. NUR acts as a **Strict Archivist (Reporter)**.

- **Rule**: The LLM must never use its pre-trained parametric knowledge to answer a question. It must never invent logical links between concepts (e.g., it must not equate "hiding a sin" with "lying" unless a source explicitly does so).
- **Mechanism**: The LLM is physically forced via a strict JSON Schema to report *exactly* what the retrieved sources say, point by point, before synthesizing the final answer.

---

## 2. The Pipeline Steps (Sequential Execution)

### Step 1: Dynamic Query Decomposition (The Architect)
Complex theological questions (e.g., *“I committed adultery, repented, must I tell my wife?”*) cannot be searched directly in a vector database without losing nuance.

1. The user's raw question is sent to a fast, lightweight LLM (`llama-3.1-8b-instant` on Groq).
2. The LLM dynamically generates an array of sub-questions targeting the distinct jurisprudential themes of the query.
   - *Example output*: `["Rule of exposing sins in Islam", "Conditions of sincere repentance (Tawbah)", "Scholarly opinion on revealing past adultery to a spouse"]`
3. **Rule**: The number of sub-questions is dynamic (1 to N). If the question is simple (*"How many rakats for Fajr?"*), it generates only 1 sub-question.

### Step 2: Multi-Query Local Hybrid Retrieval (The Fetch)
The retrieval system runs locally on the MacBook (M4). It searches the 4 separate indexes (Quran, Hadith, Tafsir, Scholar).

1. **The Queries**: The system launches parallel hybrid searches (Dense via ChromaDB + Sparse via JSON dot-product) for:
   - The Raw User Question (to preserve exact keywords/tone).
   - Sub-question 1
   - Sub-question 2... etc.
2. **RRF Fusion**: Results are fused using Reciprocal Rank Fusion (`k=25`, `α=0.4 dense / 0.6 sparse`).
3. **Output**: The system collects roughly 30 unique chunks (deduplicated).

### Step 3: Mathematical Reranking & Abstention (The Gatekeeper)
This is the critical anti-hallucination layer. We do not trust an LLM to evaluate if it has "enough" information. We trust a **Cross-Encoder**.

1. **The Engine**: The 30 chunks are passed through `bge-reranker-v2-m3` locally.
2. **How it works (Cross-Encoder vs Bi-Encoder)**: Unlike standard embeddings that evaluate text separately, the Reranker concatenates the `[User Question]` and the `[Chunk Text]` into a single sequence. Using the Transformer attention mechanism, every word in the question "attends" to every word in the chunk. It outputs a precise relevance score (0.0 to 1.0).
3. **The Abstention Rule (Pillar 4)**: 
   - The 30 chunks are sorted by score.
   - If the Top 1 chunk score is `< 0.35`, the system **aborts generation**. It returns: *"I do not have sufficient reliable sources in my database to answer this question."*
   - If the score is valid, the system keeps only the **Top 10 chunks**.

### Step 4: Structured Generation (The Reporter)
The Top 10 chunks are sent to the primary reasoning LLM (`meta-llama/llama-4-scout-17b-16e-instruct` on Groq) using the `instructor` library for constrained JSON decoding. The default synthesis language is English (toggleable to French by the user).

The LLM is forced to fill the following JSON Schema:

```json
{
  "conflict_detection": "Boolean/string. Does the provided context present a dilemma or scholarly disagreement (Ikhtilaf)?",
  "direct_reports": [
    {
      "source_id": "[S1]",
      "source_type": "Hadith/Quran/Scholar",
      "arabic_text": "Exact Arabic text from the chunk.",
      "report": "Factual summary of ONLY what this specific source states. No external logic."
    }
  ],
  "synthesis": "The final answer assembling the reports. Must explicitly cite the source IDs. Must not invent logical links not present in the reports."
}
```

### Step 5: Post-Generation Verification
Before displaying the final JSON to the user, Python validates the output locally:
1. **Arabic Verification (Pillar 4)**: The `arabic_text` fields are normalized (diacritics stripped) and compared character-by-character against the local Uthmani Quran database. If mismatched, it is replaced by the authentic text.
2. **Citation Check**: The `synthesis` field is parsed to ensure every `[SX]` cited actually exists in the `direct_reports` array.

---

## 3. API & Hardware Strategy (Hybrid Resilience)

To respect the $0 cost constraint and Groq's free tier limits, the pipeline uses a primary/fallback architecture optimized for token throughput.

1. **Primary Brain (Groq - Llama 4 Scout 17B)**: Handles Step 4 (Generation). It provides 30,000 Tokens Per Minute (TPM), allowing the system to process the ~4,000 input tokens (context) + ~800 output tokens per request reliably. It has the deep reasoning required to summarize complex Fiqh dilemmas without hallucinating.
2. **Extreme Dilemma Fallback (Groq - Llama 3.3 70B)**: If the Scout model struggles with a highly complex theological conflict, the system can fallback to the 70B model (limited to 12,000 TPM) for maximum reasoning power.
3. **Offline Fallback Brain (Local PC - Llama 3.1 8B via Ollama)**: If Groq returns a `429 Rate Limit Exceeded` error, the Python orchestrator catches it and silently reroutes Step 4 to the local PC (RX 5700 XT). The 8B model is less nuanced but perfectly capable of acting as a "Reporter" by filling the JSON schema with the provided context.
4. **Local Infrastructure (MacBook M4)**: Always handles Step 2 (Retrieval), Step 3 (Reranking), and Step 5 (Verification) natively on Apple Silicon (MPS).

---

## 4. Example Use Case: The Marital Dilemma

**User Query**: *"I committed adultery, repented. Must I tell my wife?"*

1. **Architect**: Generates `["Rule of exposing sins in Islam", "Conditions of sincere repentance (Tawbah)", "Scholarly opinion on revealing past adultery to a spouse"]`.
2. **Fetch**: Retrieves 30 chunks from Hadith, Quran, and Scholar indexes.
3. **Reranker**: Identifies that Sahih Muslim 2990 (hiding sins) and an IslamQA fatwa (preserving the household) are highly relevant. Scores them > 0.85. Keeps Top 10.
4. **Reporter (Llama 4 Scout)**: Outputs the JSON.
   - `conflict_detection`: "Conflict between exposing a sin and preserving the household."
   - `report [S1]`: "It is forbidden to expose sins that Allah has concealed."
   - `report [S3]`: "Scholars apply the rule of repelling the greater evil to prevent the destruction of the marriage."
   - `synthesis`: *"According to the reports above, the scholarly opinion [S3] states that it is not obligatory to disclose this sin... because sincere repentance has erased it [S1] and disclosure would cause a greater evil to the household."*
5. **Verification**: Arabic text of Sahih Muslim 2990 validated against Uthmani script.

**Result**: A theologically safe, perfectly grounded, zero-hallucination answer that respects the user's complex dilemma.