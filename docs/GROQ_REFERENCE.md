# NUR — Groq API Reference (Verified Limits & Capabilities)

> **Source of truth for Groq API constraints.** This document records the
> verified rate limits and structured-output capabilities of each Groq model
> we use. It exists so we never hallucinate Groq capabilities again (see
> DEC-024 — the agent initially used `Mode.JSON_SCHEMA` on
> `llama-3.1-8b-instant` which doesn't support it, causing a 400 error).
>
> **If Groq changes their limits or model lineup**, update this file AND log
> a new Decision ID in `docs/brains.md`.

---

## 1. Rate Limits — Free Plan (verified 2026-06-22)

These are the per-organization limits on the Groq Free plan. Rate limits apply
at the organization level, not per-user. You hit whichever threshold you reach
first (RPM, RPD, TPM, or TPD).

| Model ID | RPM | RPD | TPM | TPD |
|----------|-----|-----|-----|-----|
| `llama-3.1-8b-instant` | 30 | 14,400 | 6,000 | 500,000 |
| `llama-3.3-70b-versatile` | 30 | 1,000 | 12,000 | 100,000 |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 30 | 1,000 | 30,000 | 500,000 |
| `qwen/qwen3-32b` | 60 | 1,000 | 6,000 | 500,000 |
| `qwen/qwen3.6-27b` | 30 | 1,000 | 8,000 | 200,000 |
| `openai/gpt-oss-120b` | 30 | 1,000 | 8,000 | 200,000 |
| `openai/gpt-oss-20b` | 30 | 1,000 | 8,000 | 200,000 |

**Key**: RPM = Requests Per Minute | RPD = Requests Per Day | TPM = Tokens Per Minute | TPD = Tokens Per Day

### NUR's model assignments and their effective limits

| Role | Model | RPM | TPM | Why this model |
|------|-------|-----|-----|----------------|
| **Architect** (Step 1) | `llama-3.1-8b-instant` | 30 | 6,000 | Fast, cheap, high-volume. Decomposition is a simple task. |
| **Reporter** (Step 4, primary) | `meta-llama/llama-4-scout-17b-16e-instruct` | 30 | 30,000 | Deep reasoning, native Arabic, supports `json_schema` for strict structured output. |
| **Reporter** (extreme Ikhtilaf fallback) | `llama-3.3-70b-versatile` | 30 | 12,000 | Maximum reasoning power for complex theological conflicts. |

**Throughput math for one user query** (verified from Groq dashboard, 2026-06-22):

| Call | Input tokens | Output tokens | Total | % of TPM limit |
|------|-------------|---------------|-------|-----------------|
| Architect (8b-instant) | 657 | 60 | 717 | 12% of 6K TPM |
| Reporter (Scout 17b) | 16,600 | 937 | 17,537 | 59% of 30K TPM |
| **Total per query** | **17,257** | **997** | **18,254** | — |

**Key takeaway**: The Reporter's input tokens (16.6K) dominate because we send 10 retrieved chunks as context (~1,600 tokens per chunk including the bilingual context prefix). This means:
- **~1 query per minute max** on the free tier (59% of TPM per query → can't do 2 in the same minute)
- The 30 RPM request limit is NOT the bottleneck — TPM is
- If we need faster throughput, we must either (a) send fewer chunks, (b) compress the chunk context, or (c) upgrade to the Developer plan

**Rate limit headers** (from the dashboard, 2026-06-22):
- The dashboard shows 25 RPM, not 30 RPM as the docs table states. This may be an org-level setting or a free-tier reduction. Treat 25 RPM as the real ceiling.

---

## 2. Structured Output Modes — Verified Compatibility

Groq supports two `response_format` modes for structured output (verified
from the Groq API reference, 2026-06-22):

### `json_schema` mode (Structured Outputs)
- **What it does**: The model is forced to match a supplied JSON schema at the
  API level. Strictest possible enforcement.
- **Compatibility**: "only available on supported models" (per Groq docs).
  `llama-3.1-8b-instant` does **NOT** support it (returns HTTP 400).
  `meta-llama/llama-4-scout-17b-16e-instruct` **does** support it (verified
  by the passing Reporter test in DEC-022).
- **instructor mode**: `instructor.Mode.JSON_SCHEMA`

### `json_object` mode (older JSON mode)
- **What it does**: Ensures the model's output is valid JSON, but does NOT
  enforce a specific schema. The schema must be described in the prompt.
- **Compatibility**: Broader — works on models that don't support `json_schema`.
- **instructor mode**: `instructor.Mode.JSON`

### Function calling / Tools (not response_format)
- **What it does**: The model calls a "function" (defined by a JSON schema)
  and returns the arguments as a structured object. Universally supported
  across Groq chat models.
- **Compatibility**: Broadest. Works on `llama-3.1-8b-instant`.
- **instructor mode**: `instructor.Mode.TOOLS`

### NUR's mode assignments

| Role | Model | instructor Mode | Why |
|------|-------|-----------------|-----|
| **Architect** | `llama-3.1-8b-instant` | `Mode.TOOLS` | 8b-instant doesn't support `json_schema`. Schema is simple (flat list of strings), so tool-call enforcement is sufficient. |
| **Reporter** | `llama-4-scout-17b-16e-instruct` | `Mode.JSON_SCHEMA` | Scout supports it. The Reporter's schema is complex (nested `direct_reports` array) and anti-hallucination-critical, so we need the strictest enforcement. |

---

## 3. Rate Limit Headers (for runtime monitoring)

Every Groq API response includes these headers (verified from Groq docs):

| Header | Meaning |
|--------|---------|
| `x-ratelimit-limit-requests` | RPD limit (always daily, not per-minute) |
| `x-ratelimit-limit-tokens` | TPM limit (always per-minute, not daily) |
| `x-ratelimit-remaining-requests` | RPD remaining |
| `x-ratelimit-remaining-tokens` | TPM remaining |
| `x-ratelimit-reset-requests` | Time until RPD resets (e.g. `2m59.56s`) |
| `x-ratelimit-reset-tokens` | Time until TPM resets (e.g. `7.66s`) |
| `retry-after` | Seconds to wait (only set on HTTP 429) |

**Phase 3 TODO**: Wire these headers into the Reporter's retry logic so we can
log remaining quota and proactively back off before hitting 429, instead of
relying on `tenacity`'s reactive retry.

---

## 4. When to Ask the User for Updated Limits

Per `.agents/rules/updatesmdfiles.md` Rule 4: if the agent cannot access the
Groq docs (they're JS-rendered and not scrapable), the agent MUST ask the user
to paste the current limits rather than guessing. The user can always say "I
can't access it either" — that's an acceptable answer; we then proceed with
the last known-good values from this document.

**Triggers for re-verification**:
- Groq announces a model lineup change (new model, deprecated model).
- NUR wants to switch to a model not listed here.
- A 400/403 error mentions an unsupported feature.
- Quarterly review (limits change over time).
