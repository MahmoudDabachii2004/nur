---
trigger: always_on
---

# Antigravity Developer Rules — Documentation Updates & Test Hand-off

## 1. Always Update `docs/PHASES.md` After a Code Change
- After completing ANY code change (new module, bug fix, refactor, test script), you MUST update `docs/PHASES.md` in the same commit (or the very next one if you forgot).
- Update the relevant phase's checklist: mark the item as `[x]` done, or add a new item if the work was unplanned.
- If the change introduces a new test script, mention it in the checklist item so the validation step is traceable.
- This rule exists so that anyone (the user, a future agent, or future-you in 6 months) can open `docs/PHASES.md` and see at a glance what is done and what is next.

## 2. Only Update Architecture Docs When a Structural Change Is Agreed
- Architecture docs (`docs/CONTEXT.md`, `docs/PILLARS.md`, `docs/RAG_PIPELINE_ARCHITECTURE.md`) are the **confirmed source of truth**.
- Code MUST conform to these docs. Do NOT silently drift the docs to match code.
- You may only edit these docs when the user explicitly agrees on a structural change (new model, new pipeline step, new pillar, etc.). In that case, update the doc AND log the change in `docs/brains.md` with a new Decision ID.
- If you notice a mismatch between docs and code, FLAG IT to the user — do not silently "fix" either side.

## 3. Never Skip Tests Silently — Ask the User to Run Them
- If you cannot run a test yourself (missing model weights, missing GPU, missing API key, disk-space limit, etc.), you MUST explicitly tell the user.
- Use this exact phrasing pattern:
  > "I can't test this on my side because [reason]. Please run `python scripts/<name>.py` on your machine and paste me the output."
- NEVER silently skip a test and present the work as validated. The user must always know which parts are verified vs. unverified.
- This rule exists because presenting unverified code as "done" breaks trust and causes bugs to surface much later when they are harder to trace.
