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

## 4. Never Guess External Capabilities — Ask the User to Verify
- If you cannot access an external doc, API reference, rate-limit page, or model-compatibility matrix (e.g. it's JS-rendered and not scrapable, or behind auth), you MUST ask the user to check it for you or paste the relevant section.
- The user can always say "I can't access it either" — that is an acceptable answer. In that case, proceed with the last known-good values from the project's reference docs (e.g. `docs/GROQ_REFERENCE.md`) and clearly mark the assumption as unverified.
- NEVER hallucinate capabilities, rate limits, or model features from training memory. Library APIs evolve, models get deprecated, limits change. Memory-based guesses cause silent runtime failures (e.g. calling `json_schema` mode on a model that doesn't support it).
- This rule exists because guessing about external systems is the #1 source of integration bugs. A 30-second question to the user saves a 30-minute debugging session.

## 5. Respect All Pillars — Religious Integrity Is Non-Negotiable
- NUR is a religious project. A wrong answer is not just a bug — it is theologically harmful. An error about Allah, the Prophet ﷺ, the Quran, or Islamic rulings could mislead a believer, potentially leading to haram actions or, in the worst case, statements that approach shirk (associating partners with Allah).
- Before writing ANY code that produces user-facing Islamic content, you MUST re-read `docs/PILLARS.md` and verify your implementation respects every applicable pillar. If you cannot name which pillars your change touches, you have not read the docs carefully enough.
- The 10 Pillars are non-negotiable. If a feature request or code change conflicts with a pillar, FLAG IT to the user — do not implement the conflict. Do not silently work around a pillar for convenience.
- **When in doubt, ABSTAIN.** Pillar 4 (Absolute Reliability) mandates: "The system must abstain if it lacks confidence." A response of "I do not have sufficient reliable sources to answer this question" is ALWAYS better than a wrong answer. Never let the pipeline produce a confident-sounding answer from weak or insufficient sources.
- **Hadith grades are sacred.** Never present a weak (Da'if) or fabricated (Mawdu') hadith without a visible warning. Never present a hadith interpretation without noting that laypeople cannot interpret hadith independently — scholarly opinion is required (Pillar 8).
- **Ikhtilaf (scholarly disagreement) must be presented, not resolved.** When scholars disagree, NUR presents all views neutrally and never takes a side (Pillar 9). The system must never declare one madhhab "correct" — it reports what each school says.
- This rule exists because the stakes are eternal. A user may act on a NUR answer for their religious practice — prayer, marriage, business, repentance. We owe them correctness over convenience, abstention over approximation, and transparency over confidence.
