# NUR — The 10 Pillars

> Full details in [`ARCHITECTURE.md`](ARCHITECTURE.md). This is the executive summary.

## Pillar 1 — Triple-Index + Scholar Index

Four separate ChromaDB collections instead of one mixed index:

| Collection | Chunks | Granularity | Why separate |
|-----------|--------|-------------|--------------|
| `quran_dense` | 6,236 | 1 ayah = 1 chunk | Word of Allah — never confused with anything else |
| `hadith_dense` | 33,738 | 1 hadith = 1 chunk | Word of the Prophet ﷺ — grade-aware |
| `tafsir_ar_dense` | 6,236 | 1 section = 1 chunk | Classical commentary |
| `tafsir_en_dense` | 6,236 | 1 section = 1 chunk | English commentary |

Plus a **scholar index** in Phase 7 for fatwas and opinions.

## Pillar 2 — Arabic-First Cross-Lingual Retrieval

Arabic is the source of truth. English is a comprehension aid.

`BAAI/bge-m3` places Arabic and English in the **same vector space** — so a query like
"charity obligatory" matches `زكاة` directly, without a translation step.

## Pillar 3 — Authenticity-Weighted Retrieval

| Grade | Weight | Meaning |
|-------|--------|---------|
| Sahih | ×1.30 | Authentic chain — +30% boost |
| Hasan | ×1.10 | Good chain — +10% boost |
| Da'if | ×0.50 | Weak chain — -50% penalty |
| Mawdu' | ×0.00 | Fabricated — excluded (kept only for fake-hadith detection) |

A less-similar Sahih hadith can rank above a more-similar Da'if one — because in Islam,
the strength of the chain matters more than text similarity.

## Pillar 4 — Post-Generation Verification

Three-layer anti-hallucination (Phase 6):

1. **NLI verification** — every sentence of the LLM response must be entailed by retrieved chunks (threshold 0.95)
2. **Character-by-character Quran check** — every cited verse must match the Uthmani text exactly (after normalization)
3. **Decoupled grounding** — LLM produces `exact_citations` and `synthesis` separately; parser rejects if citation isn't word-for-word

## Pillar 5 — Context-Enriched Chunks (Anthropic Technique)

Each chunk is prefixed with LLM-generated context (Phase 4):

> "This verse is from Surah Al-Baqarah, ayah 255, known as Ayat al-Kursi. It discusses the omnipotence and omniscience of Allah. Meccan revelation."

Reduces retrieval failures by 67% (Anthropic, 2024). Essential for Islamic text where
verses depend heavily on revelation context (Asbab al-Nuzul).

## Pillar 6 — Hybrid LLM (Groq + Ollama)

| Use case | Model | Why |
|----------|-------|-----|
| Default Arabic/English | Groq Qwen3-32B | Best Arabic on free tier (60 RPM) |
| Complex fiqh reasoning | Groq Qwen3.6-27B | Thinking mode, GPQA 87.8 |
| Offline / privacy | Ollama Qwen2.5-7B (local) | 100% on-device, ~15 tok/s on M4 |
| Groq down | OpenRouter Qwen3-32B | Free fallback |

**Key principle**: embeddings run 100% locally — sacred text never leaves the machine.
Only the question + retrieved context go to the cloud LLM.

## Pillar 7 — Source ID Protocol

Inject numbered IDs into the prompt:

```xml
<document id="S1">
  <source_id>SRC-QURAN-2-255</source_id>
  <source_type>quran</source_type>
  <label>Quran 2:255</label>
  <arabic>اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ...</arabic>
  <english>Allah! There is no deity except Him...</english>
  <url>https://quran.com/2/255</url>
</document>
```

LLM is forced to cite `[S1]`, `[S2]`, etc. Post-processing maps each `[SX]` to the
rich display with Arabic text + translation + clickable URL.

## Pillar 8 — Scholar Opinions Are Mandatory

The system **never** gives its own opinion. It **always** reports:

> "Imam Malik said [opinion], based on [Quran verse] and [hadith]. Source: Al-Muwatta, Book X, Chapter Y."

Sources (all free): IslamQA.info, Islamweb.net, Dar al-Ifta Egypt, Shamela.ws.

## Pillar 9 — Ikhtilaf Awareness

Five levels of consensus presentation (absolute neutrality required):

1. **Ijma'** (unanimous consensus)
2. **Overwhelming majority** (1 dissenting school)
3. **Significant disagreement** (no clear majority)
4. **Isolated opinion** (1 school vs all others) — most delicate
5. **Aqeedah** (no ikhtilaf tolerated — must match authentic texts)

**Absolute rule**: never encourage, never discourage. Present facts. Let the Muslim decide.

## Pillar 10 — Arabic-English Bilingual

Arabic: source of truth, always displayed.
English: comprehension aid, always shown alongside Arabic.

(French was considered in the original architecture doc but dropped — project is
English+Arabic only per project direction.)
