# NUR V3 — Documentation

> Architecture V3 du pipeline NUR (Islamic RAG).
> Statut : **DESIGN VALIDATED** — prêt pour implémentation
> Date : 2026-06-23

---

## Vue d'ensemble

V3 est une refonte from-scratch du pipeline NUR pour résoudre les 5 failles structurelles de V1/V2 qui rendaient le retrieval broken.

### Décision fondatrice

**Le tafsir est le pont sémantique** entre la Parole d'Allah (immuable, 7e siècle) et la question de l'utilisateur (moderne, dans n'importe quelle langue). Le tafsir n'est pas un supplément optionnel — c'est le connecteur qui permet de répondre à "smoking haram ?" en récupérant le verset 2:195 (self-destruction) + son tafsir Ibn Kathir (qui explicite "any action causing harm to one's body").

### Principes clés

1. **2 collections** : `quran_v3` (6,236 chunks) + `hadith_v3` (33,738 chunks)
2. **3 couches dans le chunk Quran** : Context Card (LLM) + Word of Allah (PURE) + 4 Tafsirs (labellisés)
3. **2 phases séquentielles** : Phase A (Quran+Tafsir) puis Phase B (Hadith) — jamais en compétition
4. **Confidence score par phase** : STRONG/WEAK/EMPTY → gate sur Phase B + disclaimer approprié
5. **Truncation 600 chars** par tafsir dans l'embedding (full text en metadata pour le LLM)
6. **Cross-refs Quran→Hadith** pré-calculés via parsing tafsir
7. **4 tafsirs** : Ibn Kathir EN+AR + Al-Tabari AR + As-Sa'di AR
8. **Vérification anti-hallucination** : NLI + Quran char-by-char + Source ID validation

---

## Index des documents

| # | Document | Sujet | Statut |
|---|----------|-------|--------|
| 00 | [00_OVERVIEW.md](00_OVERVIEW.md) | Vue d'ensemble + pourquoi V3 | ✅ |
| 01 | [01_DATA_SOURCES.md](01_DATA_SOURCES.md) | Sources de données (Quran, Hadith, 115 tafsirs) | ✅ |
| 02 | [02_CHUNK_SCHEMA.md](02_CHUNK_SCHEMA.md) | Schéma détaillé des chunks V3 | ✅ |
| 03 | [03_TAFSIR_STRATEGY.md](03_TAFSIR_STRATEGY.md) | Stratégie multi-tafsir + Ikhtilaf | ✅ |
| 04 | [04_RETRIEVAL_PIPELINE.md](04_RETRIEVAL_PIPELINE.md) | Pipeline 2 phases + RRF + rerank | ✅ |
| 05 | [05_EMBEDDING_DESIGN.md](05_EMBEDDING_DESIGN.md) | Analyse complète truncation tafsir | ✅ |
| 06 | [06_GENERATION_VERIFICATION.md](06_GENERATION_VERIFICATION.md) | Reporter LLM + vérification | ✅ |
| 07 | [07_FAILURE_MODES.md](07_FAILURE_MODES.md) | 8 failles V1/V2 + fixes V3 | ✅ |
| 08 | [08_EXAMPLES.md](08_EXAMPLES.md) | 5 exemples end-to-end | ✅ |

---

## Décisions figées (V3 frozen spec)

| Aspect | Décision |
|--------|----------|
| Collections | `quran_v3` + `hadith_v3` (2 seulement) |
| Tafsirs par verset | 4 (Ibn Kathir EN + AR, Al-Tabari AR, As-Sa'di AR) |
| Tafsir truncation | 600 chars dans embedding, full text en metadata |
| Context Card | LLM-generated (Groq llama-3.1-8b-instant), FR+EN+AR |
| Pipeline phases | 2 séquentielles (Phase A Quran, Phase B Hadith) |
| Confidence thresholds | STRONG ≥ 0.5, WEAK > 0.3, EMPTY ≤ 0.3 |
| Embedding model | BAAI/bge-m3 (1024 dim) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Reporter LLM | Groq llama-4-scout-17b-16e-instruct (fallback: 3.3-70b, Ollama 8b) |
| Verification | NLI (0.95) + Quran char-by-char + Source ID validation |
| Cross-refs Quran→Hadith | Pré-calculés via parsing tafsir + Asbab al-Nuzul |

### Ce qui est EXCLU de V3 (pour V3.1+)

- Multi-chunk (1 parent + N tafsir children) — gardé simple pour V3
- Plus de 4 tafsirs par verset — pour limiter la dilution
- Qurtubi, Jalalayn, Ibn Uthaymeen — V3.1
- Traduction FR du tafsir Ibn Kathir — V3.1
- Anthropic Claude pour Context Cards — testé avec Groq d'abord

---

## Plan de build V3 (ordre strict)

| # | Script | Description | Temps estimé |
|---|--------|-------------|--------------|
| 1 | `01_download_quran.py` | alquran.cloud API (Uthmani + Saheeh EN) | 5 min |
| 2 | `02_download_hadith.py` | meetif API (6 collections) | 10 min |
| 3 | `03_download_tafsirs.py` | Ibn Kathir AR + Tabari + Sa'di (upstream JSON, pas parquet) | 15 min |
| 4 | `04_generate_context_cards.py` | LLM batch pour 6,236 versets (Groq) | 4h |
| 5 | `05_compute_ikhtilaf.py` | NLI cross-tafsir + AR→EN translation | 2h15 |
| 6 | `06_compute_cross_refs.py` | Parsing tafsirs → hadiths cités | 30 min |
| 7 | `07_build_chunks.py` | Assembler chunks V3 (JSONL) | 15 min |
| 8 | `08_embed_and_index.py` | BGE-M3 + ChromaDB | 30 min |
| 9 | `09_verify_pipeline.py` | 5 exemples du doc 08 doivent passer | 10 min |

**Total** : ~7-8 heures, $0 (Groq free tier)

---

## 5 cas de validation (ground truth)

| Cas | Question | Catégorie | Phase A attendue | Confidence attendue |
|-----|----------|-----------|------------------|---------------------|
| 1 | "Pourquoi la prière est obligatoire ?" | Classique FR | STRONG | high |
| 2 | "معلومات عن كرسي الله" | Classique AR | STRONG | high |
| 3 | "Fumer est haram ?" | Moderne pont tafsir | STRONG | high |
| 4 | "Wudu avec eau du puits ?" | Fiqh Ikhtilaf | STRONG | high |
| 5 | "L'IA a-t-elle une âme ?" | Moderne EMPTY | WEAK | low |

Le build V3 est considéré **réussi** si les 5 cas produisent les phases/confidences attendues.

---

## Prochaines étapes

1. ✅ **Documentation complète** (ce dépôt — fait)
2. ⏭️ **Implémenter `scripts/v3/01_download_quran.py`**
3. ⏭️ **Implémenter `scripts/v3/02_download_hadith.py`**
4. ⏭️ **Implémenter `scripts/v3/03_download_tafsirs.py`** (re-fetch upstream, NE PAS utiliser le parquet existant qui a 70% de vides)
5. ⏭️ **Implémenter les 6 autres scripts**
6. ⏭️ **Run `09_verify_pipeline.py`** sur les 5 cas du doc 08

---

## Références aux Pillars

V3 implémente les 10 Pillars originaux de NUR :

| Pillar | Implémentation V3 |
|--------|-------------------|
| 1 (Triple-Index) | 2 collections (Quran+Tafsir unifié + Hadith) |
| 2 (Arabic-First) | AR Uthmani source of truth, BGE-M3 multilingual |
| 3 (Authenticity-Weighted) | grade_weight Sahih 1.30, Hasan 1.10, Da'if 0.50 |
| 4 (Post-Gen Verification) | NLI + Quran char-by-char + Source ID |
| 5 (Dual-Layer Context) | 3 couches (Context Card + Word of Allah + Tafsirs) |
| 6 (Hybrid LLM) | Architect (8b), Reporter (17b), Reasoning fallback (70b), Offline (8b local) |
| 7 (Structured Citation) | XML prompt + [Sn] format + source_id SRC-* |
| 8 (Scholar Opinions) | Tafsirs labellisés avec source, category, era |
| 9 (Ikhtilaf) | NLI cross-tafsir + metadata.ikhtilaf + Reporter prompt instruction |
| 10 (Bilingual + AR) | FR/EN/AR Context Card + AR toujours affiché |
