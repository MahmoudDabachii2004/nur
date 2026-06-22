# NUR V3 — Vue d'ensemble

> Document fondateur de l'architecture V3. À lire en premier.
> Statut : **DESIGN** (en cours de validation)
> Date : 2026-06-23

---

## Pourquoi V3 ? — Le constat d'échec de V1/V2

Les versions V1 et V2 du pipeline NUR ont échoué pour **5 raisons structurelles** qui rendaient le retrieval fondamentalement broken. Aucun tuning de reranker, aucun changement de prompt, aucune ajustement de `top_k` ne pouvait sauver l'archi — les chunks eux-mêmes étaient mal construits.

### Faille 1 — Le tafsir orphelin

V1/V2 avait **4 collections séparées** : `quran`, `hadith`, `tafsir_ar`, `tafsir_en`. Le tafsir était embeddé **seul**, sans le verset qu'il explique. Conséquence : BGE-M3 voyait "Ibn Kathir explique que..." mais ne voyait pas le verset expliqué. Quand l'utilisateur cherchait un verset, le tafsir orphelin n'avait aucun signal sémantique pour matcher — il fallait que les mots du verset soient présents dans le tafsir pour qu'il remonte.

### Faille 2 — Numérotation globale décalée

Le chunker V1 utilisait un compteur global d'ayah (`quran_2_10`) au lieu de la numérotation standard Quran (`quran_2_3`). Conséquence : URLs cassées (`quran.com/2/10` au lieu de `quran.com/2/3`), métadonnées fausses, impossible de cross-référencer avec des sources externes (sunnah.com, quran.com, tafsir apps).

### Faille 3 — Compétition inter-collections

V1/V2 requêtait les 4 collections en parallèle avec RRF fusion. Conséquence : pour une question comme "establish prayer", le tafsir (texte plus long, plus de keywords) remportait le top-5 et **évinçait le verset lui-même**. L'utilisateur voyait "selon Ibn Kathir..." mais jamais le verset 2:43 directement.

### Faille 4 — Pas de pont sémantique pour questions modernes

Le Quran est **7ème siècle**. Les questions sont **21ème siècle** : "smoking haram ?", "crypto halal ?", "AI and fate ?", "dating apps ?". Aucun de ces mots n'existe dans le Quran. Sans tafsir comme pont sémantique, BGE-M3 ne trouve rien → le LLM hallucine ou dit "pas de réponse".

### Faille 5 — Pas de context card cross-lingual

V1/V2 n'avait que AR + EN dans l'embedding. Une requête FR "Pourquoi la prière est obligatoire ?" ne matchait pas "establish prayer" car BGE-M3, bien que multilingual, n'a pas assez de signal pour aligner "prière" et "salah" sans keyword bridge. Resultat : 67% des requêtes FR échouaient (cf. `scratch/retrieval_audit_results.md`).

---

## Le principe clé de V3 — **Le tafsir EST le pont**

L'insight central de V3 : **le tafsir n'est pas un supplément optionnel, c'est le connecteur sémantique** entre la Parole d'Allah (immuable, 7ème siècle) et la question de l'utilisateur (moderne, dans n'importe quelle langue).

### Pourquoi Ibn Kathir est légitime comme pont

1. **Méthodologie bil-Mathur** : Ibn Kathir explique le Quran par le Quran lui-même, puis par les Hadiths authentiques, puis par les dires des Compagnons. Ce n'est pas une opinion personnelle — c'est une chaîne de transmission.
2. **Accepté consensusellement** (Ijma' des savants sunnites) comme référence.
3. **Inclut les causes de révélation (Asbab al-Nuzul)** : sans le tafsir, le verset 2:195 ("do not throw yourselves into destruction") semble abstrait. Le tafsir révèle qu'il a été révélé suite à un événement spécifique, et que les scholars (Mujahid, Qatadah) l'ont étendu à toute forme de self-harm — c'est ce qui permet de répondre à "smoking haram ?".

### Pourquoi on garde la séparation divine/humain (Pillar 1)

Pillar 1 dit : "We do not mix the Word of Allah with human commentary." V3 respecte ça **visuellement** dans l'embedding_text via 3 couches labellisées :

```
[1] CONTEXT CARD       → LLM-generated, FR/EN/AR keywords
[2] WORD OF ALLAH      → PURE, untouched, labellisé "Quran | Surah X | Ayah Y"
[3] HUMAN COMMENTARY   → Labellisé "Tafsir Ibn Kathir — Human interpretation"
```

BGE-M3 voit les 3 couches (bon pour le retrieval), mais elles sont **typographiquement distinctes**. Le Reporter LLM reçoit aussi les 3 couches avec les labels, et **doit citer** : `[Quran 2:195]` pour la Parole, `[Tafsir Ibn Kathir: ...]` pour le commentaire. Post-generation check : si un citation Quran contient du texte du tafsir, on rejette.

---

## Les 4 piliers de V3

### Pilier V3-1 — Triple-Layer Chunk (Quran)

Chaque chunk Quran contient 3 couches connectées :
1. **Context Card** (LLM-generated, 50-100 tokens) : FR + EN + AR keywords, thème, règle
2. **Word of Allah** (PURE) : AR Uthmani + EN Saheeh + Previous/Next ayah
3. **Tafsir** (1 ou plusieurs, labellisés) : Ibn Kathir + optionnellement Tabari/Sa'di/Mukhtasar

### Pilier V3-2 — Two-Phase Sequential Retrieval

```
PHASE A : Search quran_v3 ONLY → top 5 chunks (verse + tafsir)
PHASE B : Search hadith_v3 ONLY → top 5 chunks (hadith + grade)
→ Combine en 2 blocs XML séparés pour le Reporter LLM
```

Jamais de compétition. Le Quran a priorité absolue. Si Phase A retourne 0 chunk > seuil, le LLM doit dire "Le Quran n'aborde pas ceci directement" et basculer sur Phase B.

### Pilier V3-3 — Multi-Tafsir Strategy

Le parquet existant a Ibn Kathir EN. Le repo spa5k/tafsir_api expose **115 tafsirs**. V3 utilise :
- **Ibn Kathir EN** (primary, bil-Mathur) — déjà dans la data
- **Ibn Kathir AR** (primary AR, bil-Mathur) — à télécharger
- **Al-Tabari AR** (référence classique, bil-Mathur, 10ème siècle) — à télécharger
- **As-Sa'di AR** (moderne, accessible, 20ème siècle) — à télécharger
- **Al-Mukhtasar EN** (concis, moderne) — à télécharger

Le chunk peut contenir plusieurs tafsirs, chacun labellisé. Le LLM peut citer `[Tafsir Ibn Kathir: ...]` ou `[Tafsir As-Sa'di: ...]` — voire présenter l'**Ikhtilaf** (désaccord) entre tafsirs quand ils divergent.

### Pilier V3-4 — Standard Numbering + URL Integrity

- ID : `SRC-QURAN-2-195` (standard `surah:ayah`, pas de compteur global)
- URL : `https://quran.com/2/195` (validée, pas décalée)
- Tafsir URL : `https://quran.com/tafsir/2/195`
- Hadith URL : `https://sunnah.com/{collection}:{hadith_number}` (stockée dans metadata, JAMAIS reconstruite dynamiquement — voir ARCHITECTURE.md Sin #5)

---

## Collections ChromaDB

```
quran_v3       6,236 chunks  (1 ayah = 1 chunk + tafsir + context card)
hadith_v3     33,738 chunks  (1 hadith = 1 chunk + grade + context card)
```

Plus de `tafsir_ar`, plus de `tafsir_en` séparés. Le tafsir est **dans** le chunk Quran.

## Modèles

| Composant | Modèle | Hébergement |
|-----------|--------|-------------|
| Embeddings | `BAAI/bge-m3` (1024 dim) | Local (BGE-M3 2.3GB) |
| Sparse (BM25-like) | Built-in ChromaDB | Local |
| Reranker | `BAAI/bge-reranker-v2-m3` | Local |
| Architect (query decomposition) | `llama-3.1-8b-instant` | Groq |
| Reporter (final answer) | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq |
| Context Card Generator | `llama-3.1-8b-instant` | Groq |
| Fallback offline | `llama3.1:8b` | Local Ollama |

## Index des documents V3

| # | Document | Sujet |
|---|----------|-------|
| 00 | `00_OVERVIEW.md` | Ce document — vue d'ensemble |
| 01 | `01_DATA_SOURCES.md` | Sources de données brutes (Quran, Hadith, 115 tafsirs) |
| 02 | `02_CHUNK_SCHEMA.md` | Schéma détaillé des chunks V3 |
| 03 | `03_TAFSIR_STRATEGY.md` | Stratégie multi-tafsir |
| 04 | `04_RETRIEVAL_PIPELINE.md` | Pipeline 2 phases + RRF + rerank |
| 05 | `05_EMBEDDING_DESIGN.md` | Design du embedding_text |
| 06 | `06_GENERATION_VERIFICATION.md` | Reporter LLM + vérification anti-hallucination |
| 07 | `07_FAILURE_MODES.md` | 8 failles V1/V2 et corrections V3 |
| 08 | `08_EXAMPLES.md` | 5 exemples concrets end-to-end |

## Prochaines étapes (ordre strict)

1. Valider cette doc (review utilisateur)
2. `scripts/v3/01_download_quran.py` — alquran.cloud API (Uthmani + Saheeh EN)
3. `scripts/v3/02_download_hadith.py` — meetif API (6 collections)
4. `scripts/v3/03_download_tafsirs.py` — Ibn Kathir AR + Tabari + Sa'di + Mukhtasar EN
5. `scripts/v3/04_generate_context_cards.py` — LLM batch pour 6,236 versets (Groq, ~$30)
6. `scripts/v3/05_build_chunks.py` — assembler les chunks V3 (3 couches Quran, 2 couches Hadith)
7. `scripts/v3/06_embed_and_index.py` — BGE-M3 + ChromaDB (Colab T4 ou local)
8. `scripts/v3/07_verify_pipeline.py` — 5 exemples du doc 08 doivent passer
