# V3 — Pipeline de retrieval (2 phases séquentielles)

> Comment on retrouve les bons chunks pour une question utilisateur.
> Implémente : 2 phases séparées (Quran puis Hadith), RRF fusion, reranker, **confidence score par phase**.
> Date : 2026-06-23

---

## Vue d'ensemble du pipeline

```
User question (FR/EN/AR)
        │
        ▼
┌────────────────────────────────────┐
│ STEP 1: Architect LLM               │
│  Decompose en 1..N sub-questions    │
│  Output: [Q1, Q2, Q3, ...]          │
└────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│ STEP 2: PHASE A — Quran+Tafsir      │
│  Search quran_v3 ONLY                │
│  Multi-query × dense+sparse         │
│  RRF fusion → top 400                │
│  Rerank (bge-reranker-v2-m3) → top 5│
│  → Confidence score A                │
└────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│ STEP 3: Auto-pull hadiths liés      │
│  Pour chaque top-Quran chunk:       │
│    Si hadith_cross_refs present:    │
│      Auto-pull hadiths              │
└────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│ STEP 4: PHASE B — Hadith            │
│  Decision gate:                      │
│    If Confidence A ≥ threshold:     │
│      Run Phase B with normal weight  │
│    If Confidence A < threshold:     │
│      Run Phase B + penalty flag      │
│                                      │
│  Search hadith_v3 ONLY               │
│  Same flow → top 5                  │
│  → Confidence score B                │
└────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│ STEP 5: Reporter LLM                │
│  Receive top-5 Quran+Tafsir chunks  │
│  + top-5 Hadith chunks              │
│  + auto-pulled hadiths              │
│  + confidence flags                 │
│                                      │
│  Generate structured JSON response   │
└────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│ STEP 6: Verification (Pillar 4)     │
│  NLI check + Quran char-by-char     │
│  + Source ID validation              │
└────────────────────────────────────┘
        │
        ▼
  Final answer to user
```

---

## Step 1 — Architect LLM (query decomposition)

### Objectif

Une question utilisateur est souvent complexe. La décomposer en sous-questions augmente le recall.

### Exemple

```
User: "Est-ce que fumer est haram ?"

Architect output:
[
  "smoking haram islam",
  "self-harm forbidden quran",          ← KEY: pont sémantique
  "destroying oneself quran",
  "wasteful spending quran",
  "tobacco ruling islam"                ← keyword moderne
]
```

### Modèle

- **Modèle** : `llama-3.1-8b-instant` (Groq, fast/cheap)
- **Temperature** : 0.0 (déterministe)
- **Max tokens** : 256 (juste une liste de sous-questions)
- **Prompt** :

```
You are an Islamic search query decomposer.
Decompose the user's question into 3-7 sub-questions that will be 
used to search an Islamic database (Quran + Hadith + Tafsir).

Rules:
- Output: JSON array of strings
- Each sub-question ≤ 10 words
- Mix FR/EN/AR keywords when relevant
- Include synonyms and root words
- Include the modern topic AND the Quranic equivalent

User question: "{user_question}"

Output:
```

### Fallback offline

Si Groq 429 → reroute vers Ollama `llama3.1:8b` local.

---

## Step 2 — Phase A : Quran+Tafsir retrieval

### Sub-step 2a : Multi-query dense retrieval

Pour chaque sous-question Q_i et la question originale :

```python
queries = [user_question] + sub_questions  # 4-8 queries
quran_scores = {}  # chunk_id → score

for q in queries:
    embedding = bge_m3.encode(q)
    # ChromaDB dense search, top 100
    results = quran_v3.query(
        query_embeddings=[embedding],
        n_results=100
    )
    for rank, (chunk_id, score) in enumerate(zip(results["ids"], results["distances"])):
        # RRF score for dense
        rrf_dense = 1.0 / (RRF_K + rank + 1)
        quran_scores[chunk_id] = quran_scores.get(chunk_id, 0) + \
            RRF_ALPHA_DENSE * rrf_dense
```

### Sub-step 2b : Multi-query sparse retrieval

```python
for q in queries:
    # BM25-like sparse search
    results = sparse_search_quran(q, top_k=100)
    for rank, (chunk_id, score) in enumerate(results):
        rrf_sparse = 1.0 / (RRF_K + rank + 1)
        quran_scores[chunk_id] = quran_scores.get(chunk_id, 0) + \
            (1 - RRF_ALPHA_DENSE) * rrf_sparse
```

### Sub-step 2c : RRF Fusion + dedup

```python
# Sort by RRF score desc
ranked = sorted(quran_scores.items(), key=lambda x: -x[1])

# Take top 400 (DEC-034: pool=400 reaches 50% recall)
top_400 = ranked[:400]
```

### Sub-step 2d : Reranker

```python
# BGE reranker scores each (query, chunk) pair
reranker_inputs = [(user_question, chunk_text) for chunk_id, chunk_text in top_400]
rerank_scores = bge_reranker_v2_m3.predict(reranker_inputs)

# Sort by rerank score
reranked = sorted(zip(top_400, rerank_scores), key=lambda x: -x[1])

# Take top 5
top_5_quran = reranked[:5]
```

### Sub-step 2e : Authenticity weighting

```python
# For Quran chunks, grade_weight = 1.0 (word of Allah, neutral)
# But we boost chunks based on relevance to user's question

for chunk, score in top_5_quran:
    chunk["final_score"] = score * chunk["metadata"]["grade_weight"]
```

(Quran grade_weight = 1.0 toujours, mais on garde la logique pour cohérence avec Hadith)

### Sub-step 2f : **Confidence score Phase A**

**C'est ici qu'on implémente le point 1 validé**.

```python
# Top-1 rerank score (normalisé 0..1 par bge-reranker)
top1_score = top_5_quran[0]["rerank_score"]  # 0..1

# Confidence A = max score among top-5
confidence_A = max(c["rerank_score"] for c in top_5_quran)

# Threshold
THRESHOLD_PHASE_A = 0.5

if confidence_A >= THRESHOLD_PHASE_A:
    phase_a_status = "STRONG"   # Le Quran répond directement
elif confidence_A > 0.3:
    phase_a_status = "WEAK"     # Le Quran aborde le sujet mais indirectement
else:
    phase_a_status = "EMPTY"    # Le Quran n'aborde pas ce sujet
```

### Decision gate (Phase B)

```python
if phase_a_status == "STRONG":
    # Phase B normale — Hadith en complément
    run_phase_b(weight_penalty=1.0, instruction=None)
elif phase_a_status == "WEAK":
    # Phase B avec instruction "Le Quran aborde indirectement..."
    run_phase_b(
        weight_penalty=1.0,
        instruction="Quran addresses this indirectly. 
                     Hadiths may provide more detail."
    )
else:  # EMPTY
    # Phase B avec instruction explicite
    run_phase_b(
        weight_penalty=0.8,  # 20% penalty: les hadiths sont "weak signal"
        instruction="The Quran does NOT directly address this topic. 
                     The following hadiths may provide guidance. 
                     You MUST start your answer with: 
                     'Le Quran n'aborde pas directement ce sujet. 
                     Selon les hadiths...'"
    )
```

---

## Step 3 — Auto-pull hadiths liés (cross-refs)

Pour chaque chunk Quran retenu en Phase A, on récupère les hadiths pré-calculés via `hadith_cross_refs` :

```python
auto_pulled_hadiths = []

for chunk in top_5_quran:
    cross_refs = chunk["metadata"].get("hadith_cross_refs", {})
    if cross_refs.get("high_confidence"):
        for hadith_id in cross_refs["high_confidence"]:
            hadith = hadith_v3.get(id=hadith_id)
            if hadith:
                hadith["auto_pulled"] = True
                hadith["auto_pulled_from"] = chunk["id"]
                auto_pulled_hadiths.append(hadith)

# Limit to top 5 auto-pulled (avoid flooding)
auto_pulled_hadiths = auto_pulled_hadiths[:5]
```

**Note** : ces hadiths sont **garantis d'être présentés au LLM**, indépendamment de Phase B. C'est la chaîne Quran → Tafsir → Hadith.

---

## Step 4 — Phase B : Hadith retrieval

Identique à Phase A mais sur `hadith_v3` :

### Sub-step 4a-4d : Multi-query dense + sparse + RRF + rerank

(Pareil que Phase A, mais sur hadith_v3 collection)

### Sub-step 4e : Authenticity weighting (Pillar 3)

```python
# ICI c'est important — grade_weight pour hadiths
for chunk, score in top_5_hadith:
    chunk["final_score"] = score * chunk["metadata"]["grade_weight"]
    # Sahih: 1.30 (boost), Hasan: 1.10, Da'if: 0.50 (penalty), Mawdu: 0 (excluded)
```

### Sub-step 4f : Confidence score Phase B

```python
confidence_B = max(c["rerank_score"] * c["metadata"]["grade_weight"] 
                   for c in top_5_hadith)

if confidence_B >= 0.5:
    phase_b_status = "STRONG"
elif confidence_B > 0.3:
    phase_b_status = "WEAK"
else:
    phase_b_status = "EMPTY"
```

---

## Step 5 — Reporter LLM

### Input du Reporter

```xml
<user_question>Est-ce que fumer est haram ?</user_question>
<phase_a_status>STRONG</phase_a_status>
<phase_a_confidence>0.78</phase_a_confidence>
<phase_b_status>WEAK</phase_b_status>
<phase_b_confidence>0.42</phase_b_confidence>

<quran_chunks>
  <document id="S1">
    <source_id>SRC-QURAN-2-195</source_id>
    <label>Quran 2:195</label>
    <arabic>وَأَنفِقُوا۟ فِي سَبِيلِ ٱللَّهِ...</arabic>
    <english>And spend in the way of Allah...</english>
    <context_card>
      [FR] Thème: Préservation de la vie. Règle: Ne pas se nuire à soi-même.
      [EN] Topic: Self-preservation. Rule: Do not harm oneself.
      Keywords: self-harm, destruction, تهلكة, ضرر
    </context_card>
    <tafsirs>
      <tafsir source="Ibn Kathir" category="bil-Mathur" language="en">
        Ibn Kathir explains: "Do not throw yourselves into destruction"...
      </tafsir>
      <tafsir source="Al-Tabari" category="bil-Mathur" language="ar">
        قال الطبري: ولا تلقوا بأيديكم إلى التهلكة...
      </tafsir>
      <tafsir source="As-Sa'di" category="modern" language="ar">
        يقول السعدي: وهذا نهي عن الإلقاء بالنفس إلى التهلكة...
      </tafsir>
    </tafsirs>
    <ikhtilaf detected="false" />
    <url>https://quran.com/2/195</url>
  </document>
  <!-- S2, S3, S4, S5 similarly -->
</quran_chunks>

<hadith_chunks>
  <document id="S6" auto_pulled="true">
    <source_id>SRC-HADITH-IBNMAJAH-1234</source_id>
    <label>Sunan Ibn Majah #1234</label>
    <grade>Hasan</grade>
    <arabic>...</arabic>
    <english>No harm and no harming in Islam...</english>
    <note>Auto-pulled because cited in Tafsir Ibn Kathir for Quran 2:195</note>
    <url>https://sunnah.com/ibnmajah:1234</url>
  </document>
  <!-- S7, S8, ... -->
</hadith_chunks>

<instructions>
1. Answer in the user's language (FR detected).
2. Cite sources with [S1], [S2], etc. format.
3. If phase_a_status = STRONG: lead with the Quran evidence.
4. If phase_a_status = EMPTY: start with "Le Quran n'aborde pas directement ce sujet."
5. If ikhtilaf detected: present BOTH views, mention Ikhtilaf, suggest consulting scholar.
6. If a tafsir has contains_isra_iliyyat: include the disclaimer.
7. NEVER invent a verse. Only cite [Sn] from the provided sources.
8. Arabic text of Quran must be EXACT (we'll verify post-gen).

Output format: JSON with:
{
  "answer_fr": "...",
  "answer_en": "...",
  "citations": [
    {"source_id": "SRC-QURAN-2-195", "label": "Quran 2:195", 
     "arabic": "...", "english": "...", "tafsir_used": "Ibn Kathir"},
    ...
  ],
  "ikhtilaf": false,
  "confidence": "high" | "medium" | "low"
}
</instructions>
```

### Modèle

- **Modèle** : `meta-llama/llama-4-scout-17b-16e-instruct` (Groq)
- **Temperature** : 0.0
- **Max tokens** : 2048
- **Fallback** : `llama-3.3-70b-versatile` (pour Ikhtilaf complexe)
- **Offline fallback** : Ollama `llama3.1:8b`

---

## Step 6 — Verification (Pillar 4)

### NLI Verification

Pour chaque phrase de la réponse, on vérifie qu'elle est entailed par au moins un chunk source :

```python
for sentence in answer_sentences:
    max_nli = 0
    for chunk in all_chunks:
        nli_score = nli_model.predict([
            {"premise": chunk["text"], "hypothesis": sentence}
        ])
        max_nli = max(max_nli, nli_score["entailment"])
    
    if max_nli < NLI_THRESHOLD:  # 0.95
        # Sentence not entailed by any chunk → potential hallucination
        verification_errors.append({
            "sentence": sentence,
            "max_nli": max_nli,
            "issue": "not_entailed"
        })
```

### Quran character verification

Pour chaque citation Quran :

```python
for citation in response["citations"]:
    if citation["source_id"].startswith("SRC-QURAN"):
        # Get the original text from quran_v3
        original = quran_v3.get(id=citation["source_id"])["metadata"]["text_ar"]
        
        # Normalize both (tashkeel, etc.)
        norm_cited = normalize_arabic(citation["arabic"])
        norm_original = normalize_arabic(original)
        
        if norm_cited != norm_original:
            verification_errors.append({
                "citation": citation["source_id"],
                "issue": "quran_text_mismatch",
                "expected": original,
                "got": citation["arabic"]
            })
```

### Source ID validation

```python
# Vérifier que tous les [Sn] dans la réponse existent dans les chunks fournis
cited_ids = parse_source_ids(response)  # [1, 2, 5, 6]
provided_ids = list(range(1, len(all_chunks) + 1))

invalid_ids = set(cited_ids) - set(provided_ids)
if invalid_ids:
    verification_errors.append({
        "issue": "invalid_source_ids",
        "invalid": list(invalid_ids)
    })
```

### Si verification_errors non vide

- **Si erreur critique** (Quran text mismatch OU invalid source IDs) : rejeter la réponse, rerun avec un prompt plus strict
- **Si erreur mineure** (NLI < 0.95 sur 1-2 phrases) : ajouter une note de prudence à la réponse finale

---

## Tableau récapitulatif des thresholds

| Threshold | Valeur | Rationale |
|-----------|--------|-----------|
| RRF K | 25 | Standard RRF (Cormack 2009) |
| RRF alpha dense | 0.4 | Dense légèrement sous-weighté vs sparse (DEC-034) |
| top_k_initial (pool) | 400 | DEC-034: pool=400 reaches 50% recall |
| top_k_rerank | 5 | 5 chunks × ~1500 tokens = 7500 tokens (within Groq 30K TPM) |
| Confidence A threshold (STRONG) | 0.5 | bge-reranker normalized score |
| Confidence A threshold (WEAK) | 0.3 | Below = EMPTY |
| Confidence B threshold (STRONG) | 0.5 | Same as A |
| NLI threshold | 0.95 | Pillar 4 strict |
| Ikhtilaf detection threshold | 0.7 | NLI contradiction score |
| Auto-pulled hadiths max | 5 | Avoid flooding the prompt |

---

## Exemple complet — "Est-ce que fumer est haram ?"

```
Step 1 — Architect:
  Q1: "smoking haram islam"
  Q2: "self-harm forbidden quran"
  Q3: "destroying oneself quran"
  Q4: "wasteful spending quran"
  Q5: "tobacco ruling islam"

Step 2 — Phase A (Quran+Tafsir):
  Top-5 after rerank:
    S1: SRC-QURAN-2-195  (score 0.78) ← self-harm/destruction
    S2: SRC-QURAN-4-29   (score 0.72) ← kill yourselves
    S3: SRC-QURAN-17-27  (score 0.65) ← wasteful (israf)
    S4: SRC-QURAN-5-90   (score 0.58) ← intoxicants (khamr)
    S5: SRC-QURAN-7-157  (score 0.52) ← forbids bad things
  
  Confidence A = 0.78 → STRONG
  Phase B normal weight

Step 3 — Auto-pull hadiths:
  For S1 (2:195): cross_refs = [SRC-HADITH-IBNMAJAH-1234]
    → auto-pull Ibn Majah #1234 "La darar wa la dirar"
  For S3 (17:27): cross_refs = [SRC-HADITH-BUKHARI-5225]
    → auto-pull Bukhari #5225 (wasteful spending)

Step 4 — Phase B (Hadith):
  Top-5 after rerank:
    S6: SRC-HADITH-BUKHARI-5550  (score 0.71) ← "do not cause harm"
    S7: SRC-HADITH-MUSLIM-3221   (score 0.65) ← "your body has rights"
    S8: SRC-HADITH-IBNMAJAH-1234 (auto-pulled, score 0.62)
    S9: SRC-HADITH-TIRMIDHI-740  (score 0.55) ← "no harm"
    S10: SRC-HADITH-BUKHARI-5225 (auto-pulled, score 0.51)
  
  Confidence B = 0.71 × 1.30 (Sahih) = 0.92 → STRONG

Step 5 — Reporter LLM:
  Input: 5 Quran chunks + 5 hadith chunks (2 auto-pulled)
  Output JSON: {
    "answer_fr": "Le tabac n'est pas mentionné nommément dans le Coran,
    mais plusieurs versets permettent de déduire son interdiction. 
    [S1] (2:195) interdit de se jeter dans la destruction — or le tabac 
    cause un préjudice prouvé au corps. [S2] (4:29) renforce ce principe.
    [S4] (5:90) interdit les intoxicants — les savants classent souvent 
    le tabac dans cette catégorie par analogie (qiyas).
    
    Les hadiths corroborent : [S6] (Bukhari) et [S9] (Tirmidhi) 
    interdisent de causer du tort. [S8] (Ibn Majah) cite directement 
    le principe 'La darar wa la dirar'.
    
    Conclusion : la majorité des savants contemporains (y compris Ibn 
    Baz, Ibn Uthaymeen) classent le tabac comme haram, par analogie 
    avec les versets sur la préservation de la vie et l'interdiction 
    de se nuire. Il s'agit d'un consensus moderne (ijma' contemporain).",
    
    "citations": [
      {"source_id": "SRC-QURAN-2-195", "label": "Quran 2:195", ...},
      ...
    ],
    "ikhtilaf": false,
    "confidence": "high"
  }

Step 6 — Verification:
  NLI check: all sentences entailed by chunks ✓
  Quran char check: AR text matches ✓
  Source IDs: [S1, S2, S4, S6, S8, S9] all valid ✓

→ Final answer sent to user.
```

---

## Prochain document

→ `05_EMBEDDING_DESIGN.md` : **analyse complète et détaillée** de la truncation tafsir (point 2 que tu as validé). Étude comparative 400/600/800/1200/2000 chars avec mesures.
