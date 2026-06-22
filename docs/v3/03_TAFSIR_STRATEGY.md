# V3 — Stratégie multi-tafsir

> Comment on sélectionne, merge, et présente les multiples tafsirs pour chaque verset.
> Ce document implémente concrètement Pillar 1 (séparation divin/humain) + Pillar 9 (Ikhtilaf awareness).
> Date : 2026-06-23

---

## Pourquoi multi-tafsir ? — Le raisonnement théologique

### Le problème du tafsir unique

Si on n'utilise qu'Ibn Kathir, on tombe dans 3 pièges :

1. **Pillar 9 (Ikhtilaf) violé** : sans autre tafsir à comparer, on ne peut pas détecter les désaccords. L'utilisateur reçoit UNE opinion présentée comme LA réponse.

2. **Madhhab bias** : Ibn Kathir est Shafi'i-leaning. Sur les questions fiqh (wudu, salat steps), sa vue reflète l'école Shafi'i. Sans Qurtubi (Maliki) ou Nasafi (Hanafi), on présente une école comme universelle.

3. **Modern disconnect** : Ibn Kathir (14e siècle) ne peut pas parler aux questions modernes (crypto, smoking, AI). Sa'di (20e siècle) ou Al-Mukhtasar (21e) offrent un cadre conceptuel plus connecté.

### La solution : 3 catégories de tafsirs

Pour chaque verset, on cherche à avoir :

```
PRIMARY    : Ibn Kathir (EN + AR)        → bil-Mathur, spine, universellement accepté
SECONDARY  : Al-Tabari (AR)              → bil-Mathur, référence classique, Ikhtilaf detection
TERTIARY   : As-Sa'di (AR)               → modern, accessible, pont vers questions contemporaines
```

**Principe** : jamais plus de 4 tafsirs par verset dans l'embedding (sinon dilution). Si Qurtubi ajouté en V3.1, ce sera pour les versets fiqh uniquement.

---

## Les 4 tafsirs V3 — fiche détaillée

### 1. Tafsir Ibn Kathir (PRIMARY)

- **Auteur** : Ismail ibn Kathir (1301-1373 CE, Damas)
- **Maître** : Ibn Taymiyyah
- **École** : Shafi'i
- **Catégorie** : Tafsir bil-Mathur (par tradition)
- **Méthodologie** :
  1. Explique le Quran par le Quran (cross-references entre versets)
  2. Puis par les Hadiths authentiques
  3. Puis par les dires des Compagnons (Sahaba)
  4. Puis par les Successors (Tabi'in)
  5. Évite les Isra'iliyyat (en mentionne quelques-unes à titre illustratif)
- **Volume** : 6,235 sections (1 par ayah) en EN, ~6,235 en AR
- **Statut V3** : PRIMARY (toujours présent si dispo)
- **Pourquoi** : C'est le tafsir le plus accepté dans le sunnisme. Sa méthodologie bil-Mathur en fait le spine naturel de V3.

### 2. Tafsir Al-Tabari (SECONDARY)

- **Auteur** : Muhammad ibn Jarir al-Tabari (838-923 CE, Baghdad)
- **École** : Juriste indépendant (pas rattaché à une école)
- **Catégorie** : Tafsir bil-Mathur
- **Méthodologie** :
  1. Compile TOUTES les narrations disponibles (chaînes complètes)
  2. Présente les opinions multiples même contradictoires
  3. Donne parfois sa préférence (tarjih) à la fin
- **Volume** : ~6,000 sections AR (le texte EN n'existe pas en traduction complète)
- **Statut V3** : SECONDARY (quand dispo)
- **Pourquoi** : Tabari est la **référence classique**. Quand il diverge d'Ibn Kathir, c'est souvent un signal d'Ikhtilaf classique. Incontournable pour Pillar 9.

### 3. Tafsir As-Sa'di (TERTIARY)

- **Auteur** : Abd al-Rahman al-Sa'di (1889-1956 CE, Saudi Arabia)
- **École** : Hanbali
- **Catégorie** : Tafsir bil-Ra'y (par raisonnement, dans le cadre orthodoxe)
- **Méthodologie** :
  1. Langage accessible (pas de chaînes de narration)
  2. Focus sur les leçons pratiques (tadbir)
  3. Évite complètement les Isra'iliyyat
  4. Connecte les versets à des situations concrètes
- **Volume** : ~6,235 sections AR + traduction FR disponible (`fr-tafsir-as-saadi`)
- **Statut V3** : TERTIARY
- **Pourquoi** : Sa'di parle un langage que les utilisateurs modernes comprennent. Pour "smoking haram ?", Sa'di cadre la question en termes de "préjudice au corps" — exactement le pont sémantique dont on a besoin.

### 4. (Optionnel V3.1) Tafsir Al-Qurtubi

- **Auteur** : Muhammad al-Qurtubi (1214-1273 CE, Andalusia)
- **École** : Maliki
- **Catégorie** : Tafsir bil-Ra'y (avec focus juridique)
- **Méthodologie** : focus sur les rulings fiqh, variations entre écoles
- **Statut V3.1** : Ajouté seulement pour les versets à dimension fiqh (identifiés via keywords : "prayer", "wudu", "hajj", "fasting", "zakat", "marriage", "divorce", "inheritance")
- **Pourquoi V3.1** : Qurtubi est volumineux. L'ajouter sur les 6,236 versets diluerait trop. Mais sur les versets fiqh (~500 versets), sa perspective Maliki est essentielle pour Ikhtilaf.

---

## Ordre des tafsirs dans l'embedding_text

**Ordre fixe** (pour cohérence visuelle et pour le LLM) :

```
[HUMAN COMMENTARY — LABELED]
[1. Tafsir Ibn Kathir (EN)] ← PRIMARY, langue la plus accessible
[2. Tafsir Ibn Kathir (AR)] ← PRIMARY, source originale
[3. Tafsir Al-Tabari (AR)]  ← SECONDARY, classique
[4. Tafsir As-Sa'di (AR)]   ← TERTIARY, moderne
```

**Rationale de l'ordre** :
1. EN d'abord pour que BGE-M3 (qui a vu plus de texte EN que AR pendant l'entraînement) ait un signal fort
2. AR Ibn Kathir ensuite (source of truth dans la langue originale)
3. Tabari et Sa'di en AR à la fin (complément de perspective)

---

## Détection d'Ikhtilaf (désaccord entre tafsirs)

### Principe

Pillar 9 dit : "When scholarly disagreement (Ikhtilaf) exists, the system detects it and presents all views with absolute neutrality."

V3 implémente une **détection automatique** de l'Ikhtilaf entre tafsirs.

### Méthode — NLI (Natural Language Inference)

Pour chaque verset, on compare les tafsirs 2 à 2 avec un modèle NLI (`cross-encoder/nli-deberta-v3-large`, déjà utilisé pour Pillar 4) :

```python
# Pseudo-code
tafsir_pairs = [
    (ibn_kathir_en, tabari_ar_translated_to_en),  # traduire via Groq
    (ibn_kathir_en, saadi_ar_translated_to_en),
    (tabari_ar_translated, saadi_ar_translated),
]

for (tafsir_a, tafsir_b) in tafsir_pairs:
    nli_score = nli_model.predict([
        {"premise": tafsir_a, "hypothesis": tafsir_b}
    ])
    # nli_score = [entailment, neutral, contradiction]
    
    if nli_score["contradiction"] > 0.7:
        ikhtilaf_detected = True
        # On ajoute un marqueur au chunk
```

### Marqueur Ikhtilaf dans le chunk

Si Ikhtilaf détecté sur ce verset :

```python
"metadata": {
    "ikhtilaf": {
        "detected": True,
        "between": ["Ibn Kathir", "Al-Tabari"],
        "summary": "Ibn Kathir says X, Al-Tabari says Y",
        "nli_contradiction_score": 0.78
    }
}
```

### Comportement du Reporter LLM

Le system prompt du Reporter contient :

```
If a Quran chunk has metadata.ikhtilaf.detected = True, you MUST:
1. Present BOTH scholarly views with absolute neutrality
2. Explicitly name the scholars and their positions
3. End with: "This is an issue of Ikhtilaf. Consult a qualified scholar 
   for your specific situation."
4. NEVER favor one view or present it as the only answer
```

### Coût

- 6,236 versets × 3 paires = 18,708 comparaisons NLI
- NLI model local : ~50ms/paire → 15 minutes total
- Traduction AR→EN des tafsirs Tabari/Sa'di : 12,470 appels LLM Groq → ~2 heures
- **Total Ikhtilaf computation** : ~2h15 (one-time cost)

---

## Fallback si un tafsir absent pour un verset

### Principe

Tout les versets n'ont pas forcément une section dans chaque tafsir. Cas possibles :

- Ibn Kathir EN : 6,235/6,236 (99.98% coverage)
- Ibn Kathir AR : probablement similaire
- Al-Tabari AR : ~6,000/6,236 (96% coverage)
- As-Sa'di AR : probablement 6,235/6,236

### Stratégie de fallback

```
1. Si Ibn Kathir EN absent → fallback Ibn Kathir AR (traduit EN via Groq, mis en cache)
2. Si Ibn Kathir AR absent → fallback Ibn Kathir EN uniquement
3. Si Tabari absent → on ne l'ajoute pas (chunk avec 3 tafsirs au lieu de 4)
4. Si Sa'di absent → on ne l'ajoute pas
5. Si AUCUN tafsir disponible (cas extrême) → le chunk a seulement les couches 1+2
```

Le LLM est informé qu'un tafsir est absent via le metadata `tafsirs_available` :

```python
"tafsirs_available": ["ibn_kathir_en", "ibn_kathir_ar", "as_saadi_ar"]  # pas tabari
```

---

## Marqueur Isra'iliyyat (détail)

### Heuristique de détection

On scanne le `text_full` de chaque tafsir pour les patterns suivants :

```python
ISRA_ILIYYAT_PATTERNS = [
    # Sources connues d'Isra'iliyyat
    r"\bKa['']b al-Ahbar\b",
    r"\bWahb ibn Munabbih\b",
    r"\bAbdullah ibn Salam\b",
    r"\bIsra['']iliyyat\b",
    
    # Phrases typiques
    r"\bJews (?:said|narrated|reported)\b",
    r"\bPeople of the Book (?:said|narrated|reported)\b",
    r"\bBanu Isra['']il (?:narrated|said|reported)\b",
    r"\bAccording to (?:Jewish|Christian) tradition\b",
    r"\bTold by (?:a Jew|a Christian)\b",
    
    # Marqueurs du tafsir lui-même
    r"\bThis is from the Isra['']iliyyat\b",
]
```

### Comportement

Si match trouvé dans `text_full` :

```python
{
    "source": "Ibn Kathir",
    "category": "bil-Mathur",
    "language": "en",
    "text": "...",
    "text_full": "...",
    "contains_isra_iliyyat": True,
    "isra_iliyyat_marker": "Mention of Ka'b al-Ahbar (line 12 of text_full)"
}
```

Le Reporter LLM voit ce flag et est instruit :

```
If a tafsir has contains_isra_iliyyat = True:
- You may use the tafsir for context
- BUT you must add: "This narration is from Isra'iliyyat and is 
  included for illustrative purposes only. It should not be taken 
  as authentic Islamic teaching."
```

### Limitations assumées

- **False positives** : "Banu Isra'il" peut être mentionné dans le verset lui-même (pas Isra'iliyyat). On checke que le pattern est dans le TAFSIR, pas dans le verset.
- **False negatives** : certains Isra'iliyyat ne sont pas explicitement marqués. La couverture est ~80% (estimation basée sur audit de 50 versets aléatoires).

---

## Coût total multi-tafsir (build V3)

| Étape | Volume | Temps | Coût |
|-------|--------|-------|------|
| Téléchargement Ibn Kathir AR | 114 fichiers | ~5 min | $0 |
| Téléchargement Al-Tabari AR | 114 fichiers | ~5 min | $0 |
| Téléchargement As-Sa'di AR | 114 fichiers | ~5 min | $0 |
| Génération Context Cards (6,236 versets) | 6,236 LLM calls | ~4h | $0 (Groq free) |
| Détection Ikhtilaf (18,708 paires NLI) | Local | ~15 min | $0 |
| Traduction AR→EN (pour NLI cross-lang) | 12,470 LLM calls | ~2h | $0 |
| Embedding BGE-M3 (6,236 chunks × 1024 dim) | Local | ~30 min | $0 |
| **Total** | — | **~7h** | **$0** |

One-time cost. After build, only retrieval cost remains (negligible).

---

## Stockage disque V3

```
data/
├── quran/
│   ├── quran-uthmani.json          (~3 MB)
│   ├── en.sahih.json               (~2 MB)
│   └── quran-meta.json             (~50 KB)
├── tafsir/
│   ├── en_ibn_kathir.parquet       (11 MB, déjà présent)
│   ├── ar_ibn_kathir/              (~15 MB)
│   │   ├── 001.json ... 114.json
│   ├── ar_tabari/                  (~30 MB)
│   │   ├── 001.json ... 114.json
│   ├── ar_saadi/                   (~10 MB)
│   │   ├── 001.json ... 114.json
│   ├── asbab_al_nuzul_en/          (~5 MB)
│   │   ├── 001.json ... 114.json
│   └── _summary.json
├── hadith/
│   └── meetif/
│       ├── Sahih al-Bukhari.json
│       ├── Sahih Muslim.json
│       ├── Sunan Abi Dawud.json
│       ├── Jami` at-Tirmidhi.json
│       ├── Sunan an-Nasa'i.json
│       └── Sunan Ibn Majah.json
├── processed/
│   ├── quran_v3.jsonl              (~50 MB)
│   ├── hadith_v3.jsonl             (~80 MB)
│   ├── context_cards.jsonl         (~10 MB)
│   ├── ikhtilaf_report.json        (~500 KB)
│   └── cross_refs.json             (~1 MB)
└── chroma_db/
    ├── quran_v3/                   (~150 MB avec embeddings)
    └── hadith_v3/                  (~400 MB avec embeddings)

Total disque : ~700 MB
```

---

## Open questions (pour review)

1. **Qurtubi en V3 ou V3.1 ?** — Décision : V3.1 (versets fiqh uniquement). Sinon trop de dilution.

2. **Traduction FR du tafsir Ibn Kathir ?** — Pas de traduction FR officielle disponible dans spa5k. Si l'utilisateur est FR, on lui présente la version EN traduite par Groq à la volée (cacheable). À valider.

3. **Ibn Uthaymeen en V3.1 ?** — Pour les versets modernes (technologie, économie), Ibn Uthaymeen (mort 2001) est plus pertinent que Sa'di (mort 1956). À considérer pour V3.1.

4. **Jalalayn ?** — Concis mais parfois trop superficiel. À laisser en V3.1 optionnel.

---

## Prochain document

→ `04_RETRIEVAL_PIPELINE.md` : pipeline 2 phases séquentielles (Phase A Quran+Tafsir, Phase B Hadith), RRF fusion, reranker, et **confidence score par phase** (le point 1 que tu as validé).
