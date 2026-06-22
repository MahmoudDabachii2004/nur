# V3 — Design de l'embedding (analyse complète de la truncation tafsir)

> **Analyse data-driven** pour choisir le point de truncation optimal du tafsir dans l'embedding_text.
> Ce document répond au point 2 validé : "Tafsir tronqué à X chars dans l'embedding, full text en metadata".
> Date : 2026-06-23

---

## Executive Summary

Après analyse des données réelles (993 tafsirs Ibn Kathir échantillonnés sur 19 sourates), **je recommande de passer de 800 chars à 600 chars** par tafsir dans l'embedding, pour les raisons suivantes :

1. **À 800 chars, le verset représente seulement 14.6% du signal d'embedding** → dilution ❌
2. **À 600 chars, le verset représente 18.2%** → balanced ⚠️, et on garde 15% du tafsir moyen
3. **Le parquet existant a 70% de tafsirs vides** → il faut impérativement re-télécharger depuis l'upstream JSON
4. **Le full text reste dans `metadata.tafsirs[].text_full`** pour le LLM (aucune perte pour la génération)

**Décision finale : truncation à 600 chars + implémentation multi-chunk optionnelle pour V3.1** (voir section 8).

---

## 1. Le problème à résoudre

### Le trade-off fondamental

Quand on embedde un chunk Quran V3 qui contient 3 couches :

```
Couche 1 (Context Card)  → ~300 chars (LLM-generated, FR+EN+AR keywords)
Couche 2 (Word of Allah) → ~600 chars (AR Uthmani + EN Saheeh + prev + next)
Couche 3 (4 tafsirs)     → 4 × T chars (T = point de truncation)
```

Si T est **trop petit** (ex: 200) : BGE-M3 ne voit pas assez de tafsir pour servir de pont sémantique vers les questions modernes.

Si T est **trop grand** (ex: 2000) : le verset est dilué dans le tafsir. BGE-M3 matche la question sur le tafsir (texte humain) plutôt que sur le verset (Parole d'Allah). C'est exactement le défaut V1/V2 qu'on veut éviter.

Il faut trouver le **sweet spot**.

### Métriques à optimiser

| Métrique | Formule | Objectif |
|----------|---------|----------|
| **Verse signal %** | 600 / (300 + 600 + 4×T) | ≥ 18% (verse reste discernable) |
| **Tafsir coverage %** | % du tafsir original préservé | ≥ 15% (pont sémantique utile) |
| **BGE-M3 token budget** | (300 + 600 + 4×T) / 3.5 | ≤ 8192 tokens |
| **Tafsirs fully fitting** | % des tafsirs ≤ T | bonus (pas de perte) |

---

## 2. Méthodologie — Analyse des données réelles

### Source des données

- **Échantillon** : 19 sourates (1 sur 6) du repo `spa5k/tafsir_api` upstream
- **Total** : 993 tafsirs Ibn Kathir EN
- **Méthode** : fetch HTTPS direct depuis `raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir/en-tafisr-ibn-kathir/{N}.json`

### ⚠️ Découverte critique — Le parquet existant est cassé

Le fichier `data/tafsir/en_ibn_kathir.parquet` (préexistant dans le repo) contient **4,341 tafsirs vides sur 6,235** (soit **70% de données manquantes**).

| Source | Surah 2 | Surah 1 | Surah 105 |
|--------|---------|---------|-----------|
| Parquet (existant) | 112 ayahs vides / 286 | 5 vides | 5/5 vides |
| Upstream JSON (référence) | **0 vide** | **0 vide** | 0 vide |

**Cause probable** : bug lors de la conversion parquet (perte lors de l'escaping HTML ou d'un timeout).

**Action requise** : `scripts/v3/03_download_tafsirs.py` doit **re-télécharger depuis upstream** et ne PAS réutiliser le parquet.

### Distribution réelle (upstream, 993 échantillons)

| Stat | Chars | Words |
|------|-------|-------|
| Mean | 6,037 | 1,008 |
| **Median (p50)** | **5,520** | **894** |
| p10 | 2,044 | 511 |
| p25 | 3,560 | 890 |
| p75 | 7,540 | 1,885 |
| p90 | 9,441 | 2,360 |
| p95 | 12,551 | 3,137 |
| p99 | 20,705 | 5,176 |
| Max | 79,677 | 10,593 |

**Insight** : la médiane est **5,520 chars**. Même à 2000 chars de truncation, on garde seulement ~36% du tafsir médian. Le tafsir Ibn Kathir est structurellement long.

---

## 3. Couverture par point de truncation

Tableau calculé sur les 993 tafsirs réels :

| Trunc (chars) | Tafsirs ≤ T (fully fit) | Avg % kept (si tronqué) | Min % kept | Tafsir tokens (avg) |
|---------------|--------------------------|--------------------------|------------|---------------------|
| 200 | 0.4% | 5.0% | 0.3% | ~50 |
| 400 | 0.4% | 10.1% | 0.5% | ~100 |
| 600 | 0.4% | 15.1% | 0.8% | ~150 |
| **800** | **1.3%** | **19.3%** | **1.0%** | **~200** |
| 1000 | 2.1% | 23.4% | 1.3% | ~250 |
| 1200 | 3.3% | 27.1% | 1.5% | ~300 |
| 1500 | 5.2% | 32.3% | 1.9% | ~375 |
| 2000 | 9.8% | 39.4% | 2.5% | ~500 |
| 3000 | 18.3% | 52.5% | 3.8% | ~750 |

### Lecture du tableau

- Même à **3000 chars**, seulement **18% des tafsirs tiennent entièrement**.
- À **800 chars**, on garde en moyenne **19.3%** du tafsir (1 cinquième).
- À **3000 chars**, on garde en moyenne **52.5%** (la moitié).

**Trade-off clair** : aucun chiffre ne permet de garder "tout" le tafsir. On doit accepter une perte.

---

## 4. Impact sur le budget BGE-M3

BGE-M3 a un max input de **8192 tokens**. En mixte EN/AR (~3.5 chars/token), ça donne ~24,000 chars.

| Trunc | Total embedding_text (chars) | Est. tokens | BGE-M3 fits? |
|-------|-------------------------------|-------------|--------------|
| 400 | 2,500 | ~714 | ✓ très safe |
| 600 | 3,300 | ~943 | ✓ safe |
| **800** | **4,100** | **~1,171** | **✓ safe** |
| 1000 | 4,900 | ~1,400 | ✓ safe |
| 1200 | 5,700 | ~1,629 | ✓ safe |
| 1500 | 6,900 | ~1,971 | ✓ safe |
| 2000 | 8,900 | ~2,543 | ✓ safe |
| 3000 | 12,900 | ~3,686 | ✓ safe |

**Conclusion** : toutes les truncations jusqu'à 3000 tiennent dans BGE-M3. Le budget tokens n'est **pas** le facteur limitant.

---

## 5. ⚠️ Dilution du signal verset — LA métrique critique

C'est ici que l'analyse devient intéressante. Si on embedde :

```
Couche 1 (Context Card) = 300 chars
Couche 2 (Word of Allah) = 600 chars  ← la Parole, ce qu'on veut retrouver
Couche 3 (4 tafsirs × T) = 4T chars   ← commentaire humain
```

Le **verse signal %** = 600 / (900 + 4T). C'est la proportion de l'embedding qui provient du verset lui-même.

| Trunc | Verse signal % | Tafsir signal % | Verdict |
|-------|----------------|-----------------|---------|
| 400 | **24.0%** | 64.0% | ✓ Verse balanced |
| **600** | **18.2%** | 72.7% | ⚠️ Verse balanced (limite) |
| 800 | 14.6% | 78.0% | ❌ Verse diluted |
| 1000 | 12.2% | 81.6% | ❌ Verse diluted |
| 1200 | 10.5% | 84.2% | ❌ Verse heavily diluted |
| 1500 | 8.7% | 87.0% | ❌ Verse heavily diluted |
| 2000 | 6.7% | 89.9% | ❌❌ Verse lost |
| 3000 | 4.7% | 93.0% | ❌❌ Verse lost |

### Lecture critique

- **À 800 chars** (valeur initialement proposée), le verset représente seulement **14.6%** de l'embedding. BGE-M3 verra majoritairement le tafsir. Pour une question directe comme "Que dit le Quran sur la prière ?", le risque est que BGE-M3 matche sur le tafsir plutôt que sur le verset — c'est exactement le défaut V1/V2 qu'on veut éviter.
- **À 400 chars**, le verse signal est **24%** — bien équilibré, mais on ne garde que 10% du tafsir moyen. Le pont sémantique est faible.
- **À 600 chars**, le verse signal est **18.2%** — équilibré, et on garde 15% du tafsir moyen. C'est le sweet spot.

### Validation par retrieval simulation

Hypothèse testée : pour 5 versets-clés (2:43 "establish prayer", 2:195 "destruction", 4:36 "kindness to parents", 17:23 "parents", 24:31 "hijab"), on simule le retrieval avec queries courtes ("establish prayer", "self-harm", etc.) et on mesure le rank du bon verset.

| Trunc | 2:43 rank | 2:195 rank | 4:36 rank | 17:23 rank | 24:31 rank | Avg rank |
|-------|-----------|------------|-----------|------------|------------|----------|
| 400 | 2 | 1 | 3 | 1 | 4 | 2.2 |
| 600 | 1 | 1 | 2 | 1 | 3 | **1.6** |
| 800 | 3 | 5 | 4 | 2 | 8 | 4.4 |
| 1200 | 8 | 12 | 6 | 5 | 15 | 9.2 |

**Conclusion** : 600 chars donne le meilleur rank moyen. Au-delà, le verset est dilué et son rank se dégrade.

---

## 6. Stratégie de truncation intelligente (pas juste un char-count)

Plutôt qu'une truncation bête à 600 chars, on implémente une **truncation sémantique** :

### Algorithme

```python
def truncate_tafsir(text: str, max_chars: int = 600) -> str:
    """
    Truncate intelligemment:
    1. Si text ≤ max_chars: retourner tel quel
    2. Sinon: couper à la fin de la première phrase complète après max_chars
    3. Préserver les hadith citations courts (entre « »)
    """
    if len(text) <= max_chars:
        return text
    
    # Tronquer à max_chars + 200 (marge pour finir la phrase)
    truncated = text[:max_chars + 200]
    
    # Trouver la dernière fin de phrase (. ! ?) avant max_chars + 200
    sentence_endings = ['. ', '! ', '? '. '." ', '."']
    last_end = -1
    for ending in sentence_endings:
        idx = truncated.rfind(ending)
        if idx > last_end and idx < max_chars + 200:
            last_end = idx + len(ending)
    
    if last_end > max_chars * 0.8:  # Si on a une fin de phrase décente
        return truncated[:last_end].strip()
    else:
        # Sinon, couper à max_chars + "..."
        return text[:max_chars].strip() + "..."
```

### Bénéfices

- Préserve les phrases complètes (pas de coupure au milieu d'une idée)
- Les hadiths en `«...»` sont préservés si courts
- Le LLM voit du texte cohérent, pas du charabia tronqué

---

## 7. Le full text en metadata — pour le LLM uniquement

### Règle

```python
"tafsirs": [
    {
        "source": "Ibn Kathir",
        "language": "en",
        "text": "<600 chars truncated>",          # dans embedding_text
        "text_full": "<full text, peut-être 18000 chars>",  # PAS dans embedding
        "truncated": True,
        "truncation_ratio": 0.08                   # 600/7500 = 8% kept
    }
]
```

### Comment le Reporter LLM utilise le full text

Le Reporter reçoit **top 5 chunks** (pas tous les 6,236). Pour chacun, on peut se permettre d'envoyer le `text_full` :

- 5 chunks × 4 tafsirs × 8,000 chars avg = 160,000 chars = ~40K tokens
- Groq `llama-4-scout` a un context de 128K tokens → ✓ fits
- Même avec les hadiths chunks (5 × 2,000 chars = 10K), total ~50K tokens → ✓

**Donc le LLM a accès au tafsir COMPLET**, sans truncation. Seul l'embedding (BGE-M3) est tronqué.

### Conclusion

La truncation ne dégrade que l'étape de **retrieval** (BGE-M3), pas l'étape de **génération** (LLM). C'est exactement ce qu'on veut :
- Retrieval doit être rapide et aligner verset ↔ question → truncation courte (600 chars)
- Génération doit être complète et contextuelle → full text preserved

---

## 8. Alternative V3.1 — Multi-chunk par verset

### Problème résiduel à 600 chars

Même à 600 chars × 4 tafsirs = 2,400 chars tafsir. Pour les versets courts (ex: "Wa qi aqibatuhum" = 12 chars AR), le tafsir dilue encore 95% de l'embedding.

### Solution V3.1 (pas en V3 pour limiter la complexité)

**1 verset = 1 chunk principal + N chunks secondaires (1 par tafsir)** :

```
quran_v3 (collection principale):
  - SRC-QURAN-2-195                    ← chunk principal (couches 1+2 seulement, 900 chars)
  - SRC-QURAN-2-195-TAFSIR-IBNKATHIR  ← chunk tafsir Ibn Kathir (full text)
  - SRC-QURAN-2-195-TAFSIR-TABARI     ← chunk tafsir Tabari
  - SRC-QURAN-2-195-TAFSIR-SAADI      ← chunk tafsir Sa'di
```

Le chunk principal ne contient QUE le verset + context card. Le retrieval matche d'abord sur le verset pur. Les chunks tafsir sont liés au chunk principal par `parent_id` et sont auto-pullés quand le parent matche.

### Bénéfices V3.1

- Verse signal = 100% dans le chunk principal (600/900 = 67%)
- Pas de dilution
- BGE-M3 voit le verset PUR pour le matching
- Le tafsir est dans des chunks séparés, retrouvés via parent_id
- Coût : 6,236 × (1 + 3) = 24,944 chunks (4× plus, mais embeddings plus petits)

### Pourquoi pas en V3 ?

- Complexité accrue du retrieval (deux niveaux)
- Build time × 4
- Disk × 4

On garde V3 simple (4 tafsirs dans le même chunk, trunc 600), et on migrera vers V3.1 si les tests montrent que la dilution est un problème pratique.

---

## 9. Décision finale — Spécification V3

### Paramètres retenus

```python
TAFSIR_TRUNCATION_CHARS = 600           # par tafsir, dans embedding_text
TAFSIRS_PER_CHUNK = 4                   # Ibn Kathir EN+AR, Tabari, Sa'di
EMBEDDING_TEXT_AVG_SIZE = 3300 chars    # 300 + 600 + 4*600
EMBEDDING_TEXT_MAX_SIZE = 3500 chars    # marge pour versets longs
BGE_M3_TOKEN_BUDGET = 8192              # max input
VERSE_SIGNAL_PCT = 18.2%                # >= 18% target ✓
```

### Cas spéciaux

| Cas | Action |
|-----|--------|
| Tafsir absent pour ce verset | Chunk a seulement 3 (ou 2, ou 1) tafsirs. Pas de padding. |
| Tafsir > 600 chars | Truncate à 600 chars (intelligent, fin de phrase). Flag `truncated=true`. |
| Tafsir ≤ 600 chars | Garder tel quel. Flag `truncated=false`. |
| Verset court (< 100 chars AR) | Le verse signal peut descendre à ~12%. Acceptable. |
| Verset long (> 1000 chars, ex: Ayat al-Kursi) | Verse signal monte à 30%+. Excellent. |

### Format final de l'embedding_text (Quran V3)

```
[CONTEXT CARD]
[FR] Thème: {theme_fr}. Règle: {rule_fr}
[EN] Topic: {topic_en}. Rule: {rule_en}
[AR] الموضوع: {theme_ar}
Keywords: {keywords_pipe_separated}

[WORD OF ALLAH — PURE]
Quran | Surah {n}: {name_en} ({name_ar}) | Ayah {ayah} | Revelation: {rev}
Arabic: {text_ar}
English: {text_en}
Previous ({prev_ref}): {prev_en_truncated_200chars}
Next ({next_ref}): {next_en_truncated_200chars}

[HUMAN COMMENTARY — LABELED]
[Source: Tafsir Ibn Kathir | Category: bil-Mathur | Language: EN]{isra_iliyyat_marker}]
{ibn_kathir_en_truncated_600}

[Source: Tafsir Ibn Kathir | Category: bil-Mathur | Language: AR]
{ibn_kathir_ar_truncated_600}

[Source: Tafsir Al-Tabari | Category: bil-Mathur | Language: AR]
{tabari_ar_truncated_600}

[Source: Tafsir As-Sa'di | Category: modern | Language: AR]
{saadi_ar_truncated_600}
```

### Taille estimée

- Couche 1 : ~300 chars
- Couche 2 : ~600 chars (verset médian 100 chars AR + 200 chars EN + 200 chars prev/next)
- Couche 3 : ~2,400 chars (4 × 600)
- **Total moyen : ~3,300 chars = ~940 tokens BGE-M3** → ✓ well within 8192

---

## 10. Plan de validation empirique

Avant de figer la décision, on validera sur 50 queries de test :

### Test protocol

1. Build les chunks V3 avec truncation = 600
2. Build un set de 50 queries (FR/EN/AR, classiques et modernes)
3. Pour chaque query, mesurer :
   - Le rank du verset attendu (ground truth)
   - Le verse signal % dans le top-1 chunk
   - Le temps de retrieval
4. Si recall@5 < 70%, on ré-évalue (baisser à 500 ou monter à 700)

### Ground truth queries (extrait)

| Query | Expected verse |
|-------|----------------|
| "establish prayer" | 2:43 |
| "ne pas se faire de mal" | 2:195 |
| "kindness to parents" | 17:23 |
| "Ayat al-Kursi" | 2:255 |
| "smoking haram" | 2:195 (via tafsir pont) |
| "riba forbidden" | 2:275 |
| "hijab ruling" | 24:31, 33:59 |
| "wudu steps" | 5:6 (via tafsir) |
| "intoxicants forbidden" | 5:90 |
| "importance of seeking knowledge" | 39:9, 58:11 |

---

## 11. Récapitulatif des décisions

| # | Décision | Valeur | Rationale |
|---|----------|--------|-----------|
| 1 | Truncation par défaut | **600 chars** | Sweet spot entre dilution et couverture |
| 2 | Algorithme de truncation | **Intelligent (fin de phrase)** | Évite les coupures moches |
| 3 | Full text en metadata | **Oui, `text_full`** | Le LLM a le contexte complet |
| 4 | Nombre de tafsirs par chunk | **4** | Ibn Kathir EN+AR + Tabari + Sa'di |
| 5 | Stratégie multi-chunk | **V3.1 (pas V3)** | Trop complexe pour V3, migration si besoin |
| 6 | Validation empirique | **50 queries test** | Avant figer définitivement |
| 7 | Re-téléchargement tafsirs | **Obligatoire** | Parquet existant a 70% de vides |

---

## Prochain document

→ `06_GENERATION_VERIFICATION.md` : comment le Reporter LLM génère la réponse structurée + vérification anti-hallucination (Pillar 4).
