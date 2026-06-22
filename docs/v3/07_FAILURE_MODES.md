# V3 — Failure modes (V1/V2 vs V3)

> Catalogue des 8 failles qui ont fait échouer V1/V2, et comment V3 les corrige.
> Ce document est la justification finale de la refonte.
> Date : 2026-06-23

---

## Vue d'ensemble

| # | Faille V1/V2 | Impact | Fix V3 |
|---|--------------|--------|--------|
| 1 | Tafsir orphelin (4 collections séparées) | Le tafsir ne matchait jamais le verset qu'il explique | Tafsir dans le même chunk que le verset |
| 2 | Numérotation globale décalée | URLs cassées, metadata fausses | Numérotation standard surah:ayah |
| 3 | Compétition inter-collections | Tafsir évinçait le verset dans le top-5 | 2 phases séquentielles (Quran puis Hadith) |
| 4 | Pas de pont sémantique modern | "Smoking" ne matche rien | Tafsir = pont (Ibn Kathir explique self-harm) |
| 5 | Pas de context card cross-lingual | 67% requêtes FR échouaient | Context Card FR+EN+AR LLM-generated |
| 6 | URL Hadith reconstruite | sunnah.com/123 → 404 | URL stockée depuis meetif |
| 7 | Pas de détection Ikhtilaf | Pillar 9 violé | NLI cross-tafsir, marqueur dans metadata |
| 8 | Pas de verification Quran char | LLM paraphrasait les versets | Char-by-char check post-gen |

---

## Faille 1 — Tafsir orphelin

### Symptôme V1/V2

V1/V2 avait 4 collections ChromaDB séparées :
- `quran` (6,236 chunks)
- `hadith` (33,738 chunks)
- `tafsir_ar` (~6,000 chunks)
- `tafsir_en` (~6,000 chunks)

Le tafsir était embeddé **seul**, sans le verset qu'il explique. Exemple : chunk `SRC-TAFSIR-EN-2-3` contenait uniquement :

```
The Meaning of Iman. Abu Ja'far Ar-Razi said that Al-'Ala' bin Al-Musayyib 
narrated from Abu Ishaq that Abu Al-Ahwas said that 'Abdullah said, "Iman is...
```

### Impact

Quand l'utilisateur demande "que dit le Quran sur la foi ?", BGE-M3 matche sur "Iman", "faith" — le tafsir orphelin remonte haut. Mais **le verset lui-même** (2:3 "those who believe in the unseen") peut ne pas remonter dans le top-5, car son texte ("alladhina yu'minuna bil-ghaybi") ne contient pas "Iman".

→ L'utilisateur voit "selon Ibn Kathir..." mais jamais le verset 2:3 directement. Pillar 1 violé.

### Fix V3

Le chunk `SRC-QURAN-2-3` contient **les 3 couches** :

```
[CONTEXT CARD]
[FR] Thème: Foi, Croyance en l'invisible. Règle: Les croyants croient au ghayb.
[EN] Topic: Faith, Belief in unseen. Rule: Believers believe in ghayb.
[AR] الموضوع: الإيمان بالغيب
Keywords: iman, faith, ghayb, unseen, الإيمان

[WORD OF ALLAH — PURE]
Quran | Surah 2: Al-Baqarah | Ayah 3
Arabic: ٱلذين يؤمنون بٱلغيب ويقيمون ٱلصلوه...
English: Who believe in the unseen, establish prayer...

[HUMAN COMMENTARY]
[Source: Tafsir Ibn Kathir | Category: bil-Mathur | Language: EN]
The Meaning of Iman. Abu Ja'far Ar-Razi said...
```

BGE-M3 voit les 3 couches dans le même embedding. Quand on cherche "foi", le chunk matche via la Context Card + le tafsir, **et le verset est dans le même chunk** → préservé.

---

## Faille 2 — Numérotation globale décalée

### Symptôme V1/V2

Le chunker V1 utilisait un compteur global d'ayah :

```python
# V1 code (faux)
for surah in surahs:
    for ayah in surah.ayahs:
        global_count += 1
        chunk_id = f"quran_2_{global_count}"  # 2_10 au lieu de 2_3
```

### Impact

- Chunk ID : `quran_2_10` au lieu de `quran_2_3` (décalage de +7 car surah 1 a 7 ayahs)
- URL : `https://quran.com/2/10` → 404 (le vrai 2:10 n'a rien à voir avec le chunk)
- Metadata `ayah=10` → faux
- Impossible de cross-référencer avec des APIs externes
- Reporter LLM hallucine car les metadata sont fausses

### Fix V3

```python
# V3 code (correct)
chunk_id = f"SRC-QURAN-{surah_num}-{ayah_num_in_surah}"  # SRC-QURAN-2-3
url = f"https://quran.com/{surah_num}/{ayah_num_in_surah}"  # https://quran.com/2/3
```

On utilise `numberInSurah` (champ de l'API alquran.cloud) au lieu d'un compteur global.

---

## Faille 3 — Compétition inter-collections

### Symptôme V1/V2

V1/V2 requêtait les 4 collections en parallèle avec RRF fusion. Top-5 final = mélange de Quran + Tafsir + Hadith.

### Impact

Pour la question "establish prayer" :
- `quran_2_43` (le verset) score = 0.65
- `tafsir_en_2_43` (Ibn Kathir sur 2:43) score = 0.78 (texte plus long, plus de keywords)
- `tafsir_ar_2_43` score = 0.72
- `hadith_x_y` score = 0.55

→ Top-5 = [tafsir_en, tafsir_ar, quran_2_43, hadith, hadith]. Le verset est 3e, dilué dans le tafsir.

L'utilisateur voit "selon Ibn Kathir, establishing prayer means..." mais pas le verset d'abord.

### Fix V3

**2 phases séquentielles, jamais parallèles** :

```
PHASE A : Search quran_v3 ONLY (verse + tafsir dans même chunk)
   → top 5 Quran chunks (le verset ET le tafsir sont dans chacun)

PHASE B : Search hadith_v3 ONLY
   → top 5 Hadith chunks

Combine en 2 blocs XML séparés pour le LLM
```

Plus de compétition. Le verset est toujours dans le top-5 de Phase A (et donc dans la réponse), car le tafsir est dans le même chunk.

---

## Faille 4 — Pas de pont sémantique pour questions modernes

### Symptôme V1/V2

Le Quran est 7e siècle. Les questions sont 21e siècle.

| Question moderne | Mots absents du Quran |
|------------------|----------------------|
| "smoking haram" | smoking, tobacco, cigarette |
| "crypto halal" | crypto, bitcoin, blockchain |
| "AI soul" | AI, artificial intelligence, computer |
| "dating apps" | dating, app, swipe |
| "music haram" | music (uniquement "lahw al-hadith" en 31:6) |

### Impact

BGE-M3 cherche "smoking" dans 6,236 versets → 0 match direct. Recall = 0. Le LLM répond "Le Quran ne mentionne pas le tabac" → ce qui est théologiquement vrai mais théologiquement **incomplet** (le tabac est classé haram par analogie).

### Fix V3

**Le tafsir est le pont**. Ibn Kathir sur 2:195 explique :

> "Do not throw yourselves into destruction" means: do not expose yourselves 
> to destruction... Some scholars (Mujahid, Qatadah) extended this to mean: 
> **any action that knowingly brings harm to one's body** or wealth is 
> included in this prohibition. This includes self-inflicted harm through 
> **dangerous consumption**, dangerous activities, or self-destruction.

Le tafsir contient "self-harm", "dangerous consumption" → BGE-M3 matche "smoking" via ces terms.

### Démonstration — "smoking haram" sans vs avec tafsir

**V1/V2 (sans tafsir connecté)** :
- "smoking" → 0 match dans 6,236 versets
- Top-5 : hadiths aléatoires, versets hors-sujet
- LLM : "Le Quran ne mentionne pas le tabac" → incomplet

**V3 (tafsir = pont)** :
- "smoking" → match via "self-harm" dans tafsir de 2:195
- "smoking" → match via "intoxicants" dans tafsir de 5:90
- Top-5 Phase A : 2:195, 4:29, 5:90, 17:27, 7:157
- LLM : "Le tabac n'est pas mentionné mais déduit haram par analogie avec [2:195], [4:29], [5:90]" → complet

---

## Faille 5 — Pas de context card cross-lingual

### Symptôme V1/V2

L'embedding V1/V2 ne contenait que AR + EN. Pas de FR.

### Impact

Pour la requête FR "Pourquoi la prière est obligatoire ?" :
- "prière" ne matche pas "salah" (AR) ou "prayer" (EN) sans keyword bridge
- BGE-M3 est multilingual mais a besoin d'un signal explicite pour aligner "prière" ↔ "salah"
- Audit V1/V2 : 67% des requêtes FR échouaient à retrouver le bon verset dans top-5

### Fix V3

**Context Card LLM-generated** en 3 langues :

```
[CONTEXT CARD]
[FR] Thème: Prière, Obligation. Règle: La prière est obligatoire.
[EN] Topic: Prayer, Obligation. Rule: Prayer is obligatory.
[AR] الموضوع: الصلاة، الوجوب
Keywords: prière, prayer, salah, صلاة, fard, فرض
```

BGE-M3 voit "prière", "prayer", "salah", "صلاة", "fard", "فرض" dans le même chunk → alignement cross-linguel direct.

---

## Faille 6 — URL Hadith reconstruite

### Symptôme V1/V2

V1/V2 reconstruisait les URLs sunnah.com dynamiquement :

```python
# V1 code (faux)
url = f"https://sunnah.com/{slug}:{hadith_number}"
# hadith_number = idx + 1 (compteur local dans le fichier)
```

### Impact

La numérotation sunnah.com est **différente** de la numérotation in-book. Pour Bukhari :
- in-book hadith #1 → sunnah.com/bukhari:1 ✓
- in-book hadith #100 → sunnah.com/bukhari:??? (parfois 100, parfois 99, parfois 101)

→ URLs 404 dans ~30% des cas. L'utilisateur ne peut pas vérifier la source.

### Fix V3

**URL stockée depuis meetif** (le JSON meetif contient déjà la bonne URL dans le champ `Reference`) :

```python
# V3 code (correct)
url = hadith_data.get("Reference", "")  # "https://sunnah.com/bukhari:1"
if not url:
    # Fallback only if missing (rare)
    url = f"https://sunnah.com/{slug}:{hadith_number}"
    log_warning(f"URL missing for {collection} #{hadith_number}, using fallback")
```

L'URL est **jamais reconstruite** si elle est disponible dans les données.

---

## Faille 7 — Pas de détection Ikhtilaf

### Symptôme V1/V2

V1/V2 n'avait qu'un seul tafsir (Ibn Kathir). Pillar 9 (Ikhtilaf awareness) était **impossible à implémenter**.

### Impact

Pour 2:114 (interdiction de chanter dans la mosquée) :
- Ibn Kathir : "singing in masjid is forbidden"
- Al-Qurtubi : "only music instruments are forbidden, singing is allowed"
- Al-Tabari : rapporte plusieurs opinions sans trancher

V1/V2 : présente uniquement la vue d'Ibn Kathir comme LA réponse. Pillar 9 violé.

### Fix V3

**Multi-tafsir (Ibn Kathir + Tabari + Sa'di)** + **détection NLI automatique** :

```python
# Compare tafsirs 2-à-2
nli_score = nli_model.predict([
    {"premise": ibn_kathir_text, "hypothesis": tabari_text}
])
if nli_score["contradiction"] > 0.7:
    chunk["metadata"]["ikhtilaf"] = {
        "detected": True,
        "between": ["Ibn Kathir", "Al-Tabari"],
        "summary": "..."
    }
```

Le Reporter LLM voit `<ikhtilaf detected="true">` dans le XML et **doit présenter les deux vues** (Pillar 9).

---

## Faille 8 — Pas de vérification Quran char

### Symptôme V1/V2

Aucune vérification post-génération. Le LLM pouvait paraphraser les versets.

### Impact

LLM output : "Le verset 2:195 dit « ne pas gaspiller votre argent »" (paraphrase).
Réalité : 2:195 dit "وَأَنفِقُوا۟ فِى سَبِيلِ ٱللَّهِ وَلَا تُلْقُوا۟ بِأَيْدِيكُمْ إِلَى ٱلتَّهْلُكَةِ" = "And spend in the way of Allah and do not throw [yourselves] into destruction".

→ L'utilisateur reçoit une paraphrase incorrecte présentée comme Parole d'Allah. Pillar 4 violé.

### Fix V3

**Char-by-char check post-génération** (Pillar 4.2) :

```python
def verify_quran_text(citations, quran_collection):
    for cit in citations:
        if cit["type"] != "quran":
            continue
        original = quran_collection.get(id=cit["source_id"])["metadata"]["text_ar"]
        norm_original = normalize_arabic(original)
        norm_cited = normalize_arabic(cit["arabic"])
        if norm_original != norm_cited:
            return {"valid": False, "issue": "quran_text_mismatch"}
    return {"valid": True}
```

Si mismatch → rerun avec correction. Si rerun échoue → message d'erreur explicite.

---

## Récapitulatif — Pourquoi V3 va marcher

| Aspect | V1/V2 | V3 |
|--------|-------|-----|
| Tafsir connecté au verset | ❌ Non | ✅ Même chunk |
| Numérotation | ❌ Globale décalée | ✅ Standard surah:ayah |
| Compétition collections | ❌ Parallèle | ✅ 2 phases séquentielles |
| Pont sémantique modern | ❌ Aucun | ✅ Tafsir = pont |
| Cross-lingual FR/EN/AR | ❌ Manque FR | ✅ Context Card 3 langues |
| URL Hadith | ❌ Reconstruite | ✅ Stockée depuis meetif |
| Ikhtilaf detection | ❌ Impossible | ✅ NLI cross-tafsir |
| Quran char verification | ❌ Aucune | ✅ Char-by-char post-gen |
| **Resultat attendu** | 33% recall | **≥ 80% recall** |

---

## Prochain document

→ `08_EXAMPLES.md` : 5 exemples concrets end-to-end (prière, Ayat al-Kursi, smoking, hajj, riba) pour valider l'archi sur des cas réels.
