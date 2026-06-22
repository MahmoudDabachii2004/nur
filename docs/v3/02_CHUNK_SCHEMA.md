# V3 — Schéma des chunks

> Structure exacte des chunks V3 stockés dans ChromaDB.
> Ce document est la source de vérité pour le code `scripts/v3/05_build_chunks.py`.
> Date : 2026-06-23

---

## Vue d'ensemble — 2 collections, 2 schémas

```
quran_v3   →  6,236 chunks  →  3 couches embedding_text
hadith_v3  →  33,738 chunks →  2 couches embedding_text
```

Les deux collections partagent un schéma **JSON commun** pour ChromaDB, mais le contenu de `embedding_text` et de `metadata` diffère.

---

## Schéma JSON Commun (ChromaDB)

```python
{
    "id": "SRC-QURAN-2-195",                          # PK, unique
    "embedding": [0.0123, -0.0456, ...],              # 1024 floats (BGE-M3)
    "embedding_text": "...",                          # texte embeddé (3 ou 2 couches)
    "metadata": {
        "kind": "quran",                              # "quran" | "hadith"
        "surah": 2,                                   # int | null (hadith)
        "ayah": 195,                                  # int | null (hadith)
        "surah_name_ar": "البقرة",
        "surah_name_en": "Al-Baqarah",
        "revelation_type": "Medinan",                 # "Meccan" | "Medinan"
        "collection": null,                           # hadith only
        "hadith_number": null,                        # hadith only
        "narrator": null,                             # hadith only
        "grade": null,                                # hadith only ("Sahih"|"Hasan"|"Da'if"|null)
        "grade_weight": 1.0,                          # pour scoring retrieval

        # Textes originaux (PAS dans embedding_text, mais dans metadata pour le LLM)
        "text_ar": "وَأَنفِقُوا۟ فِي سَبِيلِ ٱللَّهِ...",
        "text_en": "And spend in the way of Allah...",
        "text_fr": "",                                # optionnel (Hamidullah)
        "previous_ayah_ar": "...",                    # Quran only
        "previous_ayah_en": "...",
        "next_ayah_ar": "...",                        # Quran only
        "next_ayah_en": "...",

        # Context Card (couche 1)
        "context_card": {
            "fr": {
                "theme": "Préservation de la vie, Auto-destruction",
                "rule": "Il est interdit de se causer du tort à soi-même",
                "keywords": ["self-harm", "destruction", "نقتل أنفسنا", "ضرر"]
            },
            "en": {
                "topic": "Preservation of life, Self-destruction",
                "rule": "Causing harm to oneself is forbidden",
                "keywords": ["self-harm", "destruction", "suicide", "harm"]
            },
            "ar": {
                "theme": "حفظ النفس، الإلقاء في التهلكة",
                "keywords": ["تهلكة", "ضرر", "إتلاف"]
            }
        },

        # Tafsirs (couche 3 — Quran only)
        "tafsirs": [
            {
                "source": "Ibn Kathir",
                "category": "bil-Mathur",
                "language": "en",
                "text": "Ibn Kathir explains: \"Do not throw yourselves into destruction\"...",
                "text_full": "...",                   # texte complet non tronqué
                "truncated": false,                   # true si > 800 chars
                "url": "https://quran.com/tafsir/2/195"
            },
            {
                "source": "Ibn Kathir",
                "category": "bil-Mathur",
                "language": "ar",
                "text": "...",
                "text_full": "...",
                "truncated": false,
                "url": "https://quran.com/tafsir/2/195"
            },
            {
                "source": "Al-Tabari",
                "category": "bil-Mathur",
                "language": "ar",
                "text": "...",
                "text_full": "...",
                "truncated": true,
                "url": "https://quran.com/tafsir/2/195"
            },
            {
                "source": "As-Sa'di",
                "category": "modern",
                "language": "ar",
                "text": "...",
                "text_full": "...",
                "truncated": false,
                "url": "https://quran.com/tafsir/2/195"
            }
        ],

        # Cross-refs Quran → Hadith (pré-calculés via parsing tafsir)
        "hadith_cross_refs": ["SRC-HADITH-BUKHARI-1234", "SRC-HADITH-MUSLIM-567"],

        # URL canonique
        "url": "https://quran.com/2/195",

        # Build info
        "build_version": "v3",
        "build_date": "2026-06-23"
    }
}
```

---

## Couche 1 — Context Card (LLM-generated)

### Objectif
Ajouter un **pont sémantique cross-lingual** au-dessus du verset. Sans ça, "prière" FR ne matche pas "salah" AR.

### Contenu

```json
{
  "fr": {
    "theme": "Préservation de la vie, Auto-destruction",
    "rule": "Il est interdit de se causer du tort à soi-même",
    "keywords": ["self-harm", "destruction", "نقتل أنفسنا", "ضرر"]
  },
  "en": {
    "topic": "Preservation of life, Self-destruction",
    "rule": "Causing harm to oneself is forbidden",
    "keywords": ["self-harm", "destruction", "suicide", "harm"]
  },
  "ar": {
    "theme": "حفظ النفس، الإلقاء في التهلكة",
    "keywords": ["تهلكة", "ضرر", "إتلاف"]
  }
}
```

### Règles strictes pour le LLM qui génère la Context Card

1. **`keywords` = intersection stricte** des termes présents dans :
   - Le verset AR (forme racine)
   - La traduction EN
   - Le tafsir Ibn Kathir (premier paragraphe)
   - **JAMAIS extrapolés** (pas de keywords inventés par le LLM)
2. **`theme`** doit être ≤ 5 mots, en FR/EN/AR
3. **`rule`** doit être ≤ 15 mots, en FR/EN (AR optionnel)
4. Les keywords incluent toujours **au moins 1 terme AR** (la racine) + 1 terme EN + 1 terme FR
5. Si le verset est purement narratif (pas de ruling clair), `rule = ""`

### Coût de génération

- 6,236 versets × 1 LLM call (Groq `llama-3.1-8b-instant`)
- ~800 tokens in, ~200 tokens out
- Groq free tier : 30K TPM → 6,236 / (30K / 1000) = ~4 heures
- Coût monétaire : $0 (free tier)
- Si Groq 429 → fallback Ollama local (`llama3.1:8b`)

---

## Couche 2 — Word of Allah (PURE, untouched)

### Contenu (dans l'embedding_text)

```
[WORD OF ALLAH — PURE]
Quran | Surah 2: Al-Baqarah (البقرة) | Ayah 195 | Revelation: Medinan
Arabic: وَأَنفِقُوا۟ فِي سَبِيلِ ٱللَّهِ وَلَا تُلْقُوا۟ بِأَيْدِيكُمْ إِلَى ٱلتَّهْلُكَةِ
English: And spend in the way of Allah and do not throw [yourselves] into destruction with your own hands.
Previous (2:194): ...fight within the sacred months...
Next (2:196): And complete the Hajj and 'umrah for Allah...
```

### Règles

1. **Arabic = Uthmani script** (source of truth, jamais normalisé dans l'embedding_text — on garde la forme révélation)
2. **English = Saheeh International** (la plus acceptée)
3. **Previous/Next ayah** : juste le 1er segment (jusqu'à la première virgule) pour donner le contexte sans diluer. Max 200 chars chacun.
4. **Pas de commentaire**, pas de "this verse means...", pas de glossaire. Le texte EST le texte.

### Cas spécial — Sourate 1 (Al-Fatiha)

Al-Fatiha ayah 1 = `بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ`. Pas de "previous" (c'est le début).

### Cas spécial — Sourate 9 (At-Tawbah)

Pas de Bismillah au début. On ne l'ajoute pas artificiellement.

### Cas spécial — Dernier ayah d'une sourate

Pas de "next" (on passe à la sourate suivante — c'est OK, on laisse vide).

---

## Couche 3 — Tafsirs (labellisés, multi-source)

### Contenu (dans l'embedding_text, AVEC truncation)

```
[HUMAN COMMENTARY — LABELED]
[Source: Tafsir Ibn Kathir | Category: bil-Mathur | Language: EN]
Ibn Kathir explains: "Do not throw yourselves into destruction" means:
do not expose yourselves to destruction by refraining from spending
in Allah's cause. Some scholars (Mujahid, Qatadah) extended this to
mean: any action that knowingly brings harm to one's body or wealth
is included in this prohibition. This includes self-inflicted harm
through dangerous consumption, dangerous activities, or self-destruction.

[Source: Tafsir Al-Tabari | Category: bil-Mathur | Language: AR]
قال الطبري: ولا تلقوا بأيديكم إلى التهلكة، يقول: ولا تلقوا بأنفسكم
إلى الهلاك، بترككم النفقة في سبيل الله...

[Source: Tafsir As-Sa'di | Category: modern | Language: AR]
يقول السعدي: وهذا نهي عن الإلقاء بالنفس إلى التهلكة، والمراد بذلك
ترك الواجبات وفعل المحرمات...
```

### Truncation — l'analyse complète (voir `05_EMBEDDING_DESIGN.md`)

**Décision** : chaque tafsir est tronqué à **800 chars** dans `embedding_text`. Le texte complet est conservé dans `metadata.tafsirs[].text_full` pour le LLM.

**Rationale détaillée** : voir `05_EMBEDDING_DESIGN.md` (étude comparative des truncations 400/600/800/1200/2000 chars, avec mesures sur 6,236 versets).

### Ordre des tafsirs dans la couche 3

1. **Ibn Kathir EN** (PRIMARY, langage utilisateur EN/FR — broad access)
2. **Ibn Kathir AR** (PRIMARY, source of truth dans langue originale)
3. **Al-Tabari AR** (SECONDARY, Ikhtilaf detection)
4. **As-Sa'di AR** (TERTIARY, modern frame)

### Marqueur Isra'iliyyat

Si le tafsir contient du matériel identifié comme Isra'iliyyat (biblical narratives), on ajoute dans le texte :

```
[Source: Tafsir Ibn Kathir | Category: bil-Mathur | Language: EN | Contains Isra'iliyyat — illustrative narrations, not authentically verified]
```

**Détection** : heuristique sur keywords `("Banu Isra'il", "Jews said", "Ka'b al-Ahbar", "Israelites narrated", "Wahb ibn Munabbih")`. Cette détection est conservative — mieux vaut manquer un Isra'iliyyat que marquer faussement un hadith authentique comme Isra'iliyyat.

---

## Schéma Hadith V3

```python
{
    "id": "SRC-HADITH-BUKHARI-1",
    "embedding": [...],
    "embedding_text": "...",  # 2 couches
    "metadata": {
        "kind": "hadith",
        "surah": null,
        "ayah": null,
        "collection": "Sahih al-Bukhari",
        "hadith_number": 1,
        "narrator": "Umar bin Al-Khattab",
        "grade": "Sahih",
        "grade_weight": 1.30,

        # Textes originaux
        "text_ar": "حدّثنا الحميدي عبد الله بن الزيد قال...",
        "text_en": "Narrated 'Umar bin Al-Khattab (ra): I heard Allah's Messenger (ﷺ) saying...",
        "text_fr": "",  # pas de trad FR en V3

        # Context Card
        "context_card": {
            "en": {
                "topic": "Intentions, Actions",
                "rule": "Actions are judged by intentions",
                "keywords": ["intention", "niyyah", "نية", "actions", "deeds"]
            },
            "ar": {
                "theme": "النيات، الأعمال",
                "keywords": ["نية", "أعمال", "إخلاص"]
            }
        },

        # Chapter info (meetif)
        "book": "Book of Revelation",
        "chapter_number": 1,
        "chapter_arabic": "كتاب الوحي",
        "chapter_english": "Revelation",
        "in_book_reference": "Book 1, Hadith 1",

        # URL canonique (TOUJOURS depuis meetif "Reference", jamais reconstruite)
        "url": "https://sunnah.com/bukhari:1",

        "build_version": "v3",
        "build_date": "2026-06-23"
    }
}
```

### embedding_text Hadith (2 couches)

```
[CONTEXT CARD]
[EN] Topic: Intentions, Actions. Rule: Actions are judged by intentions.
[AR] الموضوع: النيات، الأعمال
Keywords: intention, niyyah, نية, actions, deeds, إخلاص

[HADITH — Sahih al-Bukhari #1 | Grade: Sahih | Narrator: Umar bin Al-Khattab]
Arabic: حدّثنا الحميدي عبد الله بن الزيد قال حدّثنا سفيان قال حدّثنا
يعلى بن سعيد عن أبي ثور عن أبي هريرة قال قال رسول الله ﷺ إنما الأعمال
بالنيات وإنما لكل امرئ ما نوى...
English: Narrated 'Umar bin Al-Khattab (ra): I heard Allah's Messenger (ﷺ)
saying, "The reward of deeds depends upon the intentions and every person
will get the reward according to what he has intended..."
```

### Notes Hadith

- **Pas de couche 3 (pas de tafsir)** : les hadiths sont déjà eux-mêmes du commentaire. Pas besoin d'ajouter une couche humaine par-dessus.
- **Narrator extraction** : depuis `English_Text`, regex `^Narrated ([^:]+):`. Si échec, `narrator = null`.
- **Grade fallback** : si `grade` absent et collection ∈ {Bukhari, Muslim}, set `grade = "Sahih"`.
- **URL** : **TOUJOURS** prise depuis `Reference` du JSON meetif. Si absente (rare), fallback construction `https://sunnah.com/{slug}:{hadith_number}` (avec warning dans logs).

---

## Cross-refs Quran → Hadith (implémentation détaillée)

### Objectif

Quand le verset 2:195 est retenu en Phase A, on veut pouvoir **automatiquement** récupérer les hadiths qui le commentent, sans requérir l'utilisateur de poser une 2e question.

### Comment pré-calculer

3 méthodes combinées pour identifier les hadiths liés à un verset :

#### Méthode 1 — Parsing tafsir (haute précision)

Le tafsir Ibn Kathir cite souvent des hadiths explicitement avec leur chaîne (`It is reported from Bukhari that...` ou `The Prophet (ﷺ) said: ...`). On parse le `text_full` des tafsirs et on extrait :

```python
# Patterns à matcher
PATTERNS_HADITH_CITATION = [
    r"(?:Bukhari|Sahih Bukhari|al-Bukhari)[\w\s,]*?(\d{1,5})",  # "Bukhari 123"
    r"(?:Muslim|Sahih Muslim)[\w\s,]*?(\d{1,5})",
    r"(?:Abu Dawud|AbuDawud)[\w\s,]*?(\d{1,5})",
    r"(?:Tirmidhi|at-Tirmidhi)[\w\s,]*?(\d{1,5})",
    r"(?:Nasa'i|an-Nasa'i)[\w\s,]*?(\d{1,5})",
    r"(?:Ibn Majah)[\w\s,]*?(\d{1,5})",
]
```

Si un pattern match, on ajoute `SRC-HADITH-{COLLECTION}-{NUM}` à `hadith_cross_refs`.

**Précision** : ~85% (parfois faux positifs sur "Bukhari 123" mentionné dans un contexte annexe)

#### Méthode 2 — Asbab al-Nuzul (causes de révélation)

L'édition `en-asbab-al-nuzul-by-al-wahidi` documente pour chaque verset les événements qui ont causé sa révélation. Ces événements sont souvent liés à des hadiths spécifiques.

On parse ce fichier → on extrait les hadiths cités → on ajoute à `hadith_cross_refs`.

#### Méthode 3 — Keyword matching (faible précision, fallback)

Si méthodes 1+2 ne donnent rien, on utilise les keywords de la Context Card du verset pour trouver des hadiths qui mentionnent les mêmes concepts. On garde seulement les hadiths avec un score de matching ≥ 0.7 (calculé via BGE-M3 cosine similarity entre verset et hadith).

**Utilisé seulement si** méthodes 1+2 ne produisent aucun résultat, et le résultat est flagué `"low_confidence": true`.

### Stockage

```python
"hadith_cross_refs": {
    "high_confidence": [   # issus de parsing tafsir + Asbab al-Nuzul
        "SRC-HADITH-BUKHARI-1234",
        "SRC-HADITH-MUSLIM-567"
    ],
    "low_confidence": [],  # issus de keyword matching
    "source_methods": ["tafsir_parsing", "asbab_al_nuzul"]
}
```

### Utilisation au retrieval

Quand un chunk Quran est retenu en Phase A :

```python
for chunk in top_quran_chunks:
    if chunk.metadata.get("hadith_cross_refs"):
        for hadith_id in chunk.metadata["hadith_cross_refs"]["high_confidence"]:
            # Auto-pull le hadith depuis hadith_v3 par ID
            hadith_chunk = hadith_collection.get(hadith_id)
            if hadith_chunk:
                # On l'ajoute à la liste des hadiths à présenter, 
                # SANS qu'il soit passé par le retrieval Phase B normal
                # Il est garanti d'être présent car le verset le cite
                auto_pulled_hadiths.append(hadith_chunk)
```

Ces hadiths auto-pulled sont **présentés au LLM avec un flag** :

```xml
<document id="S6" auto_pulled="true">
  <source_id>SRC-HADITH-BUKHARI-1234</source_id>
  <source_type>hadith</source_type>
  <label>Sahih al-Bukhari #1234</label>
  <grade>Sahih</grade>
  <note>Auto-pulled because cited in Tafsir Ibn Kathir for Quran 2:195</note>
  ...
</document>
```

Le Reporter LLM sait que ces hadiths sont contextuellement liés au verset.

---

## Récapitulatif des améliorations V3 vs V1/V2

| Aspect | V1/V2 | V3 |
|--------|-------|-----|
| Collections | 4 séparées (quran, hadith, tafsir_ar, tafsir_en) | 2 (quran_v3, hadith_v3) |
| Tafsir embeddé | SEUL, orphelin du verset | AVEC le verset, dans même chunk |
| Tafsirs multiples | Non | Oui (Ibn Kathir EN+AR, Tabari, Sa'di) |
| Numérotation ayah | Globale décalée | Standard (surah:ayah) |
| URL Quran | Construite décalée | Construite standard |
| URL Hadith | Reconstruite (FAIL) | Stockée depuis meetif |
| Context Card | Inexistante | LLM-generated, 3 langues |
| Cross-refs Quran→Hadith | Inexistante | Pré-calculée via parsing tafsir |
| Isra'iliyyat | Invisibles | Marqués dans tafsir |
| Previous/Next ayah | Inexistants | Inclus (contexte sémantique) |

---

## Prochain document

→ `03_TAFSIR_STRATEGY.md` : stratégie multi-tafsir détaillée (comment on merge, ordre, détection Ikhtilaf, fallback si un tafsir absent pour un verset).
