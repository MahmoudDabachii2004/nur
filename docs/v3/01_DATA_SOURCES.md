# V3 — Sources de données (Data Sources)

> Inventaire complet et exact de toutes les sources utilisées par V3.
> Chaque source documentée : URL, schema, qualité, license, comment on l'utilise.
> Date : 2026-06-23

---

## Récapitulatif des sources

| # | Source | Type | Volume | Status | Usage V3 |
|---|--------|------|--------|--------|----------|
| 1 | alquran.cloud | Quran API | 6,236 ayahs | À télécharger | Couche 2 (Word of Allah) AR + EN |
| 2 | spa5k/tafsir_api | Tafsir (multi) | 115 éditions | Ibn Kathir EN déjà en parquet ; autres à télécharger | Couche 3 (Human Commentary) |
| 3 | meetif (github) | Hadith | 6 collections, ~33,738 hadiths | À télécharger | Collection hadith_v3 |
| 4 | sunnah.com | URLs Hadith | N/A | Dérivé des métadonnées meetif | URL canonique |
| 5 | quran.com | URLs Quran/Tafsir | N/A | Construit à partir de surah:ayah | URL canonique |

---

## Source 1 — alquran.cloud (Quran)

### Endpoint
- Base : `http://api.alquran.cloud/v1`
- Pas de clé API, pas de rate limit stricte (usage raisonnable)

### Éditions utilisées

| Édition ID | Description | Usage V3 |
|------------|-------------|----------|
| `quran-uthmani` | Arabe Uthmani — **source of truth** | Couche 2 AR |
| `en.sahih` | Saheeh International (EN, la plus acceptée) | Couche 2 EN |
| `fr.hamidullah` | Hamidullah (FR — translation officielle fr) | Couche 2 FR (optionnel, pour utilisateur FR) |
| Meta endpoint | Noms sourates, type révélation, nb ayahs | Métadonnées |

### Schema de réponse (édition)

```json
{
  "code": 200,
  "status": "OK",
  "data": {
    "number": 1,
    "name": "سُورَةُ ٱلْفَاتِحَةِ",
    "englishName": "Al-Faatiha",
    "englishNameTranslation": "The Opening",
    "revelationType": "Meccan",
    "numberOfAyahs": 7,
    "ayahs": [
      {
        "number": 1,
        "text": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
        "numberInSurah": 1,
        "juz": 1, "page": 1, "hizbQuarter": 1,
        "ruku": 1, "manzil": 1, "sajda": false
      },
      ...
    ]
  }
}
```

### Notes importantes

- **Bismillah** : à l'exception de Surah 9 (At-Tawbah), chaque sourate commence par `بِسْمِ ٱللَّهِ`. Dans l'édition `quran-uthmani`, le Bismillah est **inclus dans le `text` du 1er ayah** pour les sourates autres que Al-Fatiha. Pour Al-Fatiha, le Bismillah **est** l'ayah 1. V3 garde le texte tel quel (pas de stripping).
- **`numberInSurah`** : c'est CETTE valeur qu'on utilise comme `ayah` (pas `number` qui est global).
- **BOM (U+FEFF)** : l'API préfixe parfois le texte avec BOM. Le script V3 le strippe.

### License
- alquran.cloud : open data, libre usage

---

## Source 2 — spa5k/tafsir_api (Tafsirs)

### Repository
- URL : https://github.com/spa5k/tafsir_api
- Taille totale : ~1.15 GB
- Structure : `tafsir/{edition_name}/{surah_number}.json` (114 fichiers par édition)
- Téléchargement via `raw.githubusercontent.com` (le repo dépasse la limite jsDelivr 50MB)

### Schema d'un fichier surah

```json
[
  {
    "text": "The Meaning of Al-Hamd. Abu Ja`far bin Jarir said...",
    "ayah": 1,
    "surah": 1
  },
  {
    "text": "Allah said next, Ar-Rahman Ar-Rahim...",
    "ayah": 2,
    "surah": 1
  },
  ...
]
```

**Clé** : chaque entrée = 1 tafsir pour 1 ayah. **Déjà aligné verset-par-verset.**

### Parquet déjà disponible

`data/tafsir/en_ibn_kathir.parquet` (11 MB, 6,235 lignes) — converti depuis l'édition `en-tafisr-ibn-kathir` (notez le typo upstream "tafisr").

| Colonne | Description |
|---------|-------------|
| `sorah` | Numéro sourate (1-114) — **typo upstream** : "sorah" pas "surah" |
| `ayah` | Numéro ayah dans la sourate |
| `sentence` | Le verset AR Uthmani (déjà inclus !) |
| `en-tafsir-ibn-kathir-html` | Tafsir HTML |
| `en-tafsir-ibn-kathir-text` | Tafsir texte plat (sans HTML) ← **on utilise ça** |

### Les 115 éditions disponibles — classification

#### Catégorie A : Tafsir bil-Mathur (par tradition — le plus authentique)

| Édition | Savant | Siècle | École | Langue | Sélection V3 ? |
|---------|--------|--------|-------|-------|-----------------|
| `ar-tafsir-al-tabari` | Al-Tabari | 10e | — | AR | ✅ **SECONDARY** |
| `ar-tafsir-ibn-kathir` | Ibn Kathir | 14e | Shafi'i | AR | ✅ **PRIMARY** |
| `en-tafisr-ibn-kathir` | Ibn Kathir (traduit) | 14e | Shafi'i | EN | ✅ **PRIMARY** (déjà en parquet) |
| `ar-tafsir-al-baghawi` | Al-Baghawi | 12e | Shafi'i | AR | Optionnel V3.1 |
| `tafsir-ibn-abi-hatim` | Ibn Abi Hatim | 10e | — | AR | Optionnel V3.1 |
| `mawsoo-at-al-tafsir-al-ma-thoor` | Compilation | Moderne | — | AR | Non (redondant) |

#### Catégorie B : Tafsir bil-Ra'y (par raisonnement — linguistique/légal)

| Édition | Savant | Siècle | École | Langue | Sélection V3 ? |
|---------|--------|--------|-------|-------|-----------------|
| `ar-tafseer-al-qurtubi` | Al-Qurtubi | 13e | Maliki | AR | Optionnel V3.1 (fiqh) |
| `tafsir-al-razi` | Fakhr al-Din al-Razi | 13e | — | AR | Non (trop philosophique) |
| `tafsir-al-baydawi` | Baydawi | 13e | Shafi'i | AR | Non (redondant) |
| `tafsir-al-nasafi` | Al-Nasafi | 14e | Hanafi | AR | Optionnel V3.1 (fiqh) |
| `al-kashshaf-al-zamakhshari` | Zamakhshari | 12e | Mu'tazila | AR | Non (école controversée) |
| `tafsir-al-mawardi` | Al-Mawardi | 11e | Shafi'i | AR | Non |
| `tafsir-al-sam-ani` | Al-Sam'ani | 12e | — | AR | Non |
| `tafsir-al-samarqandi` | Al-Samarqandi | 9e | Hanafi | AR | Non |

#### Catégorie C : Tafsir Ishari (mystique/spirituel)

| Édition | Savant | Siècle | Langue | Sélection V3 ? |
|---------|--------|--------|-------|-----------------|
| `en-tafsir-al-tustari` | Tustari | 9e | EN | Non (non-littéral, hors scope V3) |
| `en-kashf-al-asrar-tafsir` | Kashf al-Asrar | 12e | EN | Non |
| `en-kashani-tafsir` | Kashani | 15e | EN | Non |
| `en-al-qushairi-tafsir` | Al-Qushairi | 11e | EN | Non |
| `kashf-al-asrar-tafsir` | (AR) | — | AR | Non |

#### Catégorie D : Tafsir moderne (accessible, pour questions contemporaines)

| Édition | Savant | Siècle | Langue | Sélection V3 ? |
|---------|--------|--------|-------|-----------------|
| `ar-tafsir-as-saadi` | As-Sa'di | 20e | AR | ✅ **TERTIARY** |
| `fr-tafsir-as-saadi` | As-Sa'di | 20e | FR | ✅ Backup FR (l'utilisateur FR) |
| `ar-tafsir-al-mukhtasar` | Al-Mukhtasar | Moderne | AR | ✅ Backup tertiary |
| `en-tafsir-al-mukhtasar` | Al-Mukhtasar | Moderne | EN | Optionnel |
| `tafsir-ibn-uthaymeen` | Ibn Uthaymeen | 20e | AR | Optionnel V3.1 |
| `en-tazkirul-quran` | Tazkirul Quran | 20e | EN | Optionnel |
| `en-tafsir-maarif-ul-quran` | Maarif-ul-Quran | 20e | EN | Optionnel |

#### Catégorie E : Concis (Jalalayn, etc.)

| Édition | Savant | Siècle | Langue | Sélection V3 ? |
|---------|--------|--------|-------|-----------------|
| `ar-tafsir-al-jalalayn` | Jalalayn (Jalal al-Din) | 15e | AR | Optionnel V3.1 |
| `en-al-jalalayn` | Jalalayn (traduit) | 15e | EN | Optionnel |
| `al-wajiz-wahidi` | Wahidi | 11e | AR | Non |

#### Catégorie F : Thématiques spéciales (causes de révélation, grammar, etc.)

| Édition | Type | Langue | Sélection V3 ? |
|---------|------|--------|-----------------|
| `en-asbab-al-nuzul-by-al-wahidi` | Causes de révélation | EN | ✅ **À merger dans Couche 3 si dispo** |
| `tadabbur-wa-amal` | Réflexion et action | AR | Non |
| `i-rab-al-quran-li-al-darwish` | Grammaire | AR | Non |
| `al-i-rab-al-muyassar` | Grammaire | AR | Non |
| `tahlil-kalimat-al-qur-an` | Lexique | AR | Non |

#### Catégorie G : Traductions régionales (hors scope V3)
Tagalog, Tamil, Thai, Turkish, Urdu, Vietnamese, Uyghur, Uzbek, Kurdish, Khmer, Bengali, Hindi, Italian, Japanese, Spanish, Russian, Persian, Pashto, Bosnian, etc. → **Non utilisées en V3**.

### Sélection V3 finale — 4 tafsirs par verset (quand dispo)

```
PRIMARY (toujours) :
  - Ibn Kathir EN (parquet existant)
  - Ibn Kathir AR (à télécharger)

SECONDARY (si dispo pour ce verset) :
  - Al-Tabari AR (référence classique)

TERTIARY (si dispo pour ce verset) :
  - As-Sa'di AR (moderne accessible)
```

**Rationale** :
- **Ibn Kathir** = spine bil-Mathur universellement accepté
- **Tabari** = référence classique pour détecter l'Ikhtilaf (Pillar 9) — si Tabari et Ibn Kathir divergent, on présente les deux
- **Sa'di** = moderne, langage accessible, meilleure connexion aux questions modernes (smoking, crypto, etc.)
- **Qurtubi** en V3.1 (optionnel) — pour les versets à dimension légale (fiqh), où la perspective Maliki enrichit

### Volumes estimés (V3)

| Édition | Sections estimées | Taille disque |
|---------|-------------------|---------------|
| Ibn Kathir EN | 6,235 | 11 MB (déjà) |
| Ibn Kathir AR | ~6,235 | ~15 MB |
| Al-Tabari AR | ~6,000 | ~30 MB |
| As-Sa'di AR | ~6,235 | ~10 MB |
| **Total** | ~25,000 sections | **~66 MB** |

### License
- spa5k/tafsir_api : MIT, contenu islamique public domain

---

## Source 3 — meetif (Hadiths)

### Repository
- URL : https://github.com/fawazahmed0/hadith-api (mirror de meetif)
- Endpoint alternatif : `https://github.com/meetif/meetif-hadith` (deprecated — on utilise fawazahmed0)
- Format : 1 fichier JSON par collection
- CDN : jsDelivr (fonctionne car fichiers < 50MB)

### Collections disponibles (V3 utilise les 6 canoniques)

| Collection | Slug sunnah.com | Hadiths estimés | Grade dominant |
|------------|-----------------|-----------------|----------------|
| Sahih al-Bukhari | `bukhari` | 7,563 | Sahih |
| Sahih Muslim | `muslim` | 7,564 | Sahih |
| Sunan Abi Dawud | `abudawud` | 5,274 | Mixte (Sahih/Hasan/Da'if) |
| Jami` at-Tirmidhi | `tirmidhi` | 3,956 | Mixte (avec grades explicites) |
| Sunan an-Nasa'i | `nasai` | 5,758 | Majoritairement Sahih |
| Sunan Ibn Majah | `ibnmajah` | 4,341 | Mixte |

**Total** : ~33,738 hadiths

### Schema d'un hadith (meetif)

```json
{
  "Book": "Book of Revelation",
  "Chapter_Number": 1,
  "Chapter_Title_Arabic": "كتاب الوحي",
  "Chapter_Title_English": "Revelation",
  "Arabic_Text": "حدّثنا الحميدي عبد الله بن الزيد...",
  "English_Text": "Narrated 'Umar bin Al-Khattab (ra): I heard Allah's Messenger (ﷺ) saying...",
  "Grade": "Sahih",
  "Reference": "https://sunnah.com/bukhari:1",
  "In-book reference": "Book 1, Hadith 1"
}
```

### Notes critiques

- **`Reference` (URL sunnah.com)** : **TOUJOURS utilisée telle quelle**. Ne JAMAIS reconstruire l'URL dynamiquement — la numérotation sunnah.com est décalée par rapport à la numérotation in-book. (Cf. ARCHITECTURE.md Sin #5 : c'est une faille V1/V2.)
- **`Grade`** : présent pour Tirmidhi et Abi Dawud (majoritairement). Pour Bukhari/Muslim, le grade est implicite "Sahih" (toute la collection l'est). On set `grade="Sahih"` si manquant pour ces 2 collections.
- **`Arabic_Text`** : peut contenir BOM — on strippe.
- **`English_Text`** : commence souvent par `"Narrated X:"` — on extrait le narrator pour le champ `narrator` (V3 amélioration par rapport à V1).

### License
- meetif/fawazahmed0 : public domain, hadiths sont patrimoine islamique

---

## Source 4 & 5 — URLs canoniques (dérivées)

### Quran URL
- Format : `https://quran.com/{surah}/{ayah}`
- Construction : **toujours depuis `surah` + `ayah` canoniques** (numérotation standard Quran, pas compteur global)
- Exemple : `https://quran.com/2/195` pour Al-Baqarah ayah 195
- Vérification : URL fetch OK sur quran.com (testé)

### Tafsir URL
- Format : `https://quran.com/tafsir/{surah}/{ayah}`
- Construction : depuis `surah` + `ayah`
- Exemple : `https://quran.com/tafsir/2/195`

### Hadith URL
- Format : `https://sunnah.com/{slug}:{hadith_number}`
- ⚠️ **CRITIQUE** : `hadith_number` provient du champ `Reference` du JSON meetif, **PAS** d'un compteur in-book
- Exemple valide : `https://sunnah.com/bukhari:1`
- Mapping slugs : voir `src/nur/sources.py → HADITH_COLLECTION_SLUGS`

---

## Matrice de couverture (qui explique qui)

```
                  Quran  Hadith  Tafsir
Requête FR :       ✅     ✅      ✅    (3 langues via Context Card)
Requête EN :       ✅     ✅      ✅
Requête AR :       ✅     ✅      ✅
Question moderne :  ⚠️    ✅      ✅    (smoking, crypto — via tafsir pont)
Question fiqh :    ⚠️     ✅      ✅    (wudu steps — via tafsir + hadith)
Question aqida :   ✅     ✅      ✅
```

⚠️ = le verset seul ne suffit pas, mais le tafsir dans le chunk fournit le pont sémantique.

---

## Prochain document

→ `02_CHUNK_SCHEMA.md` : schéma détaillé des chunks V3 (3 couches Quran, 2 couches Hadith) avec la structure JSON exacte pour ChromaDB.
