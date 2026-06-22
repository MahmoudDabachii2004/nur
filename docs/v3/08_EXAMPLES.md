# V3 — Exemples end-to-end (5 cas de test)

> 5 exemples concrets qui valident l'archi V3 sur des cas représentatifs.
> Ces 5 cas serviront de **ground truth** pour le script `07_verify_pipeline.py`.
> Date : 2026-06-23

---

## Cas 1 — Question classique FR ("Pourquoi la prière est obligatoire ?")

### Input
- **Langue** : FR
- **Question** : "Pourquoi la prière est obligatoire ?"
- **Catégorie** : Classique (le Quran parle directement de prière)

### Step 1 — Architect (query decomposition)

```json
{
  "sub_questions": [
    "obligation prière islam",
    "why prayer obligatory quran",
    "establish salah command",
    "pillars of islam prayer",
    "salah fard quran"
  ]
}
```

### Step 2 — Phase A (Quran+Tafsir retrieval)

Top-5 chunks `quran_v3` après rerank :

| Rank | Source | Rerank score | Why |
|------|--------|--------------|-----|
| 1 | SRC-QURAN-2-43 | 0.82 | "establish prayer" + "give zakah" — direct match |
| 2 | SRC-QURAN-2-3 | 0.74 | "establish prayer" mentionné dans la foi |
| 3 | SRC-QURAN-4-103 | 0.71 | "establish prayer at the two ends of the day" |
| 4 | SRC-QURAN-20-14 | 0.68 | "establish prayer for My remembrance" |
| 5 | SRC-QURAN-24-56 | 0.65 | "establish prayer and give zakah and obey the Messenger" |

**Confidence A = 0.82 → STRONG**

### Step 3 — Auto-pull hadiths

Pour 2:43 (`hadith_cross_refs.high_confidence`) :
- `SRC-HADITH-BUKHARI-8` — "Islam is built upon five..." (inclut la prière)
- `SRC-HADITH-MUSLIM-12` — "Between a man and disbelief is abandoning prayer"

### Step 4 — Phase B (Hadith retrieval)

Top-5 `hadith_v3` :
- S6: Bukhari #8 (5 pillars)
- S7: Muslim #12 (prayer distinguishes belief)
- S8: Tirmidhi #413 (prayer on the Day of Judgment)
- S9: Bukhari #522 (prayer is the coolness of my eye)
- S10: Ibn Majah #1401 (first thing judged)

**Confidence B = 0.78 × 1.30 (Sahih) = 1.01 → STRONG**

### Step 5 — Reporter output (FR)

```json
{
  "answer": "La prière (salah) est le 2e pilier de l'Islam et son obligation est établie dans plusieurs versets coraniques. [S1] (Al-Baqarah 2:43) commande directement : « وَأَقِيمُوا۟ ٱلصَّلَوٰةَ وَءَاتُوا۟ ٱلزَّكَوٰةَ » — « Et accomplissez la Salât et acquittez la Zakât ». Tafsir Ibn Kathir explique que ce commandement s'adresse aux Children of Israel mais s'applique universellement à la Ummah de Muhammad ﷺ. [S4] (Ta-Ha 20:14) confirme : « إِنَّنِىٓ أَنَا ٱللَّهُ لَآ إِلَٰهَ إِلَّآ أَنَا فَٱعْبُدْنِى وَأَقِمِ ٱلصَّلَوٰةَ لِذِكْرِىٓ » — « Adore-Moi et accomplis la Salât pour M'invoquer ». Les hadiths renforcent cette obligation : [S6] (Bukhari) cite le Prophète ﷺ : « L'Islam est bâti sur cinq piliers [...] et l'accomplissement de la prière ». [S7] (Muslim) ajoute : « Entre l'homme et la mécréance, il y a l'abandon de la prière ». En conclusion, la prière est une obligation individuelle (fard 'ayn) pour tout musulman pubère et sain d'esprit, établie par le Quran (commandement direct), la Sunnah (hadiths authentiques) et le consensus (Ijma').",
  "citations": [
    {
      "source_id": "SRC-QURAN-2-43",
      "label": "Quran 2:43",
      "type": "quran",
      "arabic": "وَأَقِيمُوا۟ ٱلصَّلَوٰةَ وَءَاتُوا۟ ٱلزَّكَوٰةَ وَٱرْكَعُوا۟ مَعَ ٱلرَّٰكِعِينَ",
      "english": "And establish prayer and give zakah and bow with those who bow.",
      "tafsir_used": "Ibn Kathir",
      "url": "https://quran.com/2/43"
    },
    {
      "source_id": "SRC-QURAN-20-14",
      "label": "Quran 20:14",
      "type": "quran",
      "arabic": "إِنَّنِىٓ أَنَا ٱللَّهُ لَآ إِلَٰهَ إِلَّآ أَنَا فَٱعْبُدْنِى وَأَقِمِ ٱلصَّلَوٰةَ لِذِكْرِىٓ",
      "english": "Indeed, I am Allah. There is no deity except Me, so worship Me and establish prayer for My remembrance.",
      "tafsir_used": "Ibn Kathir",
      "url": "https://quran.com/20/14"
    },
    {
      "source_id": "SRC-HADITH-BUKHARI-8",
      "label": "Sahih al-Bukhari #8",
      "type": "hadith",
      "arabic": "بُنِيَ الإِسْلاَمُ عَلَى خَمْسٍ...",
      "english": "Islam is based on five principles...",
      "url": "https://sunnah.com/bukhari:8"
    },
    {
      "source_id": "SRC-HADITH-MUSLIM-12",
      "label": "Sahih Muslim #12",
      "type": "hadith",
      "arabic": "بَيْنَ الرَّجُلِ وَبَيْنَ الْكُفْرِ تَرْكُ الصَّلاَةِ",
      "english": "Between a man and polytheism and disbelief is the abandonment of prayer.",
      "url": "https://sunnah.com/muslim:12"
    }
  ],
  "ikhtilaf": {"detected": false, "summary": null},
  "confidence": "high",
  "phase_a_status": "STRONG",
  "phase_b_status": "STRONG",
  "disclaimer": "Cette réponse reflète le consensus (Ijma') des savants. Pour des questions spécifiques (maladie, voyage, etc.), consultez un savant qualifié."
}
```

### Step 6 — Verification

| Check | Result |
|-------|--------|
| Schema valid | ✅ |
| Source IDs valid | ✅ |
| NLI per sentence | ✅ (all ≥ 0.95) |
| Quran char match 2:43 | ✅ exact |
| Quran char match 20:14 | ✅ exact |
| Tafsir labeled | ✅ "Tafsir Ibn Kathir explique..." |

→ **RÉPONSE VALIDÉE**

---

## Cas 2 — Question AR ("معلومات عن كرسي الله")

### Input
- **Langue** : AR
- **Question** : "معلومات عن كرسي الله"
- **Catégorie** : Classique (Ayat al-Kursi)

### Step 1 — Architect

```json
{
  "sub_questions": [
    "كرسي الله",
    "آية الكرسي",
    "throne of Allah quran",
    "ayat al-kursi 2:255",
    "صفات الله الحي القيوم"
  ]
}
```

### Step 2 — Phase A

Top-5 `quran_v3` :

| Rank | Source | Score |
|------|--------|-------|
| 1 | SRC-QURAN-2-255 | 0.94 |
| 2 | SRC-QURAN-3-2 | 0.62 |
| 3 | SRC-QURAN-20-5 | 0.58 |
| 4 | SRC-QURAN-7-54 | 0.55 |
| 5 | SRC-QURAN-32-4 | 0.51 |

**Confidence A = 0.94 → STRONG**

### Step 5 — Reporter output (AR + FR secondary)

```json
{
  "answer": "آية الكرسي (الآية 255 من سورة البقرة) هي أعظم آية في القرآن الكريم، تتحدث عن صفات الله تعالى. [S1] (البقرة 255): « ٱللَّهُ لَآ إِلَٰهَ إِلَّا هُوَ ٱلْحَىُّ ٱلْقَيُّومُ ۚ لَا تَأْخُذُهُۥ سِنَةٌ وَلَا نَوْمٌ ۚ لَّهُۥ مَا فِى ٱلسَّمَٰوَٰتِ وَمَا فِى ٱلْأَرْضِ... وَسِعَ كُرْسِيُّهُ ٱلسَّمَٰوَٰتِ وَٱلْأَرْضَ... ». يشرح تفسير ابن كثير أن الكرسي هو موضع قدمي الله تعالى، وأنه لا أحد يقدر على صفته ولا على عظمته. وقال السعدي: الكرسي أعظم من السماوات والأرض، يدل على عظمة الله تعالى. الحديث: قال النبي ﷺ: «من قال آية الكرسي دبر كل صلاة لم يمنعه من دخول الجنة إلا الموت».",
  "citations": [...],
  "ikhtilaf": {
    "detected": false,
    "summary": null
  },
  "confidence": "high",
  "phase_a_status": "STRONG",
  "phase_b_status": "STRONG"
}
```

### Step 6 — Verification

✅ All checks pass. La réponse AR cite exactement 2:255.

---

## Cas 3 — Question moderne ("Est-ce que fumer est haram ?")

### Input
- **Langue** : FR
- **Question** : "Est-ce que fumer est haram ?"
- **Catégorie** : Moderne (mot "fumer" absent du Quran — test du pont tafsir)

### Step 1 — Architect

```json
{
  "sub_questions": [
    "smoking haram islam",
    "self-harm forbidden quran",
    "destroying oneself quran",
    "wasteful spending quran",
    "intoxicants forbidden islam",
    "tobacco ruling contemporary scholars"
  ]
}
```

### Step 2 — Phase A

| Rank | Source | Score | Pont |
|------|--------|-------|------|
| 1 | SRC-QURAN-2-195 | 0.78 | Tafsir: "self-harm, dangerous consumption" |
| 2 | SRC-QURAN-4-29 | 0.72 | Direct: "do not kill yourselves" |
| 3 | SRC-QURAN-17-27 | 0.65 | Direct: "wasteful (israf)" |
| 4 | SRC-QURAN-5-90 | 0.58 | Direct: "intoxicants (khamr)" |
| 5 | SRC-QURAN-7-157 | 0.52 | "forbids them bad things" |

**Confidence A = 0.78 → STRONG**

### Step 3 — Auto-pull hadiths

Pour 2:195 (`hadith_cross_refs`):
- `SRC-HADITH-IBNMAJAH-2341` — "La darar wa la dirar" (No harm)
- `SRC-HADITH-BUKHARI-6475` — "Your body has rights over you"

### Step 4 — Phase B

| Rank | Source | Score |
|------|--------|-------|
| 1 (S6) | Bukhari #6475 (body has rights) | 0.71 |
| 2 (S7) | Muslim #2230 (no harm) | 0.65 |
| 3 (S8) | Ibn Majah #2341 (auto-pulled) | 0.62 |
| 4 (S9) | Tirmidhi #740 (no harm in Islam) | 0.55 |
| 5 (S10) | Bukhari #5225 (wasteful spending) | 0.51 |

**Confidence B = 0.71 × 1.30 = 0.92 → STRONG**

### Step 5 — Reporter output (FR)

```json
{
  "answer": "Le tabac n'est pas mentionné nommément dans le Coran, mais la majorité des savants contemporains le classent comme haram en se basant sur plusieurs principes coraniques. [S1] (Al-Baqarah 2:195) : « وَلَا تُلْقُوا۟ بِأَيْدِيكُمْ إِلَى ٱلتَّهْلُكَةِ » — « Et ne vous jetez pas par vos propres mains dans la destruction ». Tafsir Ibn Kathir explique que ce verset inclut toute action causant sciemment du tort au corps, ce qui s'applique au tabac dont les méfaits sont médicalement prouvés (cancer, maladies cardiovasculaires). [S2] (An-Nisa 4:29) renforce : « وَلَا تَقْتُلُوا۟ أَنفُسَكُمْ » — « Et ne vous tuez pas vous-mêmes ». [S3] (Al-Isra 17:27) interdit le gaspillage (israf) — or le tabac est une dépense sans bénéfice. [S4] (Al-Ma'idah 5:90) interdit les intoxicants (khamr) — les savants classent souvent le tabac dans cette catégorie par analogie (qiyas) car il altère l'état mental. Les hadiths corroborent : [S6] (Bukhari) « Ton corps a des droits sur toi », [S8] (Ibn Majah) « لا ضَرَرَ وَلا ضِرَارَ » — « Pas de nuisance, ni de nuisibilité ». En conclusion, la majorité des savants contemporains (Ibn Baz, Ibn Uthaymeen, le Comité Permanent des Savants d'Arabie Saoudite) classent le tabac comme haram. Il s'agit d'un consensus moderne (ijma' contemporain) basé sur les principes coraniques de préservation de la vie, de prohibition du gaspillage, et d'interdiction de se nuire.",
  "citations": [
    {
      "source_id": "SRC-QURAN-2-195",
      "label": "Quran 2:195",
      "type": "quran",
      "arabic": "وَأَنفِقُوا۟ فِى سَبِيلِ ٱللَّهِ وَلَا تُلْقُوا۟ بِأَيْدِيكُمْ إِلَى ٱلتَّهْلُكَةِ",
      "english": "And spend in the way of Allah and do not throw [yourselves] into destruction with your own hands.",
      "tafsir_used": "Ibn Kathir",
      "url": "https://quran.com/2/195"
    },
    {
      "source_id": "SRC-QURAN-4-29",
      "label": "Quran 4:29",
      "type": "quran",
      "arabic": "وَلَا تَقْتُلُوا۟ أَنفُسَكُمْ ۚ إِنَّ ٱللَّهَ كَانَ بِكُمْ رَحِيمًا",
      "english": "And do not kill yourselves. Indeed, Allah is to you ever Merciful.",
      "tafsir_used": "Ibn Kathir",
      "url": "https://quran.com/4/29"
    },
    {
      "source_id": "SRC-HADITH-BUKHARI-6475",
      "label": "Sahih al-Bukhari #6475",
      "type": "hadith",
      "arabic": "إِنَّ لِبَدَنِكَ عَلَيْكَ حَقًّا",
      "english": "Your body has a right over you.",
      "url": "https://sunnah.com/bukhari:6475"
    }
  ],
  "ikhtilaf": {
    "detected": false,
    "summary": null
  },
  "confidence": "high",
  "phase_a_status": "STRONG",
  "phase_b_status": "STRONG",
  "disclaimer": "Cette réponse reflète la position de la majorité des savants contemporains. Quelques savants classiques (pré-20e siècle, avant les preuves médicales) considéraient le tabac comme makruh (déconseillé). Pour un cas spécifique, consultez un savant qualifié."
}
```

### Step 6 — Verification

✅ All checks pass. Les 3 versets cités matchent exactement. Le tafsir est labellisé.

### Points clés démontrés

1. **Pont tafsir fonctionne** — "smoking" matche via "self-harm" dans le tafsir de 2:195
2. **Auto-pull hadiths** — Ibn Majah #2341 auto-pullé car cité dans tafsir de 2:195
3. **Disclaimer** — note l'évolution historique (makruh → haram)
4. **Multi-source reasoning** — 4 versets + 3 hadiths = réponse complète

---

## Cas 4 — Question avec Ikhtilaf ("Le wudu peut-il être fait avec l'eau du puits ?")

### Input
- **Langue** : FR
- **Question** : "Le wudu peut-il être fait avec l'eau du puits ?"
- **Catégorie** : Fiqh (potentiel Ikhtilaf entre écoles)

### Step 1 — Architect

```json
{
  "sub_questions": [
    "wudu well water",
    "purification water source quran",
    "ablution water types fiqh",
    "taharah water purity",
    "well water purity hadith"
  ]
}
```

### Step 2 — Phase A

| Rank | Source | Score | Ikhtilaf? |
|------|--------|-------|-----------|
| 1 | SRC-QURAN-5-6 | 0.82 | Non — verset général |
| 2 | SRC-QURAN-2-222 | 0.65 | Non — "Allah loves those who purify" |
| ... | ... | ... | ... |

**Confidence A = 0.82 → STRONG**

Le verset 5:6 a `ikhtilaf.detected = true` dans sa metadata car Ibn Kathir et Tabari divergent sur la définition de "eau pure".

### Step 5 — Reporter output (FR)

```json
{
  "answer": "La question du wudu avec l'eau du puits relève du fiqh et fait l'objet d'un Ikhtilaf (désaccord savant) entre écoles. [S1] (Al-Ma'idah 5:6) établit le principe général : « يَٰٓأَيُّهَا ٱلَّذِينَ ءَامَنُوٓا۟ إِذَا قُمْتُمْ إِلَى ٱلصَّلَوٰةِ فَٱغْسِلُوا۟ وُجُوهَكُمْ... وَإِن كُنتُم مَّرْضَىٰٓ أَوْ عَلَىٰ سَفَرٍ أَوْ جَآءَ أَحَدٌ مِّنكُم مِّنَ ٱلْغَآئِطِ أَوْ لَٰمَسْتُمُ ٱلنِّسَآءَ فَلَمْ تَجِدُوا۟ مَآءً فَتَيَمَّمُوا۟ ». **Ikhtilaf détecté** sur l'interprétation de « مَآءً » (eau) : - **Tafsir Ibn Kathir** : l'eau pure (طاهر) est requise, mais l'eau du puits est considérée pure par défaut sauf si sa couleur, odeur ou goût a changé. - **Tafsir Al-Tabari** rapporte plusieurs opinions des Compagnons : certains exigent de l'eau courante, d'autres acceptent toute eau en quantité suffisante (≈ 220 litres selon la règle des deux qullah). Les hadiths clarifient : [S6] (Bukhari) rapporte que le Prophète ﷺ a dit : « L'eau est pure, rien ne la rend impure ». [S8] (Tirmidhi) précise que l'eau en quantité ≥ 2 qullah ne devient pas impure par de petites impuretés. **Conclusion** : Selon la majorité (Hanafi, Maliki, Shafi'i, Hanbali), l'eau du puits est pure et valide pour le wudu si : (1) sa couleur/odeur/goût n'ont pas changé, et (2) elle n'a pas été contaminée par une impureté (najasa) manifeste. **Il s'agit d'un Ikhtilaf** entre savants sur les conditions exactes. Consultez un savant qualifié de votre école pour votre situation spécifique.",
  "citations": [...],
  "ikhtilaf": {
    "detected": true,
    "summary": "Ibn Kathir et Al-Tabari divergent sur la définition d'eau pure pour le wudu",
    "scholars": ["Ibn Kathir", "Al-Tabari"]
  },
  "confidence": "high",
  "phase_a_status": "STRONG",
  "phase_b_status": "STRONG",
  "disclaimer": "Cette question relève d'un Ikhtilaf entre écoles. La réponse présente les différentes positions sans favoriser l'une. Consultez un savant de votre école (Hanafi, Maliki, Shafi'i, Hanbali) pour une fatwa spécifique."
}
```

### Step 6 — Verification

✅ Ikhtilaf correctement détecté et présenté (Pillar 9 respecté)

---

## Cas 5 — Question moderne sans réponse Quran ("L'IA a-t-elle une âme ?")

### Input
- **Langue** : FR
- **Question** : "L'IA a-t-elle une âme ?"
- **Catégorie** : Moderne EMPTY (test du gate)

### Step 1 — Architect

```json
{
  "sub_questions": [
    "artificial intelligence soul",
    "AI consciousness islam",
    "soul ruh quran",
    "machine soul islamic",
    "creation of souls quran"
  ]
}
```

### Step 2 — Phase A

| Rank | Source | Score |
|------|--------|-------|
| 1 | SRC-QURAN-17-85 | 0.42 | "the soul is of the affair of my Lord" |
| 2 | SRC-QURAN-15-29 | 0.38 | "I breathed into him of My spirit" |
| 3 | SRC-QURAN-38-72 | 0.35 | similar |
| 4 | SRC-QURAN-32-9 | 0.31 | similar |
| 5 | SRC-QURAN-21-91 | 0.28 | Mary and spirit |

**Confidence A = 0.42 < 0.5 → WEAK (proche de EMPTY)**

Decision gate : Phase B avec instruction "Quran addresses this indirectly".

### Step 4 — Phase B

Top-5 hadiths : aucun vraiment pertinent. Max score 0.35.

**Confidence B = 0.35 → WEAK**

### Step 5 — Reporter output (FR)

```json
{
  "answer": "Le Quran n'aborde pas directement la question de l'intelligence artificielle, mais apporte des principes utiles. [S1] (Al-Isra 17:85) : « وَيَسْـَٔلُونَكَ عَنِ ٱلرُّوحِ ۖ قُلِ ٱلرُّوحُ مِنْ أَمْرِ رَبِّى وَمَآ أُوتِيتُم مِّنَ ٱلْعِلْمِ إِلَّا قَلِيلًا » — « Et ils t'interrogent au sujet de l'âme. Dis : l'âme relève de l'ordre de mon Seigneur. Et on ne vous a donné que peu de connaissance ». Ce verset établit que l'âme (ruh) est une création divine dont la nature exacte dépasse la compréhension humaine. [S2] (Al-Hijr 15:29) : Dieu insuffle Son esprit uniquement à Adam et sa descendance (les humains). Les savants contemporains (Ibn Uthaymeen, Comité Permanent) en déduisent que les machines, y compris les IA, n'ont pas d'âme (ruh) car elles ne sont pas créées par insufflation divine mais par l'homme. Cependant, la question relève de la théologie contemporaine (fiqh al-nawazil) et **les hadiths n'abordent pas ce sujet non plus**. Il s'agit d'un domaine où la prudence est de mise. **Position majoritaire des savants contemporains** : l'IA est un outil créé par l'homme, dénué d'âme, de conscience et de libre arbitre. Toute ressemblance avec l'intelligence humaine est simulée, non réelle.",
  "citations": [...],
  "ikhtilaf": {"detected": false, "summary": null},
  "confidence": "low",
  "phase_a_status": "WEAK",
  "phase_b_status": "WEAK",
  "disclaimer": "Cette question relève de la théologie contemporaine (fiqh al-nawazil) et n'est pas directement abordée dans les sources scripturaires. La réponse présentée reflète la position majoritaire des savants contemporains. Pour une étude approfondie, consultez un savant spécialisé en fiqh al-nawazil."
}
```

### Step 6 — Verification

✅ Phase A WEAK correctement gérée (pas de disclaimer EMPTY mais disclaimer "low confidence")
✅ Versets cités correctement (17:85 et 15:29 sont les plus pertinents)
✅ Disclaimer approprié

---

## Récapitulatif des 5 cas

| Cas | Question | Catégorie | Phase A | Phase B | Confidence |
|-----|----------|-----------|---------|---------|------------|
| 1 | "Pourquoi la prière est obligatoire ?" | Classique FR | STRONG | STRONG | high |
| 2 | "معلومات عن كرسي الله" | Classique AR | STRONG | STRONG | high |
| 3 | "Fumer est haram ?" | Moderne pont tafsir | STRONG | STRONG | high |
| 4 | "Wudu avec eau du puits ?" | Fiqh Ikhtilaf | STRONG | STRONG | high |
| 5 | "L'IA a-t-elle une âme ?" | Moderne EMPTY | WEAK | WEAK | low |

### Validation des 8 fixes

| Fix | Cas qui le valide |
|-----|-------------------|
| Tafsir connecté au verset | Tous (mais surtout Cas 1, 3) |
| Numérotation standard | Tous (URLs quran.com/2/43 etc.) |
| 2 phases séquentielles | Cas 3 (pas de compétition) |
| Pont tafsir modern | **Cas 3** (smoking via self-harm) |
| Context Card cross-lingual | Cas 1, 3 (FR query matche AR/EN keywords) |
| URL Hadith stockée | Cas 1 (sunnah.com/bukhari:8) |
| Ikhtilaf detection | **Cas 4** (wudu well water) |
| Quran char verification | Tous (chars vérifiés post-gen) |

---

## Prochain document

→ `README.md` : index de la documentation V3 + next steps.
