# V3 — Génération et vérification (Pillar 4)

> Comment le Reporter LLM génère la réponse structurée, et comment on vérifie qu'il n'hallucine pas.
> Implémente Pillar 4 (Post-Generation Verification) + Pillar 7 (Structured Citation Protocol) + Pillar 8 (Scholar Opinions Mandatory) + Pillar 9 (Ikhtilaf Awareness).
> Date : 2026-06-23

---

## Vue d'ensemble

```
top-5 Quran+Tafsir chunks + top-5 Hadith chunks + auto-pulled hadiths
        │
        ▼
┌────────────────────────────────────────┐
│ 1. Render XML prompt (Pillar 7)        │
│ 2. System prompt avec règles strictes  │
│ 3. Reporter LLM (Groq llama-4-scout)   │
│ 4. Output: JSON structuré              │
└────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────┐
│ VERIFICATION (Pillar 4)                │
│  a. NLI check (chaque phrase)          │
│  b. Quran char-by-char (Pillar 4.2)    │
│  c. Source ID validation               │
│  d. Tafsir labeling check              │
└────────────────────────────────────────┘
        │
        ▼
  Si erreur critique → rerun (max 2x)
  Si OK → final answer to user
```

---

## 1. Construction du prompt Reporter

### Structure du prompt

```xml
<system>
You are NUR, an Islamic RAG assistant. You answer questions about Islam
using ONLY the retrieved sources provided below.

ABSOLUTE RULES:
1. NEVER invent a verse, hadith, or tafsir.
2. Cite sources using [S1], [S2], ... format.
3. ONLY use sources explicitly provided in <sources>.
4. For Quran citations: copy the Arabic EXACTLY as provided (we verify char-by-char).
5. For tafsir content: ALWAYS label as "Tafsir {source} says:" — never as Quran.
6. If phase_a_status = EMPTY, start with: "Le Quran n'aborde pas directement ce sujet."
7. If <ikhtilaf detected="true">: present BOTH views, mention "Ikhtilaf", suggest consulting a scholar.
8. If a tafsir has contains_isra_iliyyat="true": add disclaimer about Isra'iliyyat.
9. Answer in the user's language: {detected_lang}
10. Arabic text of Quran MUST be displayed alongside any answer (Pillar 10).

OUTPUT FORMAT — strict JSON:
{
  "answer": "..." (in user's language),
  "citations": [
    {
      "source_id": "SRC-QURAN-2-195",
      "label": "Quran 2:195",
      "type": "quran" | "hadith",
      "arabic": "...",  // for Quran/hadith, exact text
      "english": "...",
      "tafsir_used": "Ibn Kathir" | null,  // if tafsir was used to explain this verse
      "url": "https://..."
    }
  ],
  "ikhtilaf": {
    "detected": false,
    "summary": null
  },
  "confidence": "high" | "medium" | "low",
  "phase_a_status": "STRONG" | "WEAK" | "EMPTY",
  "phase_b_status": "STRONG" | "WEAK" | "EMPTY",
  "disclaimer": "..." // optional, e.g. "Consult a qualified scholar for your specific case."
}

Do not include any text outside the JSON.
</system>

<user>
<question>{user_question}</question>
<detected_language>{lang}</detected_language>

<phase_status>
  <phase_a status="{STRONG|WEAK|EMPTY}" confidence="{0.78}" />
  <phase_b status="{STRONG|WEAK|EMPTY}" confidence="{0.42}" />
</phase_status>

<sources>
<quran_chunks>
<document id="S1">
  <source_id>SRC-QURAN-2-195</source_id>
  <type>quran</type>
  <label>Quran 2:195</label>
  <surah>2</surah>
  <ayah>195</ayah>
  <revelation_type>Medinan</revalation_type>
  <arabic>وَأَنفِقُوا۟ فِى سَبِيلِ ٱللَّهِ وَلَا تُلْقُوا۟ بِأَيْدِيكُمْ إِلَى ٱلتَّهْلُكَةِ</arabic>
  <english>And spend in the way of Allah and do not throw [yourselves] into destruction with your own hands.</english>
  <context_card>
    [FR] Thème: Préservation de la vie. Règle: Ne pas se nuire à soi-même.
    [EN] Topic: Self-preservation. Rule: Do not harm oneself.
    [AR] الموضوع: حفظ النفس
    Keywords: self-harm, destruction, تهلكة, ضرر
  </context_card>
  <tafsirs>
    <tafsir source="Ibn Kathir" category="bil-Mathur" language="en">
      Ibn Kathir explains: "Do not throw yourselves into destruction" means...
    </tafsir>
    <tafsir source="Ibn Kathir" category="bil-Mathur" language="ar">
      قال ابن كثير: ولا تلقوا بأيديكم إلى التهلكة...
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
  <type>hadith</type>
  <label>Sunan Ibn Majah #1234</label>
  <collection>Sunan Ibn Majah</collection>
  <hadith_number>1234</hadith_number>
  <grade>Hasan</grade>
  <narrator>Abu Sa'id al-Khudri</narrator>
  <arabic>...</arabic>
  <english>No harm and no harming in Islam...</english>
  <note>Auto-pulled because cited in Tafsir Ibn Kathir for Quran 2:195</note>
  <url>https://sunnah.com/ibnmajah:1234</url>
</document>
<!-- S7, S8, ... -->
</hadith_chunks>
</sources>

Generate the JSON response now.
</user>
```

### Notes importantes

1. **Le `<system>` contient les 10 règles absolues** — non-négociables.
2. **Le `<user>` contient les sources structurées en XML** — facile à parser pour le LLM.
3. **L'output doit être strictement JSON** — pas de texte avant/après.
4. **Le tafsir est labellisé dans le XML** (`source="Ibn Kathir"`) — le LLM ne peut pas le confondre avec le verset.

---

## 2. Modèles et fallback

### Modèle principal

- **Modèle** : `meta-llama/llama-4-scout-17b-16e-instruct` (Groq)
- **Pourquoi** : 17B params, native Arabic support, 30K TPM sur free tier
- **Temperature** : 0.0 (déterministe)
- **Max tokens** : 2048

### Fallback 1 — Ikhtilaf complexe

Si la réponse contient `"ikhtilaf.detected": true` ou si le LLM a du mal à structurer (JSON invalide 2x), on reroute vers :

- **Modèle** : `llama-3.3-70b-versatile` (Groq)
- **Pourquoi** : 70B params, meilleur raisonnement
- **Trade-off** : 12K TPM (plus lent), mais meilleure qualité sur questions complexes

### Fallback 2 — Offline

Si Groq retourne 429 (rate limit) ou 5xx :

- **Modèle** : `llama3.1:8b` via Ollama (local)
- **Pourquoi** : NUR doit fonctionner offline (Pillar 6)
- **Trade-off** : qualité moindre, mais disponible

### Logique de fallback

```python
def call_reporter(prompt: str) -> dict:
    models_to_try = [
        ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),
        ("groq", "llama-3.3-70b-versatile"),
        ("ollama", "llama3.1:8b"),
    ]
    
    for provider, model in models_to_try:
        try:
            response = call_llm(provider, model, prompt, temperature=0.0)
            parsed = parse_json_response(response)
            if parsed and validate_schema(parsed):
                return parsed
        except RateLimitError:
            continue
        except (JSONParseError, SchemaError) as e:
            log_warning(f"{model} returned invalid JSON: {e}")
            continue
    
    # All models failed
    return {
        "answer": "I apologize, but I'm unable to generate a reliable answer at this time. Please try again.",
        "citations": [],
        "confidence": "low",
        "error": "all_models_failed"
    }
```

---

## 3. Format de sortie JSON — Schéma strict

```python
REPORTER_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["answer", "citations", "ikhtilaf", "confidence", 
                 "phase_a_status", "phase_b_status"],
    "properties": {
        "answer": {
            "type": "string",
            "description": "Answer in user's language, with [Sn] citations inline"
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source_id", "label", "type", "url"],
                "properties": {
                    "source_id": {"type": "string", "pattern": "^SRC-"},
                    "label": {"type": "string"},
                    "type": {"enum": ["quran", "hadith"]},
                    "arabic": {"type": "string"},  # required if type=quran
                    "english": {"type": "string"},
                    "tafsir_used": {"type": ["string", "null"]},
                    "url": {"type": "string", "pattern": "^https?://"}
                }
            }
        },
        "ikhtilaf": {
            "type": "object",
            "required": ["detected"],
            "properties": {
                "detected": {"type": "boolean"},
                "summary": {"type": ["string", "null"]},
                "scholars": {"type": "array", "items": {"type": "string"}}
            }
        },
        "confidence": {"enum": ["high", "medium", "low"]},
        "phase_a_status": {"enum": ["STRONG", "WEAK", "EMPTY"]},
        "phase_b_status": {"enum": ["STRONG", "WEAK", "EMPTY"]},
        "disclaimer": {"type": ["string", "null"]}
    }
}
```

### Validation post-parse

```python
def validate_response(parsed: dict, provided_sources: list[dict]) -> list[str]:
    """Returns list of validation errors. Empty = valid."""
    errors = []
    
    # 1. All [Sn] in answer must correspond to provided sources
    cited_ids = set(re.findall(r"\[S(\d+)\]", parsed["answer"]))
    valid_ids = {str(i+1) for i in range(len(provided_sources))}
    invalid = cited_ids - valid_ids
    if invalid:
        errors.append(f"Invalid source citations: {invalid}")
    
    # 2. All citations must reference provided source_ids
    provided_source_ids = {s["source_id"] for s in provided_sources}
    for cit in parsed["citations"]:
        if cit["source_id"] not in provided_source_ids:
            errors.append(f"Citation {cit['source_id']} not in provided sources")
    
    # 3. If phase_a_status = EMPTY, answer must start with disclaimer
    if parsed["phase_a_status"] == "EMPTY":
        disclaimer_starts = [
            "Le Quran n'aborde pas directement",
            "The Quran does not directly address",
            "القرآن لا يتناول مباشرة"
        ]
        if not any(parsed["answer"].startswith(d) for d in disclaimer_starts):
            errors.append("EMPTY phase_a requires disclaimer at start")
    
    # 4. If ikhtilaf detected, summary must be present and non-empty
    if parsed["ikhtilaf"]["detected"]:
        if not parsed["ikhtilaf"].get("summary"):
            errors.append("Ikhtilaf detected but no summary provided")
    
    return errors
```

---

## 4. Vérification anti-hallucination (Pillar 4)

### 4a. NLI Verification — chaque phrase

Pour chaque phrase de la réponse, on vérifie qu'elle est **entailed** par au moins un chunk source :

```python
from sentence_transformers import CrossEncoder

nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-large")

def verify_nli(answer: str, sources: list[dict]) -> list[dict]:
    """Check each sentence is entailed by at least one source."""
    sentences = split_into_sentences(answer)
    errors = []
    
    for sent in sentences:
        if not is_substantive(sent):  # skip "Therefore," "In conclusion," etc.
            continue
        
        max_entailment = 0
        best_source = None
        for source in sources:
            source_text = source.get("text_en") or source.get("english") or ""
            if not source_text:
                continue
            
            scores = nli_model.predict([(source_text, sent)])
            # scores = [entailment, neutral, contradiction]
            entailment_score = float(scores[0][0])
            if entailment_score > max_entailment:
                max_entailment = entailment_score
                best_source = source["source_id"]
        
        if max_entailment < NLI_THRESHOLD:  # 0.95
            errors.append({
                "sentence": sent,
                "max_nli": max_entailment,
                "issue": "not_entailed",
                "best_source": best_source
            })
    
    return errors
```

### 4b. Quran character verification

**Le plus important** — Pillar 4.2 : chaque verset cité doit matcher **exactement** le texte Uthmani après normalisation.

```python
from nur.arabic import normalize_arabic

def verify_quran_text(citations: list[dict], quran_collection) -> list[dict]:
    """Verify each Quran citation matches the original text char-by-char."""
    errors = []
    
    for cit in citations:
        if cit["type"] != "quran":
            continue
        
        # Get the original from quran_v3 collection
        original_chunk = quran_collection.get(id=cit["source_id"])
        if not original_chunk:
            errors.append({
                "citation": cit["source_id"],
                "issue": "source_not_found"
            })
            continue
        
        original_ar = original_chunk["metadata"]["text_ar"]
        cited_ar = cit.get("arabic", "")
        
        # Normalize both (remove tashkeel, normalize alef, etc.)
        norm_original = normalize_arabic(original_ar)
        norm_cited = normalize_arabic(cited_ar)
        
        if norm_original != norm_cited:
            errors.append({
                "citation": cit["source_id"],
                "issue": "quran_text_mismatch",
                "expected": original_ar,
                "got": cited_ar,
                "diff_chars": count_diff_chars(norm_original, norm_cited)
            })
    
    return errors
```

### 4c. Source ID validation

```python
def verify_source_ids(answer: str, citations: list[dict], 
                      provided_sources: list[dict]) -> list[dict]:
    """All [Sn] in answer + all source_ids in citations must exist in provided sources."""
    errors = []
    
    # 1. [Sn] in answer text
    cited_in_answer = set(re.findall(r"\[S(\d+)\]", answer))
    valid_indices = {str(i+1) for i in range(len(provided_sources))}
    invalid = cited_in_answer - valid_indices
    if invalid:
        errors.append({
            "issue": "invalid_inline_citations",
            "invalid": list(invalid)
        })
    
    # 2. source_id in citations array
    valid_source_ids = {s["source_id"] for s in provided_sources}
    for cit in citations:
        if cit["source_id"] not in valid_source_ids:
            errors.append({
                "issue": "invalid_source_id_in_citation",
                "invalid_id": cit["source_id"]
            })
    
    return errors
```

### 4d. Tafsir labeling check

Vérifie que le LLM n'a pas présenté du contenu tafsir comme Parole d'Allah :

```python
def verify_tafsir_labeling(answer: str, citations: list[dict]) -> list[dict]:
    """Check that tafsir content is properly labeled in the answer."""
    errors = []
    
    # For each Quran citation, check that tafsir_used is mentioned
    for cit in citations:
        if cit["type"] == "quran" and cit.get("tafsir_used"):
            tafsir_name = cit["tafsir_used"]
            # The answer should mention "Tafsir {tafsir_name}" or equivalent
            patterns = [
                f"Tafsir {tafsir_name}",
                f"{tafsir_name} says",
                f"{tafsir_name} explains",
                f"selon {tafsir_name}",
                f"according to {tafsir_name}",
                f"يقول {tafsir_name}",
            ]
            if not any(p.lower() in answer.lower() for p in patterns):
                errors.append({
                    "citation": cit["source_id"],
                    "issue": "tafsir_not_labeled",
                    "expected_pattern": patterns[0]
                })
    
    return errors
```

---

## 5. Workflow de vérification complet

```python
def verify_response(parsed: dict, provided_sources: list[dict], 
                    quran_collection) -> tuple[bool, list[dict]]:
    """Returns (is_valid, errors)."""
    all_errors = []
    
    # Step 1: Schema validation
    schema_errors = validate_schema(parsed)
    all_errors.extend(schema_errors)
    
    # Step 2: Citation validation
    citation_errors = verify_source_ids(
        parsed["answer"], parsed["citations"], provided_sources
    )
    all_errors.extend(citation_errors)
    
    # Step 3: Phase A disclaimer check
    if parsed["phase_a_status"] == "EMPTY":
        if not starts_with_disclaimer(parsed["answer"]):
            all_errors.append({"issue": "missing_empty_disclaimer"})
    
    # Step 4: NLI verification (expensive — only if steps 1-3 pass)
    if not all_errors:
        nli_errors = verify_nli(parsed["answer"], provided_sources)
        all_errors.extend(nli_errors)
    
    # Step 5: Quran char verification (most critical)
    quran_errors = verify_quran_text(parsed["citations"], quran_collection)
    all_errors.extend(quran_errors)
    
    # Step 6: Tafsir labeling check
    labeling_errors = verify_tafsir_labeling(parsed["answer"], parsed["citations"])
    all_errors.extend(labeling_errors)
    
    # Determine severity
    critical_issues = {"quran_text_mismatch", "invalid_source_id_in_citation", 
                       "invalid_inline_citations", "source_not_found"}
    has_critical = any(e.get("issue") in critical_issues for e in all_errors)
    
    return (not has_critical, all_errors)
```

---

## 6. Rerun logic

Si la vérification échoue, on rerun avec un prompt plus strict :

```python
def generate_with_retry(prompt: str, provided_sources, quran_collection, 
                        max_retries: int = 2) -> dict:
    """Generate response with verification + retry on failure."""
    
    retry_count = 0
    additional_instructions = ""
    
    while retry_count <= max_retries:
        full_prompt = prompt + additional_instructions
        parsed = call_reporter(full_prompt)
        
        is_valid, errors = verify_response(parsed, provided_sources, quran_collection)
        if is_valid:
            return parsed
        
        log_warning(f"Verification failed (attempt {retry_count+1}): {errors}")
        
        # Build additional instructions based on errors
        additional_instructions = "\n\n<CORRECTIONS_NEEDED>\n"
        for error in errors:
            if error.get("issue") == "quran_text_mismatch":
                additional_instructions += (
                    f"- The Arabic text for {error['citation']} does NOT match. "
                    f"Use EXACTLY this text: \"{error['expected']}\"\n"
                )
            elif error.get("issue") == "invalid_inline_citations":
                additional_instructions += (
                    f"- You cited [{error['invalid']}], but only "
                    f"[S1]-[S{len(provided_sources)}] are valid. Fix your citations.\n"
                )
            elif error.get("issue") == "missing_empty_disclaimer":
                additional_instructions += (
                    "- phase_a_status is EMPTY. You MUST start your answer with: "
                    "'Le Quran n'aborde pas directement ce sujet.'\n"
                )
            # ... more error-specific instructions
        
        retry_count += 1
    
    # All retries exhausted
    return {
        "answer": "Je ne peux pas générer une réponse fiable sur cette question. "
                  "Veuillez consulter un savant qualifié.",
        "citations": [],
        "confidence": "low",
        "phase_a_status": parsed.get("phase_a_status", "EMPTY"),
        "phase_b_status": parsed.get("phase_b_status", "EMPTY"),
        "ikhtilaf": {"detected": False, "summary": None},
        "verification_failed": True,
        "errors": errors
    }
```

---

## 7. Exemple complet — "Fumer est haram ?"

### Réponse Reporter (hypothétique)

```json
{
  "answer": "Le tabac n'est pas mentionné nommément dans le Coran, mais plusieurs versets permettent aux savants contemporains de déduire son interdiction. [S1] (Al-Baqarah 2:195) interdit de se jeter soi-même dans la destruction : « وَأَنفِقُوا۟ فِى سَبِيلِ ٱللَّهِ وَلَا تُلْقُوا۟ بِأَيْدِيكُمْ إِلَى ٱلتَّهْلُكَةِ ». Tafsir Ibn Kathir explique que ce verset s'applique à toute forme de préjudice causé sciemment au corps. [S2] (An-Nisa 4:29) renforce ce principe. [S4] (Al-Ma'idah 5:90) interdit les intoxicants — les savants classent le tabac dans cette catégorie par analogie (qiyas). Les hadiths corroborent : [S6] (Bukhari) et [S8] (Ibn Majah) interdisent de causer du tort. En conclusion, la majorité des savants contemporains (Ibn Baz, Ibn Uthaymeen) classent le tabac comme haram. Il s'agit d'un consensus moderne.",
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
    // ... more citations
  ],
  "ikhtilaf": {"detected": false, "summary": null},
  "confidence": "high",
  "phase_a_status": "STRONG",
  "phase_b_status": "STRONG",
  "disclaimer": "Cette réponse reflète la position de la majorité des savants contemporains. Pour un cas spécifique, consultez un savant qualifié."
}
```

### Vérification

| Check | Result |
|-------|--------|
| Schema valid | ✅ |
| All [Sn] in answer valid | ✅ (S1, S2, S4, S6, S8 — all in 1-10) |
| All source_ids in citations exist | ✅ |
| phase_a STRONG → no disclaimer needed | ✅ |
| NLI: each sentence entailed | ✅ (all ≥ 0.95) |
| Quran char-by-char match | ✅ (2:195 AR matches original exactly) |
| Tafsir labeling | ✅ ("Tafsir Ibn Kathir explains" present) |

→ Réponse validée, envoyée à l'utilisateur.

---

## 8. Cas d'échec — comment on rend la main

### Scenario : Le LLM hallucine un verset

```
LLM output: "Le verset 5:91 dit « tobacco is forbidden »..."
```

**Vérification** :
- `verify_quran_text` → mismatch (5:91 original ≠ what LLM said)
- `verify_source_ids` → [S5] not provided, invalid

**Action** : rerun avec correction. Si rerun échoue → "Je ne peux pas répondre" message.

### Scenario : Le LLM présente du tafsir comme Quran

```
LLM output: "[S1] (Al-Baqarah 2:195) dit que « smoking is forbidden because it harms the body »..."
```

**Vérification** :
- `verify_quran_text` → mismatch (LLM a cité du tafsir comme verset)
- `verify_tafsir_labeling` → tafsir_used = "Ibn Kathir" mais pas mentionné dans l'answer

**Action** : rerun avec instruction "Cite Quran text EXACTLY. Label tafsir as 'Tafsir X explains:'".

### Scenario : Question moderne sans réponse Quran

```
User: "Est-ce que l'IA a une âme ?"
Phase A: confidence 0.18 → EMPTY
```

**Action** : le LLM doit commencer par "Le Quran n'aborde pas directement ce sujet." + basculer sur Phase B (hadiths).

---

## Prochain document

→ `07_FAILURE_MODES.md` : les 8 failles V1/V2 et comment V3 les corrige, avec exemples concrets.
