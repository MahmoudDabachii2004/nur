# NUR — Data Sources & Provenance

> **MANDATORY READ for anyone auditing NUR's data integrity.**

> ⚠️ **V3 SUPERSEDE NOTICE (2026-06-23)** — The 4-collection structure described below (Quran, Hadith, Tafsir AR, Tafsir EN) reflects V1/V2. V3 uses 2 collections: `quran_v3` (with tafsir integrated in the same chunk) and `hadith_v3`. The data sources (alquran.cloud, meeAtif, spa5k/tafsir_api) are the same. V3 adds 3 new tafsirs (Ibn Kathir AR, Al-Tabari AR, As-Sa'di AR). **For V3 data sources, read `docs/v3/01_DATA_SOURCES.md`.** The audit procedure (section 7) remains valid.

---
> This document records exactly where every piece of Islamic text in the NUR
> database came from, when it was downloaded, under what license, and how to
> re-verify or re-download it.
>
> **Why this matters (theological integrity):** A wrong religious answer is
> worse than no answer. If a user one day reports that a hadith in NUR's
> output is fabricated (Mawdu') or that a Quranic verse is mis-transcribed,
> we MUST be able to trace the chunk back to its original source in seconds.
> This document is that trace.

---

## 1. Summary — The Three Data Sources

NUR's database contains 4 collections totaling 52,446 chunks. They come from
3 upstream sources, downloaded once during Phase 1 ingestion on **2026-06-21**.

| # | Source | What we take | License | Chunks in NUR |
|---|--------|--------------|---------|---------------|
| 1 | [alquran.cloud](https://alquran.cloud) (run by [Islamic Network](https://islamic.network)) | Quran — Uthmani Arabic + Saheeh International English + surah metadata | Free public API, no license file; Islamic Network is an open-source Islamic services foundation | 6,236 (Quran) |
| 2 | [meeAtif/hadith_datasets](https://huggingface.co/datasets/meeAtif/hadith_datasets) on HuggingFace | Hadith — 6 canonical collections (Kutub al-Sittah), AR + EN + grades + sunnah.com URLs | MIT | 33,738 (Hadith) |
| 3 | [spa5k/tafsir_api](https://github.com/spa5k/tafsir_api) on GitHub | Tafsir Ibn Kathir — Arabic + English editions | MIT | 6,236 × 2 = 12,472 (Tafsir AR + EN) |

**Total**: 52,446 chunks across 4 ChromaDB collections + 4 sparse JSON files.

---

## 2. Source 1 — Quran (alquran.cloud)

### 2.1 What we download

| File | Edition ID | Content | Count |
|------|-----------|---------|-------|
| `data/quran/quran-uthmani.json` | `quran-uthmani` | Uthmani Arabic script (source of truth) | 6,236 ayahs |
| `data/quran/en.sahih.json` | `en.sahih` | Saheeh International English translation | 6,236 ayahs |
| `data/quran/quran-meta.json` | (metadata endpoint) | Surah names (AR + EN), revelation type (Meccan/Medinan), ayah counts | 114 surahs |

### 2.2 Upstream source

- **Provider**: Al Quran Cloud — <https://alquran.cloud>
- **API base**: `http://api.alquran.cloud/v1`
- **Maintainer**: Islamic Network — <https://islamic.network> (a foundation that builds free, open-source digital Islamic services)
- **License**: No explicit LICENSE file. Islamic Network describes the project as "free digital services for the Islamic community" and the API has no rate limits for reasonable use. The Uthmani Quranic text itself is the Word of Allah — it is not subject to copyright. The Saheeh International translation is used under their published edition; alquran.cloud hosts it as a public service.
- **No API key required.**

### 2.3 Download script

`scripts/01_download_quran.py` — run with `python scripts/01_download_quran.py`

The script validates the download by checking that:
- Surah count == 114
- Total ayah count == 6,236

If either check fails, it prints a warning (data may have been truncated mid-download).

### 2.4 Verification commands

To re-verify the data matches the upstream:
```bash
# Re-download to a temp location and diff against the committed copy
curl -s http://api.alquran.cloud/v1/quran/quran-uthmani | python -c "import sys,json; d=json.load(sys.stdin); print(len(d['data']['surahs']))"
# Expected: 114 surahs

# Check the first ayah of Al-Fatihah
curl -s "http://api.alquran.cloud/v1/ayah/1:1/quran-uthmani" | python -c "import sys,json; d=json.load(sys.stdin); print(d['data']['text'])"
# Expected: بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ
```

---

## 3. Source 2 — Hadith (meeAtif/hadith_datasets on HuggingFace)

### 3.1 What we download

Six JSON files, one per canonical collection (Kutub al-Sittah):

| Collection | Sunnah.com slug | Approximate hadith count |
|---|---|---|
| Sahih al-Bukhari | `bukhari` | ~7,008 |
| Sahih Muslim | `muslim` | ~5,362 |
| Sunan Abi Dawud | `abudawud` | ~4,590 |
| Jami` at-Tirmidhi | `tirmidhi` | ~3,956 |
| Sunan an-Nasa'i | `nasai` | ~5,662 |
| Sunan Ibn Majah | `ibnmajah` | ~4,341 |
| **Total** | | **33,738 hadiths** |

### 3.2 Upstream source

- **Provider**: HuggingFace dataset `meeAtif/hadith_datasets` — <https://huggingface.co/datasets/meeAtif/hadith_datasets>
- **Raw file URL pattern**: `https://huggingface.co/datasets/meeAtif/hadith_datasets/resolve/main/{Collection Name}.json`
- **License**: **MIT** (declared in the dataset card YAML frontmatter)
- **Dataset card description**: "An open-source collection of authenticated Hadiths from the six major books of Sunnah, available in both JSON and CSV formats for research, study, and teaching purposes."
- **No API key required** (HuggingFace public dataset, direct resolve URL).

### 3.3 Per-hadith fields

Each hadith entry includes:
- `text_ar` — Arabic text (source of truth)
- `text_en` — English translation
- `collection` — e.g. "Sahih al-Bukhari"
- `hadith_number` — sunnah.com numbering
- `grade` — e.g. "Sahih (Darussalam)", "Hasan (Darussalam)", "Da'if (Darussalam)"
- `grade_level` — normalized: `sahih` | `hasan` | `daif` | etc.
- `grade_weight` — precomputed multiplier (1.30 / 1.10 / 0.50 / 0.00) used by retrieval scoring (Pillar 3)
- `narrator` — e.g. "Abu Huraira"
- `chapter_title_en` / `chapter_title_ar` — book chapter
- `url` — direct sunnah.com link (e.g. `https://sunnah.com/bukhari:1`)

**Critical note**: The sunnah.com URLs are stored in the dataset — we do NOT construct them dynamically. This avoids the off-by-one numbering issue that arises when sunnah.com re-indexes a collection. (Recorded in `02_download_hadith.py` docstring: *"sunnah.com URLs: stockées dans metadata Phase 1, NE PAS construire dynamiquement"*.)

### 3.4 Download script

`scripts/02_download_hadith.py` — run with `python scripts/02_download_hadith.py`

Saves each collection to `data/hadith/meetif/{Collection Name}.json`.

### 3.5 Verification commands

```bash
# Re-download Bukhari to a temp file and compare
curl -sL "https://huggingface.co/datasets/meeAtif/hadith_datasets/resolve/main/Sahih%20al-Bukhari.json" \
  | python -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} hadiths'); print(d[0]['text_en'][:80])"
# Expected: ~7008 hadiths, first hadith about revelation
```

---

## 4. Source 3 — Tafsir Ibn Kathir (spa5k/tafsir_api on GitHub)

### 4.1 What we download

Two editions of Tafsir Ibn Kathir, 114 surah files each:

| Edition | Path in upstream repo | Local path | Author |
|---------|----------------------|------------|--------|
| Arabic | `tafsir/ar-tafsir-ibn-kathir/{surah}.json` | `data/tafsir/ar/{surah}.json` | Hafiz Ibn Kathir (1301-1373 CE) |
| English | `tafsir/en-tafisr-ibn-kathir/{surah}.json` ⚠️ | `data/tafsir/en/{surah}.json` | Hafiz Ibn Kathir (translated) |

> ⚠️ **Known upstream typo**: The English folder is named `en-tafisr-ibn-kathir` (missing the second `s` in "tafsir"). This is a typo in the upstream spa5k repo that we MUST match exactly — the download script has a comment flagging it. If spa5k ever fixes the typo, our download script will need updating.

### 4.2 Upstream source (chain of custody)

```
qul.tarteel.ai
   ├── /resources/tafsir/22  (Arabic Ibn Kathir)
   └── /resources/tafsir/35  (English Ibn Kathir)
        ↓
   spa5k/tafsir_api  (GitHub, MIT license, copyright 2023 Spark)
        ↓
   NUR download script (scripts/03_download_tafsir.py)
        ↓
   data/tafsir/ar/  +  data/tafsir/en/
```

- **Provider**: spa5k/tafsir_api — <https://github.com/spa5k/tafsir_api>
- **Raw file URL pattern**: `https://raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir/{edition}/{surah}.json`
- **License**: **MIT** (LICENSE file in repo root, copyright 2023 Spark)
- **Original source**: spa5k themselves pulled the data from `qul.tarteel.ai` (Tarteel's Quranic resources site). This is documented in their README's tafsir source table:
  - AR Ibn Kathir: <https://qul.tarteel.ai/resources/tafsir/22>
  - EN Ibn Kathir: <https://qul.tarteel.ai/resources/tafsir/35>
- **Why Ibn Kathir**: Most widely accepted classical tafsir in Sunni Islam. Uses the "Quran-explains-Quran" methodology (verses cross-reference each other) and includes hadith commentary — essential for Phase 6 (Quran ↔ Hadith cross-references).

### 4.3 Per-section fields

Each surah file is a JSON array of section objects:
- `text` — the tafsir text for this section
- `ayah` — the ayah number (or ayah range) this section explains
- `surah` — the surah number (1-114)

### 4.4 Download script

`scripts/03_download_tafsir.py` — run with `python scripts/03_download_tafsir.py`

Downloads both AR and EN editions in one pass, then writes `data/tafsir/_summary.json` with counts.

### 4.5 Verification commands

```bash
# Check the first surah's Arabic tafsir
curl -sL https://raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir/ar-tafsir-ibn-kathir/1.json \
  | python -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} sections'); print(d[0]['text'][:100])"

# Check the English edition (note the typo in the folder name!)
curl -sL https://raw.githubusercontent.com/spa5k/tafsir_api/main/tafsir/en-tafisr-ibn-kathir/1.json \
  | python -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} sections'); print(d[0]['text'][:100])"
```

---

## 5. Chain of Custody — Full Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ UPSTREAM SOURCES                                                 │
├─────────────────────────────────────────────────────────────────┤
│ alquran.cloud (Islamic Network)         meeAtif/hadith_datasets │
│ └─ api.alquran.cloud/v1                  └─ huggingface.co      │
│     └─ quran-uthmani + en.sahih              └─ 6 × {Book}.json │
│                                              └─ MIT license      │
│                                                                  │
│ spa5k/tafsir_api (GitHub, MIT)                                   │
│ └─ raw.githubusercontent.com/.../tafsir/                        │
│     ├─ ar-tafsir-ibn-kathir/{surah}.json  ← qul.tarteel.ai #22   │
│     └─ en-tafisr-ibn-kathir/{surah}.json  ← qul.tarteel.ai #35   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ NUR DOWNLOAD SCRIPTS  (Phase 1, 2026-06-21)                      │
├─────────────────────────────────────────────────────────────────┤
│ scripts/01_download_quran.py   →  data/quran/*.json             │
│ scripts/02_download_hadith.py  →  data/hadith/meetif/*.json     │
│ scripts/03_download_tafsir.py  →  data/tafsir/{ar,en}/*.json    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ NORMALIZATION + CHUNKING  (DEC-001)                              │
├─────────────────────────────────────────────────────────────────┤
│ scripts/04_normalize_and_chunk.py                                │
│  ├─ Adds Layer 1 structural context (surah name, narrator,      │
│ │    grade, 300-char EN snippet) to each chunk                   │
│ └─ Output: data/processed/{quran,hadith,tafsir_ar,tafsir_en}.jsonl│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ LLM CONTEXT SYNTHESIS  (DEC-005 to DEC-013)                      │
├─────────────────────────────────────────────────────────────────┤
│ scripts/kaggle_context_synthesizer.py  (ran on Lightning AI)     │
│  ├─ Qwen2.5-14B-Instruct-AWQ generates a bilingual FR/EN index   │
│ │    card (Theme, Rule, Keywords) for each chunk                 │
│ └─ Prepends the index card to the chunk text (Layer 2 context)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ EMBEDDING + INDEXING  (DEC-002, DEC-003, colab_indexer.py)       │
├─────────────────────────────────────────────────────────────────┤
│ BGE-M3 encodes each chunk:                                       │
│  ├─ Dense vector (1024 dims) → ChromaDB collections              │
│  │   ├─ quran_dense      (6,236 docs)                            │
│  │   ├─ hadith_dense     (33,738 docs)                           │
│  │   ├─ tafsir_ar_dense  (6,236 docs)                            │
│  │   └─ tafsir_en_dense  (6,236 docs)                            │
│  └─ Sparse vector (token_id → weight) → JSON files               │
│      ├─ data/sparse/quran_sparse.json                            │
│      ├─ data/sparse/hadith_sparse.json                           │
│      ├─ data/sparse/tafsir_ar_sparse.json                        │
│      └─ data/sparse/tafsir_en_sparse.json                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                FINAL DATABASE (52,446 chunks total)
                Packed as nur_indexed_data_contextual.zip
                Hosted on Lightning AI studios for re-download.
```

---

## 6. Re-download Instructions

If the local database is lost or corrupted, the full pipeline can be re-run
from scratch in this order:

```bash
# 1. Re-download raw text from the 3 upstream sources
python scripts/01_download_quran.py
python scripts/02_download_hadith.py
python scripts/03_download_tafsir.py

# 2. Normalize + chunk with Layer 1 structural context
python scripts/04_normalize_and_chunk.py

# 3. Generate Layer 2 LLM-synthesized context (needs a GPU)
#    Run scripts/kaggle_context_synthesizer.py on Kaggle or Lightning AI
#    Output: data/processed/*_contextual.jsonl

# 4. Embed + index (needs a GPU for BGE-M3)
#    Run scripts/colab_indexer.py on Google Colab (T4) or Lightning AI
#    Output: data/chroma_db/ + data/sparse/

# 5. Verify
python scripts/verify_db.py
# Expected: 4 collections with counts 6236, 33738, 6236, 6236
```

**Fast path** (skip steps 1-4): the prebuilt database is hosted on Lightning AI
at `lit://mahmouddabachi2004/inference-optimization-project/studios/inference-devbox/nur_indexed_data_contextual.zip`.
Download it with:
```bash
pip install lightning-sdk
export LIGHTNING_USER_ID=<your_user_id>
export LIGHTNING_API_KEY=<your_api_key>
lightning cp lit://mahmouddabachi2004/inference-optimization-project/studios/inference-devbox/nur_indexed_data_contextual.zip ./
unzip nur_indexed_data_contextual.zip -d .
```

---

## 7. Audit Procedure — "A user reported a problematic hadith"

If a user reports that NUR's output contains a fabricated (`Mawdu'`) hadith or
a mis-transcribed Quranic verse, follow this procedure:

### Step 1: Identify the chunk
Note the `source_id` from NUR's output (e.g. `SRC-HADITH-BUKHARI-1` or
`SRC-QURAN-2-255`). The `source_id` format is defined in `src/nur/sources.py`.

### Step 2: Locate the chunk in the database
```python
import chromadb
client = chromadb.PersistentClient(path='./data/chroma_db')
# For a Quran chunk
col = client.get_collection('quran_dense')
res = col.get(ids=['quran_2_255'], include=['metadatas', 'documents'])
# For a Hadith chunk
col = client.get_collection('hadith_dense')
res = col.get(ids=['hadith_bukhari_1'], include=['metadatas', 'documents'])
```
Inspect the `metadata` — it contains the original `text_ar`, `text_en`,
`url`, `grade`, etc.

### Step 3: Verify against the upstream source
Click the `url` field (e.g. <https://sunnah.com/bukhari:1> or
<https://quran.com/2/255>) and compare the text manually.

### Step 4: If the upstream disagrees with NUR's chunk
This means our local copy is corrupted (rare). Re-run the relevant download
script (`scripts/0X_download_*.py`) and the chunking script
(`scripts/04_normalize_and_chunk.py`), then re-embed.

### Step 5: If the upstream itself is wrong
Report the issue to the upstream source:
- alquran.cloud: <https://alquran.cloud> (no public issue tracker; contact Islamic Network)
- meeAtif/hadith_datasets: <https://huggingface.co/datasets/meeAtif/hadith_datasets/discussions>
- spa5k/tafsir_api: <https://github.com/spa5k/tafsir_api/issues>

**Do NOT silently correct the data in NUR's local copy.** Log the discrepancy
in `docs/brains.md` with a new Decision ID, and decide as a project whether to
patch locally or wait for upstream to fix.

---

## 8. License Summary

| Source | License | Commercial use OK? | Attribution required? |
|--------|---------|--------------------|-----------------------|
| alquran.cloud (Quran) | No explicit license; public service | Yes (Word of Allah is not copyrightable) | Best practice: cite alquran.cloud |
| meeAtif/hadith_datasets (Hadith) | MIT | Yes | Yes — include the MIT notice |
| spa5k/tafsir_api (Tafsir) | MIT | Yes | Yes — include the MIT notice |
| NUR's own code | (see repo LICENSE) | — | — |

NUR's output (answers to user questions) is generated by an LLM from these
sources. NUR does not claim copyright over the Quranic or Hadith text — it
belongs to Allah and the Ummah respectively. NUR's own code is licensed under
the terms in the repository's `LICENSE` file.

---

## 9. Known Issues & Caveats

1. **Tafsir folder typo**: The English Ibn Kathir tafsir is at `en-tafisr-ibn-kathir/` (typo upstream). The download script matches this exactly. If spa5k fixes it, our script breaks silently (404s). Add a regression test in Phase 8.

2. **Hadith grades vary by grader**: The meeAtif dataset includes grades from multiple graders (Darussalam, Al-Albani, by-consensus). The `grade_level` field is a normalized lowercase version (`sahih`, `hasan`, `daif`). Some hadiths are marked `Ungraded` — these get a neutral weight of 1.0 in retrieval (see `src/nur/sources.py` `grade_weight` property).

3. **No checksums yet**: The download scripts do not currently record SHA-256 checksums of the downloaded files. This is a Phase 8 evaluation task — we should add a `_checksums.json` file alongside each download so tampering can be detected on re-download.

4. **alquran.cloud has no formal license**: The Quran text itself is the Word of Allah and not subject to copyright. The Saheeh International translation is copyrighted by its publisher but alquran.cloud distributes it as a public service. NUR uses it under fair use for educational/religious purposes. If Saheeh International ever requests removal, we must switch to a public-domain English translation (e.g. Clear Quran or Muhammad Asad).

5. **Tanzil as the deeper upstream**: alquran.cloud's Uthmani text ultimately traces back to <https://tanzil.net>, the canonical open-source Quranic text project. Tanzil publishes the text under a public-domain-style license. We do not download from Tanzil directly, but if alquran.cloud ever goes offline, Tanzil is the fallback.

6. **⚠️ CRITICAL — Quran chunks use GLOBAL ayah numbering, not standard surah:ayah** (discovered 2026-06-22, DEC-030):

   The NUR Quran collection (`quran_dense`) uses **cumulative/global ayah numbering** for both the chunk ID and the `ayah_num` metadata field. This means `quran_4_596` is NOT standard 4:596 — it is standard 4:103 (the global count: 7 Al-Fatihah + 286 Al-Baqarah + 200 Al Imran + 103 An-Nisa = 596).

   This was discovered during Phase 3 recall testing. The `scripts/test_recall.py` audit showed that searching for `quran_2_43` (standard 2:43 "establish prayer") returned a chunk with text "But Satan caused them to slip out of it" (which is standard 2:36). Direct text-content matching against the alquran.cloud API confirmed the offset:

   | Standard surah:ayah | DB chunk ID | DB `ayah_num` | Offset | Standard verse content |
   |---|---|---|---|---|
   | 2:3 | `quran_2_10` | 10 | +7 | "establish prayer" (foundational) |
   | 2:43 | `quran_2_50` | 50 | +7 | "establish prayer and give zakah" |
   | 2:83 | `quran_2_90` | 90 | +7 | covenant of Children of Israel |
   | 2:110 | `quran_2_117` | 117 | +7 | "establish prayer and give zakah" |
   | 2:277 | `quran_2_284` | 284 | +7 | "establish prayer and give zakah" |
   | 4:77 | `quran_4_570` | 570 | +493 | "restrain hands and establish prayer" |
   | 4:103 | `quran_4_596` | 596 | +493 | "prayer decreed, remember Allah" |
   | 8:3 | `quran_8_1163` | 1163 | +1160 | "the ones who establish prayer" |
   | 9:5 | `quran_9_1240` | 1240 | +1235 | "sacred months have passed" |
   | 9:11 | `quran_9_1246` | 1246 | +1235 | "repent, establish prayer, give zakah" |
   | 9:18 | `quran_9_1253` | 1253 | +1235 | "maintain mosques, establish prayer" |

   The offset increases with each surah (+7 for surah 2, +493 for surah 4, +1235 for surah 9), confirming cumulative global numbering from 1 to 6,236.

   **Impact on the system**:
   - `SourceRef.source_id` generates IDs like `SRC-QURAN-2-255` using the DB's `ayah_num` — so `SRC-QURAN-2-255` actually points to standard 2:248, NOT 2:255. This is **theologically misleading** in the UI.
   - The `url` property generates `https://quran.com/2/255` using the same DB `ayah_num` — so the clickable link takes the user to the WRONG verse on quran.com.
   - The `display_label` shows "Quran 2:255" but the Arabic text shown is actually 2:248's text.

   **Root cause**: The Phase 1 ingestion script `scripts/01_download_quran.py` downloaded from alquran.cloud's `/quran/quran-uthmani` endpoint, which returns ayahs with `numberInQuran` (global) instead of `numberInSurah` (standard). The chunking script `scripts/04_normalize_and_chunk.py` then used this global number as both the chunk ID suffix and the `ayah_num` metadata field.

   **Fix (Phase 8 — re-indexing task)**:
   1. Re-download from alquran.cloud using the `/surah/{surah_number}` endpoint, which returns both `numberInSurah` (standard) and `numberInQuran` (global).
   2. Store BOTH in metadata: `ayah_num` (standard, for display + URL) and `ayah_num_global` (global, for legacy compatibility).
   3. Regenerate chunk IDs with standard numbering: `quran_2_43` (not `quran_2_50`).
   4. Re-embed all 6,236 Quran chunks with BGE-M3 (requires a Colab/Lightning run).
   5. Update `scripts/test_recall.py` RECALL_CHECKS to use standard numbering exclusively.

   **Workaround until Phase 8**: The `scripts/test_recall.py` script uses the DB `ayah_num` values directly (verified by text matching) for recall auditing. The recall check matches by metadata `surah_num` + `ayah_num` (global), so it correctly identifies whether a verse was retrieved. The `SourceRef` display issue (wrong verse number shown to users) is a known cosmetic bug — the Arabic text and English translation are still correct, only the reference number is wrong.

   **Authoritative source for verse identification**: The alquran.cloud search API (`http://api.alquran.cloud/v1/search/{keyword}/all/en.sahih`) returns the standard `numberInSurah` for each match. This is the canonical reference for "which verses are about topic X".
