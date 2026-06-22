# NUR V3 — Lightning AI Runner

> Run this script on a Lightning AI L40S studio to execute the GPU steps
> (4: context cards, 7: embed + index) of the V3 pipeline.

## Prerequisites

1. **Lightning AI studio** with L40S GPU (free tier gives ~3-4h)
2. **Repo cloned** to the studio home:
   ```bash
   git clone https://github.com/MahmoudDabachii2004/nur.git
   cd nur
   ```
3. **Data downloaded** (steps 1-3 run locally OR on Lightning AI):
   - Option A: Upload `data/` directory from local to Lightning AI
   - Option B: Run `python scripts/v3/01_download_quran.py && python scripts/v3/02_download_hadith.py && python/scripts/v3/03_download_tafsirs.py` on Lightning AI

## Quickstart (one-liner)

```bash
pip install vllm FlagEmbedding chromadb pyarrow && \
python scripts/v3/04_generate_context_cards.py && \
python scripts/v3/05_build_chunks.py && \
python scripts/v3/06_compute_cross_refs.py && \
python scripts/v3/07_embed_and_index.py
```

## Expected output

```
data/processed/context_cards.jsonl   (6,236 lines)
data/processed/quran_v3.jsonl        (6,236 chunks, 3-layer)
data/processed/hadith_v3.jsonl       (33,738 chunks, 2-layer)
data/chroma_db_v3/quran_v3_dense/    (~150 MB)
data/chroma_db_v3/hadith_v3_dense/   (~400 MB)
data/sparse_v3/quran_v3_sparse.json
data/sparse_v3/hadith_v3_sparse.json
nur_v3_indexed.zip                   (~600 MB, download this back to local)
```

## Time budget on L40S

| Step | Time | Memory |
|------|------|--------|
| 4: Context Cards (Qwen2.5-14B-AWQ, 6,236 prompts) | ~30 min | 14 GB VRAM |
| 5: Build chunks | ~1 min | <1 GB |
| 6: Cross-refs | ~30 sec | <1 GB |
| 7: Embed + Index (BGE-M3, 39,974 chunks) | ~25 min | 4 GB VRAM |
| **Total** | **~1 hour** | **L40S 24GB** |

Free tier L40S gives ~3-4h — we have comfortable margin.

## Download the result back to local

```bash
# On Lightning AI studio:
lightning cp nur_v3_indexed.zip lit://<your-studio>/nur_v3_indexed.zip

# On local:
lightning cp lit://<your-studio>/nur_v3_indexed.zip ./
unzip nur_v3_indexed.zip -d repos/nur/
```

## After download

Run locally:
```bash
python scripts/v3/08_verify_pipeline.py
```

This runs the 5 ground-truth examples from `docs/v3/08_EXAMPLES.md`.
