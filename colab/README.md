# NUR Phase 1 — Colab Embedding Instructions

This guide walks you through running the BGE-M3 embedding on Google Colab's free T4 GPU.
Total time: ~45 minutes (5 min setup + 40 min embedding + 5 min download).

## Why Colab?

| Hardware | Time for 52K chunks | Cost |
|----------|---------------------|------|
| MacBook M4 MPS (CPU fallback) | ~3 hours | Free |
| Google Colab T4 GPU | ~40 min | Free |
| Google Colab A100 GPU | ~12 min | $10/mo (Colab Pro) |

Colab T4 is the sweet spot — free, fast enough, no setup beyond a browser.

## Step-by-Step

### 1. Prepare your data

On your Mac, after running `scripts/01-04_*.py`:

```bash
# Verify you have all 4 JSONL files
ls -lh nur/data/processed/
# Should see:
#   quran.jsonl       (~3 MB)
#   hadith.jsonl      (~50 MB)
#   tafsir_ar.jsonl   (~10 MB)
#   tafsir_en.jsonl   (~10 MB)
```

### 2. Upload to Google Drive

1. Open [Google Drive](https://drive.google.com)
2. Create a folder: `My Drive → nur → processed`
3. Upload all 4 JSONL files into `nur/processed/`

(Alternative: upload directly in Colab — but Google Drive is more reliable for large files.)

### 3. Open Colab

1. Go to https://colab.research.google.com
2. Click **New notebook**
3. Runtime → Change runtime type → **T4 GPU** → Save

### 4. Run the embedding

In the first cell:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Verify the data is there
!ls /content/drive/MyDrive/nur/processed/
# Should list: quran.jsonl  hadith.jsonl  tafsir_ar.jsonl  tafsir_en.jsonl
```

In the second cell:

```python
# Clone the repo to get the embedding script
!git clone https://github.com/MahmoudDabachii2004/nur.git /content/nur
!cp /content/nur/colab/embed_nur_colab.py /content/

# Run it
!python /content/embed_nur_colab.py
```

The script will:
1. Install FlagEmbedding + ChromaDB (~3 min)
2. Load BGE-M3 model (~1 min)
3. Embed each chunk (~40 min total)
4. Store dense vectors in ChromaDB at `/content/drive/MyDrive/nur/chroma_db/`
5. Store sparse vectors as JSON at `/content/drive/MyDrive/nur/sparse/`

### 5. Download back to Mac

The output is already in your Google Drive at:
- `My Drive/nur/chroma_db/` (ChromaDB persistent store)
- `My Drive/nur/sparse/` (4 JSON files with sparse vectors)

On your Mac:

```bash
# Install Google Drive desktop app if not already
# Then sync, or download manually via web browser

# Once downloaded, copy to your local nur/data/ directory
cp -r ~/Google\ Drive/nur/chroma_db nur/data/
cp -r ~/Google\ Drive/nur/sparse nur/data/

# Verify
ls -lh nur/data/chroma_db/
ls -lh nur/data/sparse/
```

### 6. Verify

```bash
cd nur
python scripts/05_verify_phase1.py
```

You should see chunk counts near:
- Quran: 6,236
- Hadith: ~33,738
- Tafsir AR: ~6,236
- Tafsir EN: ~6,236
- **Total: ~52,846 chunks**

## Troubleshooting

### OOM (Out Of Memory) on T4

If you get CUDA OOM, reduce the batch size:

```python
# In embed_nur_colab.py, line ~80
dense_vectors, sparse_records = embed_chunks(model, chunks, batch_size=4)  # was 8
```

### Colab session times out

Colab free tier disconnects after ~90 min of inactivity. The full run takes ~45 min,
so you should be fine — but **don't switch tabs** during the run.

If it does disconnect, re-run from the top. The script is **idempotent**:
- It deletes existing ChromaDB collections before re-creating them
- It overwrites JSON sparse files

### BGE-M3 download fails

The model is ~2.3GB. If HuggingFace is slow from your region:

```python
# Use a HuggingFace mirror
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Then re-run
```

### ChromaDB errors on Mac after download

Make sure you're using the same ChromaDB version locally as on Colab:

```bash
pip install chromadb==0.5.5
```

## What's Next?

Once Phase 1 is verified, you're ready for Phase 2 — building the RAG pipeline.

See `docs/PHASES.md` for the full roadmap.
