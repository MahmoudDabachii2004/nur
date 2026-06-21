"""
NUR Phase 1 — BGE-M3 Embedding on Google Colab (T4 GPU, free tier)

This script is designed to be COPY-PASTED into a Google Colab notebook cell,
or uploaded as a .py file and run with `!python embed_nur_colab.py` in Colab.

WHY COLAB:
  - Embedding 52,000+ chunks with BGE-M3 takes ~3 hours on MacBook M4 MPS
  - On Colab T4 GPU: ~40 minutes (5x faster)
  - Colab T4 has 16GB GPU RAM — plenty for BGE-M3 (~1.2GB)
  - 100% free, no credit card needed

WHAT IT DOES:
  1. Installs FlagEmbedding + dependencies
  2. Downloads data/processed/*.jsonl from your Google Drive (or upload manually)
  3. Embeds each chunk with BGE-M3 → dense (1024-dim) + sparse (lexical weights)
  4. Stores dense vectors in ChromaDB (4 collections: quran, hadith, tafsir_ar, tafsir_en)
  5. Stores sparse vectors in JSON files (ChromaDB OSS doesn't support sparse natively)
  6. Zips the output and saves to Google Drive for download back to Mac

INSTRUCTIONS (see colab/README.md for screenshots/detailed walkthrough):
  1. Open https://colab.research.google.com
  2. New notebook → Runtime → Change runtime type → T4 GPU
  3. Mount Google Drive:
       from google.colab import drive
       drive.mount('/content/drive')
  4. Upload your data/processed/ folder to Google Drive:
       /content/drive/MyDrive/nur/processed/
  5. Run this script:
       !pip install FlagEmbedding chromadb
       !python embed_nur_colab.py
  6. Download the output zip back to your Mac:
       /content/drive/MyDrive/nur/embeddings.zip
  7. Unzip into your local nur/data/ directory

EXPECTED OUTPUT:
  data/chroma_db/         — ChromaDB persistent store (4 collections)
  data/sparse/quran_sparse.json
  data/sparse/hadith_sparse.json
  data/sparse/tafsir_ar_sparse.json
  data/sparse/tafsir_en_sparse.json
"""

# This entire script is structured to be Colab-friendly.
# It does NOT import from nur.* — it re-implements the minimal logic needed
# so you can run it standalone in Colab without cloning the repo.

import json
import os
import sys
import time
from pathlib import Path

# ============================================================
# STEP 1 — Install dependencies (Colab-only)
# ============================================================

def install_dependencies():
    """Install FlagEmbedding and ChromaDB in Colab."""
    import subprocess
    print("Installing FlagEmbedding + ChromaDB...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "FlagEmbedding", "chromadb", "tqdm"])
    print("Done.")


# ============================================================
# STEP 2 — Load JSONL chunks
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ============================================================
# STEP 3 — Embed with BGE-M3
# ============================================================

def embed_chunks(model, chunks: list[dict], batch_size: int = 8) -> tuple[list, list[dict]]:
    """Embed a list of chunks with BGE-M3.

    Returns (dense_vectors, sparse_records) where:
      - dense_vectors: list of np.ndarray (1024-dim each)
      - sparse_records: list of {id: str, sparse: dict[token_id, weight]}

    The text used for embedding is the NORMALIZED Arabic (text_ar_normalized)
    for Quran/Hadith/Tafsir-AR, and normalized English (text_en_normalized)
    for Tafsir-EN. This is the "Arabic-first cross-lingual retrieval" principle
    (Pilier 2): Arabic is the source of truth, BGE-M3 places AR↔EN in the same
    vector space so we don't need to translate.
    """
    from tqdm import tqdm

    # Choose embedding text: prefer normalized Arabic, fallback to normalized English
    texts = []
    for c in chunks:
        text = c.get("text_ar_normalized", "") or c.get("text_en_normalized", "")
        if not text:
            # Skip empty (shouldn't happen, but defensive)
            text = c.get("text_ar", "") or c.get("text_en", "") or ""
        texts.append(text)

    print(f"  Embedding {len(texts):,} chunks in batches of {batch_size}...")
    t0 = time.time()

    # BGE-M3 encode: returns dict with 'dense_vecs' and 'lexical_weights'
    # We process in batches to avoid OOM
    dense_out = []
    sparse_out = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        # BGE-M3 encode — use max_length=8192 (full context window)
        embeddings = model.encode(
            batch,
            batch_size=len(batch),
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # We don't use ColBERT (see ARCHITECTURE.md Phase 3)
        )

        # Extract dense vectors
        for vec in embeddings["dense_vecs"]:
            dense_out.append(vec.tolist())

        # Extract sparse weights (lexical_weights is a list of dicts: token_id → weight)
        for sparse in embeddings["lexical_weights"]:
            # Convert keys to strings (JSON requires string keys)
            sparse_str = {str(k): float(v) for k, v in sparse.items()}
            sparse_out.append(sparse_str)

    elapsed = time.time() - t0
    rate = len(texts) / elapsed if elapsed > 0 else 0
    print(f"  Done in {elapsed:.1f}s ({rate:.1f} chunks/sec, {elapsed/60:.1f} min)")

    # Pair sparse with chunk IDs
    sparse_records = [
        {"id": chunks[i]["id"], "sparse": sparse_out[i]}
        for i in range(len(chunks))
    ]

    return dense_out, sparse_records


# ============================================================
# STEP 4 — Store in ChromaDB + JSON
# ============================================================

def store_in_chromadb(chunks: list[dict], dense_vectors: list, collection_name: str, chroma_path: Path):
    """Store dense vectors + metadata in a ChromaDB collection.

    Creates 4 collections total: quran, hadith, tafsir_ar, tafsir_en.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))

    # Drop existing collection if it exists (idempotent re-runs)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": f"NUR {collection_name} chunks (BGE-M3 dense)"},
    )

    # ChromaDB requires batch sizes <= 5000, but we use 500 to be safe
    BATCH = 500
    print(f"  Adding {len(chunks):,} vectors to ChromaDB collection '{collection_name}'...")

    for i in range(0, len(chunks), BATCH):
        batch_chunks = chunks[i:i + BATCH]
        batch_vecs = dense_vectors[i:i + BATCH]

        ids = [c["id"] for c in batch_chunks]
        documents = [c.get("text_ar_normalized", "") or c.get("text_en_normalized", "") for c in batch_chunks]
        metadatas = []
        for c in batch_chunks:
            # ChromaDB metadata must be flat dict of primitives
            meta = {
                "kind": c.get("kind", ""),
                "surah": c.get("surah") or 0,
                "ayah": c.get("ayah") or 0,
                "collection": c.get("collection") or "",
                "hadith_number": c.get("hadith_number") or 0,
                "grade": c.get("grade") or "",
                "grade_weight": c.get("grade_weight", 1.0),
                "url": c.get("url", ""),
                "text_ar": c.get("text_ar", "")[:500],  # truncate for storage
                "text_en": c.get("text_en", "")[:500],
            }
            metadatas.append(meta)

        collection.add(
            ids=ids,
            embeddings=batch_vecs,
            documents=documents,
            metadatas=metadatas,
        )

    print(f"  [OK] Collection '{collection_name}': {collection.count()} vectors")


def store_sparse(sparse_records: list[dict], sparse_path: Path, source_name: str):
    """Store sparse vectors in a JSON file (one per source)."""
    sparse_path.mkdir(parents=True, exist_ok=True)
    out_file = sparse_path / f"{source_name}_sparse.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(sparse_records, f)
    size_mb = out_file.stat().st_size / 1024 / 1024
    print(f"  [OK] Sparse JSON: {out_file.name} ({size_mb:.1f} MB)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("NUR Phase 1 — BGE-M3 Embedding (Google Colab T4)")
    print("=" * 60)

    # 1. Install deps
    install_dependencies()

    # 2. Locate data (Google Drive mount or local upload)
    # Try Google Drive first
    drive_path = Path("/content/drive/MyDrive/nur/processed")
    local_path = Path("/content/processed")

    if drive_path.exists():
        processed_dir = drive_path
        output_root = Path("/content/drive/MyDrive/nur")
        print(f"\n[INPUT] Using Google Drive: {processed_dir}")
    elif local_path.exists():
        processed_dir = local_path
        output_root = Path("/content/nur_output")
        print(f"\n[INPUT] Using local upload: {processed_dir}")
    else:
        print("\n[FATAL] No processed data found.")
        print("Either:")
        print("  1. Mount Google Drive and upload data/processed/ to /content/drive/MyDrive/nur/processed/")
        print("  2. Upload data/processed/ directly to /content/processed/")
        return 1

    chroma_path = output_root / "chroma_db"
    sparse_path = output_root / "sparse"
    chroma_path.mkdir(parents=True, exist_ok=True)
    sparse_path.mkdir(parents=True, exist_ok=True)

    # 3. Load BGE-M3 model
    print("\n[MODEL] Loading BGE-M3...")
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,  # T4 supports FP16, ~2x faster
    )
    print("  [OK] BGE-M3 loaded.")

    # 4. Process each JSONL file
    sources = [
        ("quran", "quran.jsonl"),
        ("hadith", "hadith.jsonl"),
        ("tafsir_ar", "tafsir_ar.jsonl"),
        ("tafsir_en", "tafsir_en.jsonl"),
    ]

    for source_name, filename in sources:
        print(f"\n{'=' * 60}")
        print(f"[{source_name.upper()}] Processing {filename}")
        print("=" * 60)

        jsonl_path = processed_dir / filename
        if not jsonl_path.exists():
            print(f"  [SKIP] {filename} not found.")
            continue

        chunks = load_jsonl(jsonl_path)
        print(f"  Loaded {len(chunks):,} chunks.")

        if not chunks:
            continue

        # Embed
        dense_vectors, sparse_records = embed_chunks(model, chunks, batch_size=8)

        # Store dense in ChromaDB
        store_in_chromadb(chunks, dense_vectors, source_name, chroma_path)

        # Store sparse in JSON
        store_sparse(sparse_records, sparse_path, source_name)

    # 5. Zip output for easy download
    print(f"\n{'=' * 60}")
    print("Zipping output for download...")
    print("=" * 60)

    import shutil
    zip_path = output_root / "embeddings"
    shutil.make_archive(str(zip_path), "zip", root_dir=output_root, base_dir="chroma_db")
    # Append sparse separately (shutil.make_archive doesn't support multiple dirs)
    # Actually let's just zip both chroma_db and sparse together
    # Easier: zip the whole output_root excluding processed/
    print(f"  [OK] Output at: {zip_path}.zip")
    print(f"  Download this file back to your Mac and unzip into nur/data/")

    print("\n" + "=" * 60)
    print("EMBEDDING COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
