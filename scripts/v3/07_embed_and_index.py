"""
NUR V3 — Step 7: Embed chunks with BGE-M3 + index in ChromaDB

⚠️ RUN ON LIGHTNING AI L40S INSTANCE — NOT local.
   (Same setup as scripts/kaggle_context_synthesizer.py from Phase 1.)

Reads:
  data/processed/quran_v3.jsonl   (6,236 chunks)
  data/processed/hadith_v3.jsonl  (33,738 chunks)

Writes:
  data/chroma_db_v3/quran_v3/     (dense vectors + metadata)
  data/chroma_db_v3/hadith_v3/    (dense vectors + metadata)
  data/sparse_v3/quran_v3.json    (sparse lexical weights)
  data/sparse_v3/hadith_v3.json   (sparse lexical weights)
  nur_v3_indexed.zip              (final archive for download)

Model: BAAI/bge-m3 (1024 dim, multilingual AR/EN/FR)
Encoding: dense + sparse (lexical weights) in one pass via FlagEmbedding

Usage on Lightning AI:
  pip install FlagEmbedding chromadb
  python scripts/v3/07_embed_and_index.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db_v3"
SPARSE_DIR = DATA_DIR / "sparse_v3"

MODEL_ID = "BAAI/bge-m3"
BATCH_SIZE = 128
EMBEDDING_DIM = 1024


def load_jsonl(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def extract_metadata(chunk: dict) -> dict:
    """Extract ChromaDB-compatible metadata (flat dict of primitives).
    ChromaDB doesn't support nested dicts/lists in metadata."""
    meta = {}
    for k, v in chunk.items():
        if k in ("id", "embedding_text"):
            continue
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
        elif isinstance(v, dict):
            # Flatten one level
            for dk, dv in v.items():
                if isinstance(dv, (str, int, float, bool)):
                    meta[f"{k}_{dk}"] = dv
                elif isinstance(dv, list):
                    meta[f"{k}_{dk}"] = "|".join(str(x) for x in dv)[:1000]  # truncate
        elif isinstance(v, list):
            # Store as pipe-separated string (truncated)
            meta[k] = "|".join(str(x) for x in v)[:1000]
    return meta


def embed_and_store(collection_name: str, chunks: list[dict], model) -> None:
    """Generate dense + sparse embeddings, store in ChromaDB + JSON."""
    from tqdm import tqdm
    import chromadb

    print(f"\n[{collection_name}] Embedding {len(chunks):,} chunks...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if any
    try:
        client.delete_collection(f"{collection_name}_dense")
    except Exception:
        pass

    col = client.create_collection(
        name=f"{collection_name}_dense",
        metadata={"description": f"NUR V3 {collection_name} dense (semantic)"},
    )

    ids = [c["id"] for c in chunks]
    documents = [c.get("embedding_text", "") for c in chunks]
    metadatas = [extract_metadata(c) for c in chunks]

    sparse_data: dict[str, dict] = {}
    total = len(documents)

    for i in tqdm(range(0, total, BATCH_SIZE), desc=f"Embedding {collection_name}"):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_metas = metadatas[i : i + BATCH_SIZE]

        output = model.encode(
            batch_docs,
            batch_size=BATCH_SIZE,
            max_length=8192,  # FIX: default is 512 which truncates our chunks (DEC-042)
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = output["dense_vecs"]
        if hasattr(dense_vecs, "cpu"):
            dense_vecs = dense_vecs.cpu().numpy()

        # L2 normalize
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense_vecs = dense_vecs / norms

        col.add(
            ids=batch_ids,
            embeddings=dense_vecs.tolist(),
            documents=batch_docs,
            metadatas=batch_metas,
        )

        lexical_weights = output["lexical_weights"]
        for j, lw in enumerate(lexical_weights):
            if lw and len(lw) > 0:
                sparse_data[batch_ids[j]] = {
                    "indices": [int(k) for k in lw.keys()],
                    "values": [float(v) for v in lw.values()],
                }

    # Save sparse index
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    sparse_path = SPARSE_DIR / f"{collection_name}_sparse.json"
    with sparse_path.open("w", encoding="utf-8") as f:
        json.dump(sparse_data, f)
    print(f"  ChromaDB: {col.count():,} docs | Sparse: {len(sparse_data):,} entries")
    print(f"  Sparse saved: {sparse_path.relative_to(PROJECT_ROOT)}")


def zip_final_database() -> str:
    """Zip the chroma_db_v3 + sparse_v3 directories for download."""
    print("\n--- ZIPPING FINAL V3 DATABASE ---")
    zip_path = PROJECT_ROOT / "nur_v3_indexed.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _dirs, files in os.walk(CHROMA_DIR):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(PROJECT_ROOT)
                zipf.write(file_path, rel_path)
        for root, _dirs, files in os.walk(SPARSE_DIR):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(PROJECT_ROOT)
                zipf.write(file_path, rel_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  Zipped: {zip_path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")
    return str(zip_path)


def main() -> int:
    print("=" * 60)
    print("NUR V3 — Step 7: Embed + Index (Lightning AI L40S)")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Output: {CHROMA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Batch size: {BATCH_SIZE}\n")

    if not torch.cuda.is_available():
        print("[FATAL] No GPU detected. Run on Lightning AI L40S.")
        return 1

    # Load chunks
    quran_path = PROCESSED_DIR / "quran_v3.jsonl"
    hadith_path = PROCESSED_DIR / "hadith_v3.jsonl"

    if not quran_path.exists():
        print(f"[FATAL] {quran_path} not found — run 05_build_chunks.py first")
        return 1
    if not hadith_path.exists():
        print(f"[FATAL] {hadith_path} not found — run 05_build_chunks.py first")
        return 1

    quran_chunks = load_jsonl(quran_path)
    hadith_chunks = load_jsonl(hadith_path)
    print(f"Loaded {len(quran_chunks):,} Quran chunks + {len(hadith_chunks):,} Hadith chunks")

    # Setup dirs
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)

    # Load BGE-M3
    print("\nLoading BGE-M3 model on GPU...")
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(MODEL_ID, use_fp16=True, devices="cuda:0")
    print("  BGE-M3 loaded successfully")

    # Embed both collections
    embed_and_store("quran_v3", quran_chunks, model)
    embed_and_store("hadith_v3", hadith_chunks, model)

    # Cleanup model
    print("\nCleaning up BGE-M3 from VRAM...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    # Zip final database
    zip_path = zip_final_database()

    print("\n" + "=" * 60)
    print("V3 EMBED + INDEX COMPLETE")
    print("=" * 60)
    print(f"  Quran:  {len(quran_chunks):,} chunks embedded")
    print(f"  Hadith: {len(hadith_chunks):,} chunks embedded")
    print(f"  Archive: {zip_path}")
    print(f"\nNext step: download {zip_path} and run scripts/v3/08_verify_pipeline.py locally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
