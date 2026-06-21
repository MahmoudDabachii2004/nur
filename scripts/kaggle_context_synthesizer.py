"""
@file kaggle_context_synthesizer.py
@description This script processes Islamic religious texts (Quran, Hadith, Tafsir) to create a searchable,
             bilingual (English and French) database. It distributes Qwen2.5-14B-Instruct-AWQ via vLLM
             across multiple GPUs natively (Tensor Parallelism) to accelerate summary synthesis.
             It then generates dense and sparse embeddings using BGE-M3, and packages the final database for download.
"""

import json
import os
import sys
import time
import gc
import zipfile
import subprocess
import numpy as np
import torch
from vllm import LLM, SamplingParams

# Global directories
DATA_DIR = "./data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
SPARSE_DIR = os.path.join(DATA_DIR, "sparse")


def load_jsonl(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_bilingual_prompt(chunk):
    source = chunk.get("source", "unknown")
    text_to_analyze = ""
    
    if source == "quran":
        surah = chunk.get("surah_name_en", "")
        ayah = chunk.get("ayah_num", "")
        ar_text = chunk.get("text_ar", "")
        en_text = chunk.get("text_en", "")
        fr_text = chunk.get("text_fr", "")
        text_to_analyze = f"Coran - Sourate {surah}, Ayah {ayah}.\nArabe: {ar_text}\nFrançais: {fr_text}\nAnglais: {en_text}"
        
    elif source == "hadith":
        collection = chunk.get("collection", "")
        chapter_en = chunk.get("chapter_title_en", "")
        text_ar = chunk.get("text_ar", "")
        text_en = chunk.get("text_en", "")
        grade = chunk.get("grade", "")
        text_to_analyze = f"Hadith - Collection: {collection}. Chapitre: {chapter_en}. Grade: {grade}.\nArabe: {text_ar}\nAnglais: {text_en}"
        
    elif source == "tafsir":
        surah = chunk.get("surah_name_en", "")
        ayah = chunk.get("ayah_num", "")
        tafsir_text = chunk.get("tafsir_text", "")
        text_to_analyze = f"Tafsir - Sourate {surah}, Ayah {ayah}.\nTexte: {tafsir_text[:500]}"
        
    else:
        text_to_analyze = str(chunk)

    system_prompt = (
        "You are an expert Islamic theologian and database indexer. "
        "Your task is to generate a concise, precise bilingual (French and English) contextual index card "
        "explaining the theological and jurisprudence (Fiqh) meaning of the text.\n"
        "The response MUST contain the core categories, the main rules, and key search synonyms (e.g. wudu/ablution, usury/interest/riba).\n"
        "Format exactly like this:\n"
        "[FR] Thème: <catégories>. Règle: <règles/résumé>. Mots-clés: <synonymes>.\n"
        "[EN] Topic: <categories>. Rule: <rules/summary>. Keywords: <synonyms>.\n"
        "Keep the response concise (under 80 words). For short texts, be brief. For long texts, make sure all major rulings are captured. Do not write any explanations or greetings."
    )

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nTexte à indexer :\n{text_to_analyze}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def extract_metadata(chunk):
    meta = {}
    for k, v in chunk.items():
        if k in ("id", "embedding_text", "text_ar_normalized", "tafsir_text_ar_normalized"):
            continue
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
        elif isinstance(v, dict):
            for dk, dv in v.items():
                if isinstance(dv, (str, int, float, bool)):
                    meta[f"{k}_{dk}"] = dv
    return meta


def embed_and_store_unified(collection_name, chunks, model, batch_size=128):
    from tqdm import tqdm
    import chromadb
    
    print(f"\n[{collection_name}] Generating embeddings for {len(chunks)} chunks...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        client.delete_collection(f"{collection_name}_dense")
    except Exception:
        pass
            
    col = client.create_collection(
        name=f"{collection_name}_dense",
        metadata={"description": f"NUR {collection_name} dense (semantic)"}
    )
    
    ids = [c["id"] for c in chunks]
    documents = [c.get("embedding_text", "") for c in chunks]
    metadatas = [extract_metadata(c) for c in chunks]
    
    total = len(documents)
    sparse_data = {}
    
    for i in tqdm(range(0, total, batch_size), desc=f"Embedding {collection_name}"):
        batch_docs = documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        output = model.encode(
            batch_docs,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        
        dense_vecs = output['dense_vecs']
        if hasattr(dense_vecs, 'cpu'):
            dense_vecs = dense_vecs.cpu().numpy()
            
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense_vecs = dense_vecs / norms
        
        col.add(
            ids=batch_ids,
            embeddings=dense_vecs.tolist(),
            documents=batch_docs,
            metadatas=batch_metas
        )
        
        lexical_weights = output['lexical_weights']
        for j, lw in enumerate(lexical_weights):
            if lw and len(lw) > 0:
                sparse_data[batch_ids[j]] = {
                    "indices": [int(k) for k in lw.keys()],
                    "values": [float(v) for v in lw.values()]
                }
                
    sparse_path = os.path.join(SPARSE_DIR, f"{collection_name}_sparse.json")
    with open(sparse_path, "w", encoding="utf-8") as f:
        json.dump(sparse_data, f)
        
    print(f"Finished {collection_name}. ChromaDB: {col.count()} docs | Sparse: {len(sparse_data)} entries")


def zip_final_database(zip_filename="nur_indexed_data_contextual.zip"):
    print("\n--- ZIPPING FINAL DATABASE ---")
    zip_path = os.path.join("./", zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files_list in os.walk(CHROMA_DIR):
            for file in files_list:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, "./")
                zipf.write(file_path, rel_path)
                
        for root, dirs, files_list in os.walk(SPARSE_DIR):
            for file in files_list:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, "./")
                zipf.write(file_path, rel_path)
                
    print(f"Zipped final database to {zip_path} (Size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB)")
    return zip_path


def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(SPARSE_DIR, exist_ok=True)

    quran_path = os.path.join(PROCESSED_DIR, "quran_chunks.jsonl")
    hadith_path = os.path.join(PROCESSED_DIR, "hadith_chunks.jsonl")
    tafsir_path = os.path.join(PROCESSED_DIR, "tafsir_chunks.jsonl")
    
    if not (os.path.exists(quran_path) and os.path.exists(hadith_path) and os.path.exists(tafsir_path)):
        print("\n❌ Error: Missing files in data/processed/!")
        sys.exit(1)

    # 1. Ingest chunks
    print("\n--- Ingesting raw chunks ---")
    quran_chunks = load_jsonl(quran_path)
    hadith_chunks = load_jsonl(hadith_path)
    tafsir_all_chunks = load_jsonl(tafsir_path)
    
    all_chunks = quran_chunks + hadith_chunks + tafsir_all_chunks
    print(f"Total chunks loaded: {len(all_chunks)} (Quran: {len(quran_chunks)}, Hadith: {len(hadith_chunks)}, Tafsir: {len(tafsir_all_chunks)})")

    # 2. Determine GPU Count and configure environment variables safely
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 2:
        num_gpus = 2
        
    if num_gpus == 0:
        print("❌ Error: No GPU detected!")
        sys.exit(1)
        
    print(f"Using {num_gpus} GPU(s) natively via vLLM Tensor Parallelism.")
    
    # Environment cleanups for JIT compilation safety
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]  # Let vLLM see all available T4 GPUs

    # 3. Native vLLM Context Generation (Stage 1)
    print("\n[STAGE 1] Initializing vLLM Engine across available GPUs...")
    llm = LLM(
        model="Qwen/Qwen2.5-14B-Instruct-AWQ",
        tensor_parallel_size=num_gpus,  # <--- Orchestration native sur 2 GPUs
        max_model_len=4096,
        gpu_memory_utilization=0.90
    )
    
    try:
        tokenizer = llm.get_tokenizer()
        if hasattr(tokenizer, "model_max_length"):
            tokenizer.model_max_length = 4096
    except Exception as e:
        print(f"Warning: Could not override tokenizer length: {e}")
        
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=120,
        stop=["<|im_end|>", "<|im_start|>"]
    )
    
    prompts = [build_bilingual_prompt(c) for c in all_chunks]
    print(f"Generating contexts for {len(prompts)} chunks...")
    
    outputs = llm.generate(prompts, sampling_params)
    
    # Reassemble objects directly in memory
    processed_chunks = []
    for idx, out in enumerate(outputs):
        generated_text = out.outputs[0].text.strip()
        c = all_chunks[idx]
        c["bilingual_context"] = generated_text
        original_embedding = c.get("embedding_text", "")
        c["embedding_text"] = f"Context: {generated_text} | {original_embedding}"
        processed_chunks.append(c)

    # 4. Total cleanup of vLLM from VRAM before changing stage
    print("\nDestroying vLLM engine to free up VRAM...")
    from vllm.distributed.parallel_state import destroy_model_parallel
    try:
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)  # Let the kernel settle down

    # 5. Split chunks back into source groups
    quran_processed = [c for c in processed_chunks if c.get("source") == "quran"]
    hadith_processed = [c for c in processed_chunks if c.get("source") == "hadith"]
    tafsir_processed = [c for c in processed_chunks if c.get("source") == "tafsir"]
    
    tafsir_ar = [c for c in tafsir_processed if c.get("language") == "ar"]
    tafsir_en = [c for c in tafsir_processed if c.get("language") == "en"]

    # ==========================================
    # STAGE 2: EMBEDDING INGESTION (BGE-M3)
    # ==========================================
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Target single device for Embedding stage
    print("\n[STAGE 2] Loading BGEM3FlagModel on GPU 0...")
    from FlagEmbedding import BGEM3FlagModel
    
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="cuda:0")
    print("BGE-M3 loaded successfully on CUDA!")
    
    embed_and_store_unified("quran", quran_processed, model, batch_size=128)
    embed_and_store_unified("hadith", hadith_processed, model, batch_size=128)
    embed_and_store_unified("tafsir_ar", tafsir_ar, model, batch_size=128)
    embed_and_store_unified("tafsir_en", tafsir_en, model, batch_size=128)

    # ==========================================
    # STAGE 3: ZIP DATABASE FOR DOWNLOAD
    # ==========================================
    zip_path = zip_final_database()
    print(f"\nAll operations completed! Database archive ready at: {os.path.abspath(zip_path)}")


if __name__ == "__main__":
    main()