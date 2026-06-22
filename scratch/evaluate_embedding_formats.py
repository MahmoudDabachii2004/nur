import os
import json
import numpy as np
import chromadb
from FlagEmbedding import BGEM3FlagModel

DATA_DIR = "/Users/mahmoud/Documents/nur/nur/data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def normalize_arabic(text: str, for_retrieval: bool = True) -> str:
    import re
    if not text:
        return ""
    # Strip diacritics
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]', '', text)
    # Remove tatweel
    text = re.sub(r'\u0640', '', text)
    # Normalize alef variants
    text = text.replace('\u0623', '\u0627')
    text = text.replace('\u0625', '\u0627')
    text = text.replace('\u0622', '\u0627')
    if for_retrieval:
        text = text.replace('\u0649', '\u064A')
        text = text.replace('\u0629', '\u0647')
    return re.sub(r'\s+', ' ', text).strip()

def main():
    print("Loading BGEM3 model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    
    # Load hadith chunks
    chunks_path = os.path.join(PROCESSED_DIR, "hadith_chunks.jsonl")
    all_chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_chunks.append(json.loads(line))
                
    print(f"Loaded {len(all_chunks)} hadiths.")
    
    # Find some hadiths that contain congregational prayer keywords to make sure they are in our test set
    # Keywords: "جماعة" (jama'ah) or "congregation"
    target_chunks = []
    other_chunks = []
    for c in all_chunks:
        text_ar = c.get("text_ar", "")
        text_en = c.get("text_en", "")
        if "جماعة" in text_ar or "congregat" in text_en.lower():
            target_chunks.append(c)
        else:
            other_chunks.append(c)
            
    print(f"Found {len(target_chunks)} hadiths related to congregational prayer/jama'ah.")
    
    # Let's build a test dataset of 200 items (all targets + some others to fill)
    test_chunks = target_chunks[:100] + other_chunks[:200]
    print(f"Using {len(test_chunks)} chunks for evaluation database.")
    
    # We will create two in-memory collections
    client = chromadb.Client()
    
    # Format A: Arabic Only
    col_ar = client.create_collection("format_ar_only")
    # Format B: Bilingual
    col_bi = client.create_collection("format_bilingual")
    
    ids = []
    docs_ar = []
    docs_bi = []
    metas = []
    
    for c in test_chunks:
        ids.append(c["id"])
        
        # Build Format A (AR only)
        chapter_title_ar = c.get("chapter_title_ar", "")
        text_ar_norm = c.get("text_ar_normalized", "")
        doc_ar = ""
        if chapter_title_ar:
            doc_ar += f"باب: {normalize_arabic(chapter_title_ar)} | "
        doc_ar += f"متن: {text_ar_norm}"
        docs_ar.append(doc_ar)
        
        # Build Format B (Bilingual)
        chapter_title_en = c.get("chapter_title_en", "")
        text_en = c.get("text_en", "")
        doc_bi = doc_ar
        if chapter_title_en:
            doc_bi += f" | Title: {chapter_title_en}"
        if text_en:
            doc_bi += f" | Hadith: {text_en}"
        docs_bi.append(doc_bi)
        
        meta = {"id": c["id"], "collection": c["collection"], "grade": c["grade"]}
        metas.append(meta)
        
    print("\nEmbedding Format A (AR-only)...")
    embs_ar = model.encode(docs_ar, return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
    if hasattr(embs_ar, 'cpu'):
        embs_ar = embs_ar.cpu().numpy()
    # Normalize
    norms_ar = np.linalg.norm(embs_ar, axis=1, keepdims=True)
    norms_ar[norms_ar == 0] = 1
    embs_ar = embs_ar / norms_ar
    col_ar.add(ids=ids, embeddings=embs_ar.tolist(), documents=docs_ar, metadatas=metas)
    
    print("Embedding Format B (Bilingual)...")
    embs_bi = model.encode(docs_bi, return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
    if hasattr(embs_bi, 'cpu'):
        embs_bi = embs_bi.cpu().numpy()
    # Normalize
    norms_bi = np.linalg.norm(embs_bi, axis=1, keepdims=True)
    norms_bi[norms_bi == 0] = 1
    embs_bi = embs_bi / norms_bi
    col_bi.add(ids=ids, embeddings=embs_bi.tolist(), documents=docs_bi, metadatas=metas)
    
    # Test queries
    queries = [
        {"lang": "AR (normalized)", "q": normalize_arabic("ما حكم صلاة الجماعة", for_retrieval=True)},
        {"lang": "EN", "q": "What is the ruling on congregational prayer?"},
        {"lang": "FR", "q": "Quel est le statut de la prière en commun?"}
    ]
    
    for query_info in queries:
        lang = query_info["lang"]
        q = query_info["q"]
        print(f"\n==================================================")
        print(f"QUERY: '{q}' ({lang})")
        print(f"==================================================")
        
        # Encode query
        q_emb = model.encode([q], return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs'][0]
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm
            
        # Search Format A
        print(f"\n--- Format A: Arabic Only ---")
        res_ar = col_ar.query(query_embeddings=[q_emb.tolist()], n_results=3)
        for i, (doc, meta, dist) in enumerate(zip(res_ar["documents"][0], res_ar["metadatas"][0], res_ar["distances"][0])):
            print(f"  #{i+1}: {meta.get('id')} | dist={dist:.4f} | grade={meta.get('grade')}")
            print(f"      Doc preview: {doc[:180]}")
            
        # Search Format B
        print(f"\n--- Format B: Bilingual (AR + EN) ---")
        res_bi = col_bi.query(query_embeddings=[q_emb.tolist()], n_results=3)
        for i, (doc, meta, dist) in enumerate(zip(res_bi["documents"][0], res_bi["metadatas"][0], res_bi["distances"][0])):
            print(f"  #{i+1}: {meta.get('id')} | dist={dist:.4f} | grade={meta.get('grade')}")
            print(f"      Doc preview: {doc[:180]}")

if __name__ == "__main__":
    main()
