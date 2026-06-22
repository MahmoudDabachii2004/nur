import os
import json
import numpy as np
import chromadb
from FlagEmbedding import BGEM3FlagModel

DATA_DIR = "/Users/mahmoud/Documents/nur/nur/data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
SPARSE_DIR = os.path.join(DATA_DIR, "sparse")

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
    
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    print("\nCollections in ChromaDB:")
    for col in client.list_collections():
        print(f" - {col.name}: {col.count()} docs")
        
    query = "ما حكم صلاة الجماعة"
    print(f"\nOriginal query: '{query}'")
    
    # 1. Encode query
    q_output = model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    q_dense = q_output['dense_vecs'][0]
    q_sparse = q_output['lexical_weights'][0]
    
    # Normalize dense
    norm = np.linalg.norm(q_dense)
    if norm > 0:
        q_dense = q_dense / norm
        
    # Normalized query encode
    query_norm = normalize_arabic(query, for_retrieval=True)
    print(f"Normalized query: '{query_norm}'")
    q_output_norm = model.encode([query_norm], return_dense=True, return_sparse=True, return_colbert_vecs=False)
    q_dense_norm = q_output_norm['dense_vecs'][0]
    q_sparse_norm = q_output_norm['lexical_weights'][0]
    norm_norm = np.linalg.norm(q_dense_norm)
    if norm_norm > 0:
        q_dense_norm = q_dense_norm / norm_norm

    # Query Hadith Dense (original vs normalized)
    print("\n--- Hadith Dense Retrieval (Original Query) ---")
    hadith_col = client.get_collection("hadith_dense")
    res_orig = hadith_col.query(query_embeddings=[q_dense.tolist()], n_results=5)
    for i, (doc, meta, dist) in enumerate(zip(res_orig["documents"][0], res_orig["metadatas"][0], res_orig["distances"][0])):
        print(f"  #{i+1}: {meta.get('id')} | dist={dist:.4f} | collection={meta.get('collection')} | grade={meta.get('grade')}")
        print(f"      Doc preview: {doc[:150]}")
        
    print("\n--- Hadith Dense Retrieval (Normalized Query) ---")
    res_norm = hadith_col.query(query_embeddings=[q_dense_norm.tolist()], n_results=5)
    for i, (doc, meta, dist) in enumerate(zip(res_norm["documents"][0], res_norm["metadatas"][0], res_norm["distances"][0])):
        print(f"  #{i+1}: {meta.get('id')} | dist={dist:.4f} | collection={meta.get('collection')} | grade={meta.get('grade')}")
        print(f"      Doc preview: {doc[:150]}")

    # Query in French
    q_fr = "Quel est le statut de la prière en commun?"
    print(f"\nFrench query: '{q_fr}'")
    q_output_fr = model.encode([q_fr], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    q_dense_fr = q_output_fr['dense_vecs'][0]
    norm_fr = np.linalg.norm(q_dense_fr)
    if norm_fr > 0:
        q_dense_fr = q_dense_fr / norm_fr
    res_fr = hadith_col.query(query_embeddings=[q_dense_fr.tolist()], n_results=5)
    for i, (doc, meta, dist) in enumerate(zip(res_fr["documents"][0], res_fr["metadatas"][0], res_fr["distances"][0])):
        print(f"  #{i+1}: {meta.get('id')} | dist={dist:.4f} | collection={meta.get('collection')} | grade={meta.get('grade')}")
        print(f"      Doc preview: {doc[:150]}")

    # Query in English
    q_en = "What is the ruling on congregational prayer?"
    print(f"\nEnglish query: '{q_en}'")
    q_output_en = model.encode([q_en], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    q_dense_en = q_output_en['dense_vecs'][0]
    norm_en = np.linalg.norm(q_dense_en)
    if norm_en > 0:
        q_dense_en = q_dense_en / norm_en
    res_en = hadith_col.query(query_embeddings=[q_dense_en.tolist()], n_results=5)
    for i, (doc, meta, dist) in enumerate(zip(res_en["documents"][0], res_en["metadatas"][0], res_en["distances"][0])):
        print(f"  #{i+1}: {meta.get('id')} | dist={dist:.4f} | collection={meta.get('collection')} | grade={meta.get('grade')}")
        print(f"      Doc preview: {doc[:150]}")

    # Query Quran Dense (original vs normalized)
    print("\n--- Quran Dense Retrieval (Original Query) ---")
    quran_col = client.get_collection("quran_dense")
    res_quran_orig = quran_col.query(query_embeddings=[q_dense.tolist()], n_results=3)
    for i, (doc, meta, dist) in enumerate(zip(res_quran_orig["documents"][0], res_quran_orig["metadatas"][0], res_quran_orig["distances"][0])):
        print(f"  #{i+1}: {meta.get('id')} | dist={dist:.4f} | surah={meta.get('surah_name_en')} | ayah={meta.get('ayah_num')}")
        print(f"      Doc preview: {doc[:150]}")

    print("\n--- Quran Dense Retrieval (Normalized Query) ---")
    res_quran_norm = quran_col.query(query_embeddings=[q_dense_norm.tolist()], n_results=3)
    for i, (doc, meta, dist) in enumerate(zip(res_quran_norm["documents"][0], res_quran_norm["metadatas"][0], res_quran_norm["distances"][0])):
        print(f"  #{i+1}: {meta.get('id')} | dist={dist:.4f} | surah={meta.get('surah_name_en')} | ayah={meta.get('ayah_num')}")
        print(f"      Doc preview: {doc[:150]}")

    # Query Tafsir Arabic Dense (original vs normalized)
    print("\n--- Tafsir Arabic Dense Retrieval (Original Query) ---")
    tafsir_col = client.get_collection("tafsir_ar_dense")
    res_tafsir_orig = tafsir_col.query(query_embeddings=[q_dense.tolist()], n_results=3)
    for i, (doc, meta, dist) in enumerate(zip(res_tafsir_orig["documents"][0], res_tafsir_orig["metadatas"][0], res_tafsir_orig["distances"][0])):
        print(f"  #{i+1}: {meta.get('id')} | dist={dist:.4f} | surah={meta.get('surah_name_en')} | ayah={meta.get('ayah_num')}")
        print(f"      Doc preview: {doc[:150]}")

    print("\n--- Tafsir Arabic Dense Retrieval (Normalized Query) ---")
    res_tafsir_norm = tafsir_col.query(query_embeddings=[q_dense_norm.tolist()], n_results=3)
    for i, (doc, meta, dist) in enumerate(zip(res_tafsir_norm["documents"][0], res_tafsir_norm["metadatas"][0], res_tafsir_norm["distances"][0])):
        print(f"  #{i+1}: {meta.get('id')} | dist={dist:.4f} | surah={meta.get('surah_name_en')} | ayah={meta.get('ayah_num')}")
        print(f"      Doc preview: {doc[:150]}")
        
    # Let's inspect sparse retrieval
    print("\nLoading sparse weights from JSON for hadith...")
    sparse_path = os.path.join(SPARSE_DIR, "hadith_sparse.json")
    with open(sparse_path, "r", encoding="utf-8") as f:
        hadith_sparse = json.load(f)
        
    print(f"Loaded {len(hadith_sparse)} sparse hadith vectors. Matching...")
    
    # Simple sparse scorer
    # Compute dot product between query sparse weights and doc sparse weights
    def get_sparse_score(q_sp, doc_sp):
        score = 0.0
        indices = doc_sp.get("indices", [])
        values = doc_sp.get("values", [])
        doc_map = {int(idx): val for idx, val in zip(indices, values)}
        for token_id, q_weight in q_sp.items():
            token_int = int(token_id)
            if token_int in doc_map:
                score += q_weight * doc_map[token_int]
        return score
        
    # Load raw chunks to get documents and metadata for sparse results
    chunks_path = os.path.join(DATA_DIR, "processed/hadith_chunks.jsonl")
    chunks = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                chunks[c["id"]] = c
                
    sparse_scores = []
    for doc_id, doc_sp in hadith_sparse.items():
        score = get_sparse_score(q_sparse, doc_sp)
        if score > 0:
            sparse_scores.append((doc_id, score))
            
    sparse_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- Hadith Sparse Retrieval (Original Query) ---")
    for i, (doc_id, score) in enumerate(sparse_scores[:5]):
        c = chunks.get(doc_id, {})
        print(f"  #{i+1}: {doc_id} | score={score:.4f} | collection={c.get('collection')} | grade={c.get('grade')}")
        print(f"      Doc preview: {c.get('embedding_text')[:150]}")

    # Now let's calculate for normalized sparse query
    sparse_scores_norm = []
    for doc_id, doc_sp in hadith_sparse.items():
        score = get_sparse_score(q_sparse_norm, doc_sp)
        if score > 0:
            sparse_scores_norm.append((doc_id, score))
            
    sparse_scores_norm.sort(key=lambda x: x[1], reverse=True)
    print("\n--- Hadith Sparse Retrieval (Normalized Query) ---")
    for i, (doc_id, score) in enumerate(sparse_scores_norm[:5]):
        c = chunks.get(doc_id, {})
        print(f"  #{i+1}: {doc_id} | score={score:.4f} | collection={c.get('collection')} | grade={c.get('grade')}")
        print(f"      Doc preview: {c.get('embedding_text')[:150]}")

if __name__ == "__main__":
    main()
