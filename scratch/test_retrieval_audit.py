import os
import json
import time
import numpy as np
import torch
import chromadb
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_DIR = "/Users/mahmoud/Documents/nur/nur/data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
SPARSE_DIR = os.path.join(DATA_DIR, "sparse")

class DirectReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", device="mps"):
        self.device = device
        print(f"Loading {model_name} on {device} via HuggingFace...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        
    def compute_score(self, pairs, batch_size=16):
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.view(-1).float().cpu().numpy().tolist()
                scores.extend(batch_scores)
        return scores

def normalize_arabic(text: str, for_retrieval: bool = True) -> str:
    import re
    if not text:
        return ""
    # Strip diacritics
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]', '', text)
    # Remove tatweel
    text = re.sub(r'\u0640', '', text)
    # Normalize alef variants: أ إ آ → ا
    text = text.replace('\u0623', '\u0627')
    text = text.replace('\u0625', '\u0627')
    text = text.replace('\u0622', '\u0627')
    if for_retrieval:
        text = text.replace('\u0649', '\u064A')
        text = text.replace('\u0629', '\u0647')
    return re.sub(r'\s+', ' ', text).strip()

def get_sparse_scores(q_sparse, sparse_db, top_n=100):
    """Compute dot product score between query and sparse weights"""
    scores = []
    q_map = {int(k): float(v) for k, v in q_sparse.items()}
    
    for doc_id, doc_data in sparse_db.items():
        indices = doc_data.get("indices", [])
        values = doc_data.get("values", [])
        
        score = 0.0
        for idx, val in zip(indices, values):
            if idx in q_map:
                score += q_map[idx] * val
        if score > 0:
            scores.append((doc_id, score))
            
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

def reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=25):
    """
    RRF Algorithm to fuse dense and sparse ranks.
    dense_results: list of (doc_id, score) sorted desc
    sparse_results: list of (doc_id, score) sorted desc
    """
    rrf_scores = {}
    
    # Process dense ranks
    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    # Process sparse ranks
    for rank, (doc_id, _) in enumerate(sparse_results, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_rrf[:top_n]

def load_all_chunks():
    """Load raw text chunks for quick display lookup"""
    chunks = {}
    for filename in ["quran_chunks.jsonl", "hadith_chunks.jsonl", "tafsir_chunks.jsonl"]:
        path = os.path.join(PROCESSED_DIR, filename)
        if os.path.exists(path):
            print(f"Loading {filename} into memory...")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = json.loads(line)
                        chunks[c["id"]] = c
    return chunks

def load_sparse_dbs():
    """Load all sparse JSON databases"""
    sparse_dbs = {}
    for source in ["quran", "hadith", "tafsir_ar", "tafsir_en"]:
        path = os.path.join(SPARSE_DIR, f"{source}_sparse.json")
        if os.path.exists(path):
            print(f"Loading sparse index: {source}_sparse.json...")
            with open(path, "r", encoding="utf-8") as f:
                sparse_dbs[source] = json.load(f)
    return sparse_dbs

def main():
    print("=" * 60)
    print("NUR — RETRIEVAL AUDIT WITH BGE-RERANKER-V2-M3")
    print("=" * 60)
    
    # 1. Load Data
    raw_chunks = load_all_chunks()
    sparse_dbs = load_sparse_dbs()
    
    # 2. Load Models
    print("\nLoading BGEM3 Model on MPS...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="mps")
    
    print("\nLoading BGE Reranker on MPS...")
    reranker = DirectReranker("BAAI/bge-reranker-v2-m3", device="mps")
    
    # 3. Connect to ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Define test queries across different categories
    test_queries = [
        # --- Arabic Queries (Islamic Fiqh / Legal Rulings) ---
        {"id": "ar_q1", "lang": "AR", "collection": "hadith", "q": "ما حكم صلاة الجماعة"},
        {"id": "ar_q2", "lang": "AR", "collection": "quran", "q": "الربا وأكل أموال الناس بالباطل"},
        {"id": "ar_q3", "lang": "AR", "collection": "hadith", "q": "أحاديث عن الصبر عند المصيبة"},
        {"id": "ar_q4", "lang": "AR", "collection": "hadith", "q": "صلاة الكسوف وكيفيتها"},
        
        # --- English Queries (Cross-lingual Semantic Search) ---
        {"id": "en_q1", "lang": "EN", "collection": "hadith", "q": "Ruling on congregational prayer"},
        {"id": "en_q2", "lang": "EN", "collection": "quran", "q": "What does the Quran say about charity and zakat?"},
        {"id": "en_q3", "lang": "EN", "collection": "hadith", "q": "Ahadith about good character and manners"},
        {"id": "en_q4", "lang": "EN", "collection": "hadith", "q": "Patience and reward in trials"},
        {"id": "en_q5", "lang": "EN", "collection": "quran", "q": "Inheritance rules in Islam"},
        
        # --- French Queries (Cross-lingual Semantic Search) ---
        {"id": "fr_q1", "lang": "FR", "collection": "hadith", "q": "Comment faire les ablutions?"},
        {"id": "fr_q2", "lang": "FR", "collection": "hadith", "q": "Quel est le statut de l'usure Riba?"},
        {"id": "fr_q3", "lang": "FR", "collection": "hadith", "q": "Le comportement envers les voisins en Islam"},
        {"id": "fr_q4", "lang": "FR", "collection": "hadith", "q": "L'importance de la recherche de la science"},
    ]
    
    audit_report = []
    
    for idx, item in enumerate(test_queries, start=1):
        q_id = item["id"]
        q_text = item["q"]
        lang = item["lang"]
        source = item["collection"]
        
        print(f"\n[{idx}/{len(test_queries)}] Auditing Query '{q_text}' ({lang}) on '{source}'...")
        
        # Normalize if Arabic
        processed_query = q_text
        if lang == "AR":
            processed_query = normalize_arabic(q_text, for_retrieval=True)
            
        # Get query embeddings (dense & sparse)
        q_output = model.encode(
            [processed_query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        q_dense = q_output['dense_vecs'][0]
        if hasattr(q_dense, 'cpu'):
            q_dense = q_dense.cpu().numpy()
        norm = np.linalg.norm(q_dense)
        if norm > 0:
            q_dense = q_dense / norm
            
        q_sparse = q_output['lexical_weights'][0]
        
        # 1. Execute Dense Retrieval
        dense_results = []
        try:
            col = client.get_collection(f"{source}_dense")
            res = col.query(query_embeddings=[q_dense.tolist()], n_results=100)
            
            for doc_id, dist in zip(res["ids"][0], res["distances"][0]):
                similarity = 1.0 - (dist / 2.0)
                dense_results.append((doc_id, similarity))
        except Exception as e:
            print(f"  ChromaDB query error: {e}")
            
        # 2. Execute Sparse Retrieval
        sparse_results = []
        db_key = source
        if source == "tafsir":
            db_key = "tafsir_ar" if lang == "AR" else "tafsir_en"
            
        if db_key in sparse_dbs:
            sparse_results = get_sparse_scores(q_sparse, sparse_dbs[db_key], top_n=100)
            
        # 3. Fuse via RRF (Top 25)
        hybrid_candidates = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=25)
        
        # 4. Rerank via BGE-Reranker-v2-M3
        rerank_pairs = []
        candidate_ids = []
        for doc_id, _ in hybrid_candidates:
            chunk_data = raw_chunks.get(doc_id, {})
            doc_text = chunk_data.get("embedding_text", "")
            rerank_pairs.append([q_text, doc_text])
            candidate_ids.append(doc_id)
            
        rerank_scores = []
        if rerank_pairs:
            scores = reranker.compute_score(rerank_pairs)
            for doc_id, score in zip(candidate_ids, scores):
                rerank_scores.append((doc_id, score))
                
        rerank_scores.sort(key=lambda x: x[1], reverse=True)
        top_reranked = rerank_scores[:5]
        
        # Format results for the report
        query_audit = {
            "query": q_text,
            "lang": lang,
            "source": source,
            "dense": dense_results[:10],
            "sparse": sparse_results[:10],
            "hybrid": hybrid_candidates[:10],
            "reranked": top_reranked
        }
        audit_report.append(query_audit)
        
        # Print comparisons
        print("  Top Reranked Results:")
        for r, (doc_id, score) in enumerate(top_reranked, start=1):
            chunk_data = raw_chunks.get(doc_id, {})
            text_preview = chunk_data.get("embedding_text", "")[:120]
            
            # Find original ranks
            hybrid_rank = next((rank for rank, (h_id, _) in enumerate(hybrid_candidates, start=1) if h_id == doc_id), "N/A")
            dense_rank = next((rank for rank, (d_id, _) in enumerate(dense_results, start=1) if d_id == doc_id), "N/A")
            sparse_rank = next((rank for rank, (d_id, _) in enumerate(sparse_results, start=1) if d_id == doc_id), "N/A")
            
            print(f"    #{r}: {doc_id} | Rerank Score={score:.5f} | Hybrid Rank={hybrid_rank} | Dense={dense_rank} | Sparse={sparse_rank}")
            print(f"        Preview: {text_preview}...")

    # Write Markdown Audit Report
    report_path = "/Users/mahmoud/Documents/nur/nur/scratch/retrieval_audit_results.md"
    print(f"\nWriting full audit results to {report_path}...")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# NUR Retrieval Algorithm Audit Report (With BGE-Reranker-v2-M3)\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Evaluation comparing: **Dense-Only** (ChromaDB), **Sparse-Only** (BGE-M3 JSON Lexical), **Hybrid RRF** (Fusing Dense + Sparse), and **Reranked** (BGE-Reranker-v2-M3).\n\n")
        
        for idx, item in enumerate(audit_report, start=1):
            f.write(f"## {idx}. Query: \"{item['query']}\" ({item['lang']} on {item['source']})\n\n")
            f.write("| Rank | Reranked Doc | Reranker Score | Hybrid Rank | Dense Rank | Sparse Rank | Text Preview |\n")
            f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :--- |\n")
            
            for r, (doc_id, score) in enumerate(item["reranked"], start=1):
                chunk = raw_chunks.get(doc_id, {})
                preview = chunk.get("embedding_text", "").replace("\n", " ").replace("|", "\\|")[:150]
                
                hybrid_rank = next((rank for rank, (h_id, _) in enumerate(item["hybrid"], start=1) if h_id == doc_id), "N/A")
                dense_rank = next((rank for rank, (d_id, _) in enumerate(item["dense"], start=1) if d_id == doc_id), "N/A")
                sparse_rank = next((rank for rank, (d_id, _) in enumerate(item["sparse"], start=1) if d_id == doc_id), "N/A")
                
                f.write(f"| #{r} | `{doc_id}` | {score:.5f} | {hybrid_rank} | {dense_rank} | {sparse_rank} | {preview}... |\n")
            f.write("\n")
            
    print("\n" + "=" * 60)
    print("AUDIT COMPLETE! Open scratch/retrieval_audit_results.md to view the full report.")
    print("=" * 60)

if __name__ == "__main__":
    main()
