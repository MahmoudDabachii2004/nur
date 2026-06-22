import os
import json
import numpy as np
import re
import chromadb
from FlagEmbedding import BGEM3FlagModel

DATA_DIR = "/Users/mahmoud/Documents/nur/nur/data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Normalization Functions
def normalize_strategy_1(text: str) -> str:
    """Strategy 1 (Strict/Current): Strip diacritics, tatweel, normalize alef, ة->ه, ى->ي"""
    if not text:
        return ""
    # Strip diacritics
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]', '', text)
    # Remove tatweel
    text = re.sub(r'\u0640', '', text)
    # Normalize alef variants: أ إ آ -> ا
    text = text.replace('\u0623', '\u0627')
    text = text.replace('\u0625', '\u0627')
    text = text.replace('\u0622', '\u0627')
    # ى -> ي
    text = text.replace('\u0649', '\u064A')
    # ة -> ه
    text = text.replace('\u0629', '\u0647')
    return re.sub(r'\s+', ' ', text).strip()

def normalize_strategy_2(text: str) -> str:
    """Strategy 2 (Standard/Grammatical): Strip diacritics, tatweel, normalize alef. Keep ة and ى intact."""
    if not text:
        return ""
    # Strip diacritics
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]', '', text)
    # Remove tatweel
    text = re.sub(r'\u0640', '', text)
    # Normalize alef variants: أ إ آ -> ا
    text = text.replace('\u0623', '\u0627')
    text = text.replace('\u0625', '\u0627')
    text = text.replace('\u0622', '\u0627')
    return re.sub(r'\s+', ' ', text).strip()

def normalize_strategy_3(text: str) -> str:
    """Strategy 3 (Minimal): Strip diacritics and tatweel. Keep alef, ة, and ى intact."""
    if not text:
        return ""
    # Strip diacritics
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]', '', text)
    # Remove tatweel
    text = re.sub(r'\u0640', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def main():
    print("Loading BGEM3 model...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    
    # Load chunks
    chunks_path = os.path.join(PROCESSED_DIR, "hadith_chunks.jsonl")
    all_chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_chunks.append(json.loads(line))
                
    print(f"Loaded {len(all_chunks)} total hadiths.")
    
    # Identify target correct hadiths about congregational prayer
    # hadith_nasai_840, hadith_nasai_841, hadith_abudawud_559, hadith_bukhari_630, hadith_bukhari_2044
    target_ids = ["hadith_nasai_840", "hadith_nasai_841", "hadith_abudawud_559", "hadith_bukhari_630", "hadith_bukhari_2044"]
    target_chunks = [c for c in all_chunks if c["id"] in target_ids]
    
    # Also find other chunks with keyword "جماعة" to evaluate retrieval properly
    jamaah_chunks = [c for c in all_chunks if "جماعة" in c.get("text_ar", "") and c["id"] not in target_ids]
    
    # Construct a representative evaluation set
    # target chunks + 20 other jamaah chunks + 150 unrelated chunks to act as distractors
    unrelated_chunks = [c for c in all_chunks if "جماعة" not in c.get("text_ar", "") and "صل" not in c.get("text_ar", "")][:150]
    
    eval_chunks = target_chunks + jamaah_chunks[:25] + unrelated_chunks
    print(f"Evaluation set size: {len(eval_chunks)} (Targets: {len(target_chunks)}, Distractors: {len(eval_chunks) - len(target_chunks)})")
    
    # Define combinations to test
    # Normalization strategies: 1, 2, 3
    # Formats: 
    #   - A (Arabic Only): "باب: {chap_ar} | متن: {text_ar}"
    #   - B (Bilingual): "باب: {chap_ar} | متن: {text_ar} | Title: {chap_en} | Hadith: {text_en}"
    
    results = []
    
    strategies = [
        {"name": "S1 (Strict: ة->ه, ى->ي, أ->ا)", "func": normalize_strategy_1},
        {"name": "S2 (Standard: أ->ا, keep ة/ى)", "func": normalize_strategy_2},
        {"name": "S3 (Minimal: keep all letters)", "func": normalize_strategy_3}
    ]
    
    formats = [
        {"name": "AR-Only", "has_en": False},
        {"name": "Bilingual (AR + EN)", "has_en": True}
    ]
    
    client = chromadb.Client()
    col_idx = 0
    
    for strategy in strategies:
        norm_func = strategy["func"]
        for fmt in formats:
            run_name = f"{strategy['name']} | {fmt['name']}"
            print(f"\nBuilding collection for: {run_name}...")
            
            # Create unique collection name using a simple numeric index to avoid ChromaDB validation errors
            col_name = f"eval_run_{col_idx}"
            col_idx += 1
            try:
                col = client.create_collection(col_name)
            except Exception:
                client.delete_collection(col_name)
                col = client.create_collection(col_name)
                
            ids = []
            docs = []
            metas = []
            
            for c in eval_chunks:
                ids.append(c["id"])
                
                # Build Arabic part
                chap_ar = norm_func(c.get("chapter_title_ar", ""))
                text_ar = norm_func(c.get("text_ar", ""))
                
                doc_str = ""
                if chap_ar:
                    doc_str += f"باب: {chap_ar} | "
                doc_str += f"متن: {text_ar}"
                
                # Build Bilingual part if needed
                if fmt["has_en"]:
                    chap_en = c.get("chapter_title_en", "")
                    text_en = c.get("text_en", "")
                    if chap_en:
                        doc_str += f" | Title: {chap_en}"
                    if text_en:
                        doc_str += f" | Hadith: {text_en}"
                        
                docs.append(doc_str)
                metas.append({"id": c["id"], "collection": c["collection"]})
                
            # Embed documents
            embeddings = model.encode(docs, return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
            if hasattr(embeddings, 'cpu'):
                embeddings = embeddings.cpu().numpy()
                
            # Normalize embeddings for cosine distance
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
            
            col.add(ids=ids, embeddings=embeddings.tolist(), documents=docs, metadatas=metas)
            
            # Test queries
            test_queries = [
                # Arabic standard query
                {"label": "AR (Original: صلاة الجماعة)", "q": "ما حكم صلاة الجماعة"},
                # Arabic spelling variant query
                {"label": "AR (Spelling-variant: صلاه الجماعه)", "q": "ما حكم صلاه الجماعه"},
                # English query
                {"label": "EN (congregational prayer)", "q": "What is the ruling on congregational prayer?"},
                # French query
                {"label": "FR (prière en commun)", "q": "Quel est le statut de la prière en commun?"}
            ]
            
            for q_info in test_queries:
                q_text = q_info["q"]
                # Apply same strategy normalization if querying in Arabic
                if "AR" in q_info["label"]:
                    q_text = norm_func(q_text)
                    
                # Embed query
                q_emb = model.encode([q_text], return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs'][0]
                norm = np.linalg.norm(q_emb)
                if norm > 0:
                    q_emb = q_emb / norm
                    
                # Query collection
                res = col.query(query_embeddings=[q_emb.tolist()], n_results=10)
                
                # Evaluate results
                top_ids = res["ids"][0]
                top_dists = res["distances"][0]
                
                # Check ranks and distances of target hadiths
                target_metrics = {}
                for t_id in target_ids:
                    if t_id in top_ids:
                        idx = top_ids.index(t_id)
                        target_metrics[t_id] = {"rank": idx + 1, "dist": top_dists[idx]}
                    else:
                        target_metrics[t_id] = {"rank": ">10", "dist": None}
                        
                # Count how many targets in top-5 and top-10
                in_top_5 = sum(1 for t_id in target_ids if t_id in top_ids[:5])
                in_top_10 = sum(1 for t_id in target_ids if t_id in top_ids[:10])
                
                # Get distance of #1 result
                best_dist = top_dists[0] if top_dists else None
                best_id = top_ids[0] if top_ids else None
                
                results.append({
                    "strategy": strategy["name"],
                    "format": fmt["name"],
                    "query_label": q_info["label"],
                    "query_text": q_text,
                    "best_id": best_id,
                    "best_dist": best_dist,
                    "in_top_5": in_top_5,
                    "in_top_10": in_top_10,
                    "targets": target_metrics
                })

    # Print out results in structured Markdown table format
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print("| Strategy & Format | Query | Best Match (Dist) | Targets in Top 5 / 10 | Target Ranks & Distances |")
    print("|---|---|---|---|---|")
    for r in results:
        target_summary = []
        for t_id, metrics in r["targets"].items():
            t_short = t_id.replace("hadith_", "")
            if metrics["rank"] != ">10":
                target_summary.append(f"{t_short}: #{metrics['rank']} ({metrics['dist']:.3f})")
            else:
                target_summary.append(f"{t_short}: >10")
        target_str = ", ".join(target_summary)
        
        best_match_str = f"{r['best_id']} ({r['best_dist']:.3f})" if r['best_id'] else "None"
        
        print(f"| {r['strategy']} + {r['format']} | {r['query_label']} | {best_match_str} | {r['in_top_5']}/5, {r['in_top_10']}/10 | {target_str} |")

if __name__ == "__main__":
    main()
