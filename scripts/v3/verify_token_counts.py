"""Vérifie précisément le nombre de tokens des chunks V3 avec le tokenizer BGE-M3."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 1. Charger le tokenizer XLM-RoBERTa (celui de BGE-M3)
print("=" * 60)
print("VÉRIFICATION TOKEN COUNT — BGE-M3 tokenizer")
print("=" * 60)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
print(f"Tokenizer loaded: {type(tokenizer).__name__}")
print(f"Model max length: {tokenizer.model_max_length}")
print()

# 2. Charger TOUS les chunks V3 et mesurer leurs tokens
print("Loading all V3 chunks...")
all_chunks = []
with open(PROJECT_ROOT / "data" / "processed" / "quran_v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        all_chunks.append(("quran", json.loads(line)))

with open(PROJECT_ROOT / "data" / "processed" / "hadith_v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        all_chunks.append(("hadith", json.loads(line)))

print(f"Total chunks: {len(all_chunks):,}")
print()

# 3. Tokeniser chaque chunk et mesurer
print("Tokenizing all chunks (this takes ~30 sec)...")
token_counts = {"quran": [], "hadith": []}
max_chunk_info = {"tokens": 0, "kind": "", "id": "", "chars": 0}

for kind, chunk in all_chunks:
    doc = chunk.get("embedding_text", "")
    if not doc:
        continue
    # Tokeniser avec le tokenizer de BGE-M3 (pas de troncature)
    tokens = tokenizer.encode(doc, add_special_tokens=True, truncation=False)
    n_tokens = len(tokens)
    token_counts[kind].append(n_tokens)
    
    if n_tokens > max_chunk_info["tokens"]:
        max_chunk_info = {
            "tokens": n_tokens,
            "kind": kind,
            "id": chunk["id"],
            "chars": len(doc),
        }

# 4. Stats
import statistics
print()
print("=" * 60)
print("RÉSULTATS — TOKEN COUNT PAR CHUNK")
print("=" * 60)

for kind in ("quran", "hadith"):
    counts = token_counts[kind]
    if not counts:
        continue
    counts_sorted = sorted(counts)
    n = len(counts_sorted)
    print(f"\n[{kind.upper()}] ({n:,} chunks)")
    print(f"  Min tokens:    {counts_sorted[0]:,}")
    print(f"  Median tokens: {counts_sorted[n//2]:,}")
    print(f"  Mean tokens:   {statistics.mean(counts):,.0f}")
    print(f"  p90 tokens:    {counts_sorted[int(n*0.9)]:,}")
    print(f"  p95 tokens:    {counts_sorted[int(n*0.95)]:,}")
    print(f"  p99 tokens:    {counts_sorted[int(n*0.99)]:,}")
    print(f"  Max tokens:    {counts_sorted[-1]:,}")

print()
print("=" * 60)
print("CHUNK LE PLUS LONG")
print("=" * 60)
print(f"  Kind:   {max_chunk_info['kind']}")
print(f"  ID:     {max_chunk_info['id']}")
print(f"  Chars:  {max_chunk_info['chars']:,}")
print(f"  Tokens: {max_chunk_info['tokens']:,}")

print()
print("=" * 60)
print("VÉRIFICATION max_length=8192")
print("=" * 60)

# Compter combien de chunks dépassent 8192 tokens
over_8192 = 0
over_4096 = 0
over_2048 = 0
for kind in ("quran", "hadith"):
    for n in token_counts[kind]:
        if n > 8192:
            over_8192 += 1
        if n > 4096:
            over_4096 += 1
        if n > 2048:
            over_2048 += 1

total_chunks = sum(len(token_counts[k]) for k in token_counts)
print(f"  Total chunks: {total_chunks:,}")
print(f"  Chunks > 2048 tokens: {over_2048:,} ({over_2048/total_chunks*100:.2f}%)")
print(f"  Chunks > 4096 tokens: {over_4096:,} ({over_4096/total_chunks*100:.2f}%)")
print(f"  Chunks > 8192 tokens: {over_8192:,} ({over_8192/total_chunks*100:.2f}%)")

print()
if over_8192 == 0:
    print(f"  ✅ max_length=8192 est SUFFISANT")
    print(f"     Le plus long chunk fait {max_chunk_info['tokens']:,} tokens")
    print(f"     Marge de sécurité: {8192 - max_chunk_info['tokens']:,} tokens")
else:
    print(f"  ❌ max_length=8192 est INSUFFISANT")
    print(f"     {over_8192:,} chunks dépassent 8192 tokens")
    print(f"     Le plus long fait {max_chunk_info['tokens']:,} tokens")
    print(f"     Ces chunks seront tronqués!")
