"""Compare stored ChromaDB document vs JSONL embedding_text for 2:255."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import chromadb

client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "data" / "chroma_db_v3"))
col = client.get_collection("quran_v3_dense")

# Get the stored document for 2:255
result = col.get(ids=["SRC-QURAN-2-255"], include=["documents"])
stored_doc = result["documents"][0]

# Get the embedding_text from the jsonl
jsonl_doc = None
with open(PROJECT_ROOT / "data" / "processed" / "quran_v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj["id"] == "SRC-QURAN-2-255":
            jsonl_doc = obj.get("embedding_text", "")
            break

print(f"Stored doc length: {len(stored_doc)}")
print(f"JSONL doc length:  {len(jsonl_doc)}")
print(f"Are they identical? {stored_doc == jsonl_doc}")
print()

if stored_doc != jsonl_doc:
    print("=== DIFFERENCE FOUND ===")
    # Find first difference
    for i, (a, b) in enumerate(zip(stored_doc, jsonl_doc)):
        if a != b:
            print(f"First diff at char {i}:")
            stored_ctx = stored_doc[max(0, i-30):i+50]
            jsonl_ctx = jsonl_doc[max(0, i-30):i+50]
            print(f"  Stored: ...{stored_ctx!r}...")
            print(f"  JSONL:  ...{jsonl_ctx!r}...")
            break
    if len(stored_doc) != len(jsonl_doc):
        print(f"Length diff: stored={len(stored_doc)}, jsonl={len(jsonl_doc)}")
        if len(stored_doc) > len(jsonl_doc):
            extra = stored_doc[len(jsonl_doc):][:200]
            print(f"Extra in stored: {extra!r}")
        else:
            extra = jsonl_doc[len(stored_doc):][:200]
            print(f"Extra in jsonl: {extra!r}")

print()
print("=== First 300 chars comparison ===")
print(f"Stored: {stored_doc[:300]!r}")
print()
print(f"JSONL:  {jsonl_doc[:300]!r}")
print()

# Also check: does the stored doc contain the verse text?
print("=== Does stored doc contain verse AR text? ===")
meta_result = col.get(ids=["SRC-QURAN-2-255"], include=["metadatas"])
meta = meta_result["metadatas"][0]
verse_ar = meta.get("text_ar", "")
verse_en = meta.get("text_en", "")
print(f"Verse AR ({len(verse_ar)} chars): {verse_ar!r}")
print()
print(f"Verse AR in stored doc? {verse_ar in stored_doc}")
print(f"Verse EN in stored doc? {verse_en in stored_doc}")
print()

# Check what's actually in the stored doc (search for 'Ayat al-Kursi' or 'Allah')
print("=== Search for key terms in stored doc ===")
for term in ["Ayat al-Kursi", "Ayat al-Kursi", "throne", "Allah", "Ever-Living", "الحي", "القيوم", "كرسي"]:
    if term in stored_doc:
        idx = stored_doc.find(term)
        print(f"  '{term}' found at char {idx}: ...{stored_doc[max(0,idx-20):idx+60]!r}...")
    else:
        print(f"  '{term}' NOT FOUND in stored doc")
