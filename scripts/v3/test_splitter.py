"""
NUR V3 — Tafsir Smart Splitter

Splits long tafsir texts into sub-chunks of ~400 tokens, respecting
sentence boundaries. Never cuts a sentence in half.

Rules:
1. If text ≤ 400 tokens → 1 chunk (no split)
2. If text > 400 tokens → split on sentence boundaries
3. Overlap of 1 sentence between chunks (context preservation)
4. Each sub-chunk gets: parent_verse_id, parent_tafsir_id, chunk_index, total_chunks
5. Full text preserved in parent_tafsir_id for LLM display

Usage:
  python scripts/v3/test_splitter.py
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def split_tafsir_text(text, max_tokens=400, overlap_sentences=1):
    """Split a tafsir text into sub-chunks respecting sentence boundaries.
    
    Args:
        text: The full tafsir text
        max_tokens: Target max tokens per chunk (we use chars as proxy: ~4 chars/token for EN, ~2 for AR)
        overlap_sentences: Number of sentences to overlap between chunks
        
    Returns:
        List of (chunk_text, chunk_index, total_chunks) tuples
    """
    if not text or not text.strip():
        return [(text, 0, 1)]
    
    # Estimate max chars based on mixed AR/EN content
    # AR: ~2 chars/token, EN: ~4 chars/token. Mixed: ~3 chars/token
    # Use conservative 3 chars/token → 400 tokens ≈ 1200 chars
    max_chars = max_tokens * 3
    
    # If text is short enough, return as single chunk
    if len(text) <= max_chars:
        return [(text, 0, 1)]
    
    # Split into sentences using multiple delimiters
    # We split on: . ! ? ؟ ۔ » " ) followed by space or newline
    sentence_endings = r'(?<=[.!?؟۔»")\]])\s+'
    sentences = re.split(sentence_endings, text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 1:
        # Can't split by sentence, fall back to char-based split at word boundaries
        chunks = []
        words = text.split()
        current_chunk = []
        current_len = 0
        
        for word in words:
            word_len = len(word) + 1
            if current_len + word_len > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = word_len
            else:
                current_chunk.append(word)
                current_len += word_len
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return [(chunk, i, len(chunks)) for i, chunk in enumerate(chunks)]
    
    # Group sentences into chunks respecting max_chars
    chunks = []
    current_sentences = []
    current_len = 0
    
    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence) + 1  # +1 for the space
        
        # If adding this sentence would exceed max AND we already have content
        if current_len + sentence_len > max_chars and current_sentences:
            # Save current chunk
            chunks.append(" ".join(current_sentences))
            
            # Start new chunk with overlap
            overlap = current_sentences[-overlap_sentences:] if overlap_sentences > 0 else []
            current_sentences = overlap + [sentence]
            current_len = sum(len(s) + 1 for s in current_sentences)
        else:
            current_sentences.append(sentence)
            current_len += sentence_len
    
    # Don't forget the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    
    return [(chunk, i, len(chunks)) for i, chunk in enumerate(chunks)]


def test_splitter():
    """Test the splitter with real tafsir data."""
    print("=" * 70)
    print("TAFSIR SMART SPLITTER — TEST")
    print("=" * 70)
    
    # Load tafsir_v3.jsonl
    tafsir_path = PROJECT_ROOT / "data" / "processed" / "tafsir_v3.jsonl"
    if not tafsir_path.exists():
        print(f"Error: {tafsir_path} not found")
        return
    
    # Load all tafsirs
    all_tafsirs = []
    with tafsir_path.open("r", encoding="utf-8") as f:
        for line in f:
            all_tafsirs.append(json.loads(line))
    
    print(f"Loaded {len(all_tafsirs):,} tafsir chunks")
    
    # Split all tafsirs
    all_sub_chunks = []
    stats = {
        "original_count": len(all_tafsirs),
        "split_count": 0,
        "not_split_count": 0,
        "sub_chunk_count": 0,
        "max_sub_chunk_chars": 0,
        "max_sub_chunk_estimated_tokens": 0,
    }
    
    for tafsir in all_tafsirs:
        text = tafsir.get("text_full", "") or tafsir.get("embedding_text", "")
        if not text:
            continue
        
        sub_chunks = split_tafsir_text(text, max_tokens=400, overlap_sentences=1)
        
        if len(sub_chunks) > 1:
            stats["split_count"] += 1
        else:
            stats["not_split_count"] += 1
        
        for chunk_text, chunk_idx, total_chunks in sub_chunks:
            stats["sub_chunk_count"] += 1
            if len(chunk_text) > stats["max_sub_chunk_chars"]:
                stats["max_sub_chunk_chars"] = len(chunk_text)
                stats["max_sub_chunk_estimated_tokens"] = len(chunk_text) // 3
            
            sub_chunk = {
                "id": f"{tafsir['id']}-part{chunk_idx+1}",
                "kind": "tafsir",
                "source": tafsir.get("source", ""),
                "category": tafsir.get("category", ""),
                "language": tafsir.get("language", ""),
                "surah": tafsir.get("surah", ""),
                "ayah": tafsir.get("ayah", ""),
                "parent_verse_id": tafsir.get("parent_verse_id", ""),
                "parent_tafsir_id": tafsir["id"],
                "text_chunk": chunk_text,
                "text_full": text,  # Full text preserved for LLM
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks,
                "contains_isra_iliyyat": tafsir.get("contains_isra_iliyyat", False),
                "url": tafsir.get("url", ""),
                "embedding_text": f"[TAFSIR - {tafsir.get('source', '')} | {tafsir.get('language', '')}]\nSurah {tafsir.get('surah', '')}, Ayah {tafsir.get('ayah', '')}\n{chunk_text}",
                "build_version": "v3",
            }
            all_sub_chunks.append(sub_chunk)
    
    # Print stats
    print(f"\n{'=' * 50}")
    print(f"SPLITTING RESULTS")
    print(f"{'=' * 50}")
    print(f"Original tafsirs:     {stats['original_count']:,}")
    print(f"Tafsirs split:        {stats['split_count']:,} ({stats['split_count']/stats['original_count']*100:.1f}%)")
    print(f"Tafsirs not split:    {stats['not_split_count']:,} ({stats['not_split_count']/stats['original_count']*100:.1f}%)")
    print(f"Total sub-chunks:     {stats['sub_chunk_count']:,}")
    print(f"Avg sub-chunks/tafsir: {stats['sub_chunk_count']/stats['original_count']:.1f}")
    print(f"Max sub-chunk chars:  {stats['max_sub_chunk_chars']:,}")
    print(f"Max est. tokens:      {stats['max_sub_chunk_estimated_tokens']:,}")
    
    # Show examples
    print(f"\n{'=' * 50}")
    print(f"EXAMPLES (3 split tafsirs)")
    print(f"{'=' * 50}")
    
    examples_shown = 0
    for sub_chunk in all_sub_chunks:
        if sub_chunk["total_chunks"] > 1 and sub_chunk["chunk_index"] == 0:
            print(f"\n--- {sub_chunk['parent_tafsir_id']} ({sub_chunk['total_chunks']} parts) ---")
            print(f"  Parent verse: {sub_chunk['parent_verse_id']}")
            print(f"  Source: {sub_chunk['source']} ({sub_chunk['language']})")
            print(f"  Part 1 of {sub_chunk['total_chunks']}:")
            print(f"  Chars: {len(sub_chunk['text_chunk']):,}")
            print(f"  First 200 chars: {sub_chunk['text_chunk'][:200]}...")
            print(f"  Last 100 chars: ...{sub_chunk['text_chunk'][-100:]}")
            
            # Check if last char is a sentence ending
            last_char = sub_chunk['text_chunk'][-1] if sub_chunk['text_chunk'] else ''
            is_sentence_end = last_char in '.!?؟۔»")\]'
            print(f"  Ends with sentence delimiter: {'✅' if is_sentence_end else '❌'} ({repr(last_char)})")
            
            examples_shown += 1
            if examples_shown >= 3:
                break
    
    # Write new jsonl
    output_path = PROJECT_ROOT / "data" / "processed" / "tafsir_v3_split.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in all_sub_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    print(f"\n{'=' * 50}")
    print(f"OUTPUT: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"  {len(all_sub_chunks):,} sub-chunks written")
    print(f"{'=' * 50}")
    
    # Verify: check if any sub-chunk exceeds 512 tokens (estimated)
    over_512 = sum(1 for c in all_sub_chunks if len(c["text_chunk"]) > 1536)  # 512 * 3 chars
    print(f"\nVerification:")
    print(f"  Sub-chunks > 512 tokens (est): {over_512:,} ({over_512/len(all_sub_chunks)*100:.2f}%)")
    if over_512 == 0:
        print(f"  ✅ ALL sub-chunks fit within 512 tokens!")
    else:
        print(f"  ⚠️  {over_512} sub-chunks may exceed 512 tokens. Adjust max_tokens.")


if __name__ == "__main__":
    test_splitter()
