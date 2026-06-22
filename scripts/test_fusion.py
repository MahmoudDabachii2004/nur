"""
test_fusion.py

This script tests the RRFFuser module with synthetic ranked lists. It is a
pure-math test — no BGE-M3 model, no ChromaDB, no network. The goal is to
verify the RRF formula is implemented correctly by comparing the code's
output against hand-computed expected scores.

WHY THIS IS FULLY TESTABLE WITHOUT THE MODEL:
  RRFFuser.fuse() takes two ALREADY-COMPUTED ranked lists as input. It does
  not call BGE-M3 or ChromaDB. It only looks at the rank (position) of each
  chunk ID in each list and applies the formula:
    score_rrf(c) = alpha * 1/(k + rank_dense(c)) + (1-alpha) * 1/(k + rank_sparse(c))
  So we can feed it fake chunk IDs ("A", "B", "C"...) and hand-compute the
  expected scores with a calculator. The integration test (encode real query
  → dense search → sparse search → fuse) is a separate concern, tested on
  the user's Mac where the BGE-M3 model is cached.

TEST CASES:
  1. Hand-computed scores — verify exact numerical match for a small example.
  2. Chunk in both lists (the ideal case — gets both terms, boosted score).
  3. Chunk in only dense list (gets only the dense term, no penalty).
  4. Chunk in only sparse list (gets only the sparse term, no penalty).
  5. Chunk in neither list (excluded entirely).
  6. top_k truncation.
  7. Empty inputs (no crash).
  8. Determinism (same inputs → same output).
  9. Config defaults match the architecture doc (k=25, alpha=0.4).
"""

from __future__ import annotations

import os
import sys

# Add the src directory to the Python path so we can import the nur package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nur.retriever.fusion import RRFFuser
from src.nur.config import settings


def make_results(chunk_ids: list[str]) -> list[dict]:
    """Build a fake ranked list of results for testing.

    Args:
        chunk_ids: Ordered list of chunk IDs (rank 1 = first).

    Returns:
        List of dicts in the shape returned by DenseRetriever/SparseRetriever.
        The 'similarity'/'score' values are fake — RRF ignores them.
    """
    return [{"id": cid, "score": 1.0} for cid in chunk_ids]


def test_hand_computed_scores() -> bool:
    """Verify exact RRF scores against hand-computed values.

    Setup (k=25, alpha=0.4):
      dense  = [A, B, C]   (A=rank1, B=rank2, C=rank3)
      sparse = [B, A, D]   (B=rank1, A=rank2, D=rank3)

    Hand computation:
      A: dense_rank=1, sparse_rank=2
         score = 0.4 * 1/(25+1) + 0.6 * 1/(25+2)
               = 0.4 * (1/26) + 0.6 * (1/27)
               = 0.4 * 0.038462 + 0.6 * 0.037037
               = 0.015385 + 0.022222
               = 0.037607
      B: dense_rank=2, sparse_rank=1
         score = 0.4 * 1/(25+2) + 0.6 * 1/(25+1)
               = 0.4 * 0.037037 + 0.6 * 0.038462
               = 0.014815 + 0.023077
               = 0.037892
      C: dense_rank=3, sparse_rank=None
         score = 0.4 * 1/(25+3) + 0
               = 0.4 * 0.035714
               = 0.014286
      D: dense_rank=None, sparse_rank=3
         score = 0 + 0.6 * 1/(25+3)
               = 0.6 * 0.035714
               = 0.021429

    Expected order: B (0.037892) > A (0.037607) > D (0.021429) > C (0.014286)
    """
    print(f"\n{'='*70}")
    print("TEST 1: Hand-computed RRF scores")
    print(f"{'='*70}")

    fuser = RRFFuser(k=25, alpha_dense=0.4)
    dense = make_results(["A", "B", "C"])
    sparse = make_results(["B", "A", "D"])

    result = fuser.fuse(dense, sparse, top_k=10)

    # Expected scores (hand-computed above, rounded to 6 decimal places)
    expected = {
        "B": 0.037892,
        "A": 0.037607,
        "D": 0.021429,
        "C": 0.014286,
    }
    expected_order = ["B", "A", "D", "C"]

    print(f"  dense  ranks: A=1, B=2, C=3")
    print(f"  sparse ranks: B=1, A=2, D=3")
    print()
    print(f"  {'Rank':<6}{'Chunk':<8}{'Code Score':<16}{'Expected':<16}{'Match'}")
    print(f"  {'-'*56}")

    all_match = True
    for i, res in enumerate(result):
        cid = res["id"]
        code_score = res["rrf_score"]
        exp_score = expected[cid]
        match = abs(code_score - exp_score) < 1e-5
        marker = "✅" if match else "❌"
        print(f"  {i+1:<6}{cid:<8}{code_score:<16.6f}{exp_score:<16.6f}{marker}")
        if not match:
            all_match = False

    # Verify order
    actual_order = [res["id"] for res in result]
    if actual_order != expected_order:
        print(f"\n  ❌ ORDER MISMATCH: expected {expected_order}, got {actual_order}")
        all_match = False
    else:
        print(f"\n  ✅ Order correct: {actual_order}")

    if all_match:
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL")
    return all_match


def test_chunk_in_both_lists() -> bool:
    """A chunk in both lists should score higher than the same chunk in only one."""
    print(f"\n{'='*70}")
    print("TEST 2: Chunk in both lists gets boosted score")
    print(f"{'='*70}")

    fuser = RRFFuser(k=25, alpha_dense=0.4)
    # "shared" is rank 1 in both — should beat "dense_only" (rank 1 dense, absent sparse)
    dense = make_results(["shared", "dense_only"])
    sparse = make_results(["shared", "sparse_only"])

    result = fuser.fuse(dense, sparse, top_k=10)
    scores = {res["id"]: res["rrf_score"] for res in result}

    print(f"  shared score:      {scores['shared']:.6f}")
    print(f"  dense_only score:  {scores['dense_only']:.6f}")
    print(f"  sparse_only score: {scores['sparse_only']:.6f}")

    # shared gets BOTH terms; the others get only one. shared must be highest.
    if scores["shared"] > scores["dense_only"] and scores["shared"] > scores["sparse_only"]:
        print("  ✅ PASS — shared chunk ranks #1")
        return True
    else:
        print("  ❌ FAIL — shared chunk should be #1")
        return False


def test_chunk_only_in_dense() -> bool:
    """A chunk only in dense should get only the dense term (no penalty)."""
    print(f"\n{'='*70}")
    print("TEST 3: Chunk only in dense gets only dense term")
    print(f"{'='*70}")

    fuser = RRFFuser(k=25, alpha_dense=0.4)
    dense = make_results(["only_dense"])
    sparse = make_results([])

    result = fuser.fuse(dense, sparse, top_k=10)
    if not result:
        print("  ❌ FAIL — no results returned")
        return False

    res = result[0]
    expected = 0.4 * (1.0 / (25 + 1))  # alpha * 1/(k+1)
    print(f"  Code score:     {res['rrf_score']:.6f}")
    print(f"  Expected score: {expected:.6f}  (= 0.4 * 1/26)")
    print(f"  dense_rank:     {res['dense_rank']}")
    print(f"  sparse_rank:    {res['sparse_rank']}")

    if (
        abs(res["rrf_score"] - expected) < 1e-6
        and res["dense_rank"] == 1
        and res["sparse_rank"] is None
    ):
        print("  ✅ PASS")
        return True
    else:
        print("  ❌ FAIL")
        return False


def test_chunk_in_neither() -> bool:
    """A chunk ID not in either list must not appear in results."""
    print(f"\n{'='*70}")
    print("TEST 4: Chunk in neither list is excluded")
    print(f"{'='*70}")

    fuser = RRFFuser(k=25, alpha_dense=0.4)
    dense = make_results(["A", "B"])
    sparse = make_results(["B", "C"])

    result = fuser.fuse(dense, sparse, top_k=10)
    ids = [res["id"] for res in result]
    print(f"  Result IDs: {ids}")

    if "Z" not in ids and len(ids) == 3:  # A, B, C — union of both lists
        print("  ✅ PASS — only A, B, C present (no phantom chunks)")
        return True
    else:
        print("  ❌ FAIL")
        return False


def test_top_k_truncation() -> bool:
    """top_k must truncate the output to the requested length."""
    print(f"\n{'='*70}")
    print("TEST 5: top_k truncation")
    print(f"{'='*70}")

    fuser = RRFFuser(k=25, alpha_dense=0.4)
    # 10 unique chunks in dense, none in sparse — all tied at the same score
    dense = make_results([f"chunk_{i}" for i in range(10)])
    sparse = make_results([])

    result = fuser.fuse(dense, sparse, top_k=3)
    print(f"  Requested top_k=3, got {len(result)} results")
    if len(result) == 3:
        print("  ✅ PASS")
        return True
    else:
        print("  ❌ FAIL")
        return False


def test_empty_inputs() -> bool:
    """Empty inputs should not crash and should return empty list."""
    print(f"\n{'='*70}")
    print("TEST 6: Empty inputs")
    print(f"{'='*70}")

    fuser = RRFFuser(k=25, alpha_dense=0.4)
    result = fuser.fuse([], [], top_k=10)
    print(f"  fuse([], []) = {result}")
    if result == []:
        print("  ✅ PASS")
        return True
    else:
        print("  ❌ FAIL")
        return False


def test_determinism() -> bool:
    """Same inputs must produce identical output (stable sort)."""
    print(f"\n{'='*70}")
    print("TEST 7: Determinism")
    print(f"{'='*70}")

    fuser = RRFFuser(k=25, alpha_dense=0.4)
    dense = make_results(["A", "B", "C", "D", "E"])
    sparse = make_results(["C", "A", "E", "B", "D"])

    r1 = fuser.fuse(dense, sparse, top_k=10)
    r2 = fuser.fuse(dense, sparse, top_k=10)

    if r1 == r2:
        print(f"  Run 1 == Run 2 (both {len(r1)} items, identical order)")
        print("  ✅ PASS")
        return True
    else:
        print("  ❌ FAIL — non-deterministic output")
        return False


def test_config_defaults() -> bool:
    """Default k and alpha must match the architecture doc (k=25, alpha=0.4)."""
    print(f"\n{'='*70}")
    print("TEST 8: Config defaults match architecture doc")
    print(f"{'='*70}")

    fuser = RRFFuser()  # use defaults
    print(f"  fuser.k            = {fuser.k}  (doc says 25)")
    print(f"  fuser.alpha_dense  = {fuser.alpha_dense}  (doc says 0.4)")
    print(f"  fuser.alpha_sparse = {fuser.alpha_sparse:.2f}  (doc says 0.6)")

    # Also confirm these match the settings singleton
    print(f"  settings.rrf_k             = {settings.rrf_k}")
    print(f"  settings.rrf_alpha_dense   = {settings.rrf_alpha_dense}")

    if (
        fuser.k == 25
        and fuser.alpha_dense == 0.4
        and fuser.alpha_sparse == 0.6
        and settings.rrf_k == 25
        and settings.rrf_alpha_dense == 0.4
    ):
        print("  ✅ PASS — defaults match docs/RAG_PIPELINE_ARCHITECTURE.md Step 2")
        return True
    else:
        print("  ❌ FAIL — defaults do not match the architecture doc")
        return False


def test_invalid_params() -> bool:
    """Invalid k or alpha must raise ValueError."""
    print(f"\n{'='*70}")
    print("TEST 9: Invalid parameters raise ValueError")
    print(f"{'='*70}")

    all_pass = True
    try:
        RRFFuser(k=0, alpha_dense=0.4)
        print("  ❌ k=0 should have raised")
        all_pass = False
    except ValueError:
        print("  ✅ k=0 raises ValueError")

    try:
        RRFFuser(k=-5, alpha_dense=0.4)
        print("  ❌ k=-5 should have raised")
        all_pass = False
    except ValueError:
        print("  ✅ k=-5 raises ValueError")

    try:
        RRFFuser(k=25, alpha_dense=1.5)
        print("  ❌ alpha=1.5 should have raised")
        all_pass = False
    except ValueError:
        print("  ✅ alpha=1.5 raises ValueError")

    try:
        RRFFuser(k=25, alpha_dense=-0.1)
        print("  ❌ alpha=-0.1 should have raised")
        all_pass = False
    except ValueError:
        print("  ✅ alpha=-0.1 raises ValueError")

    return all_pass


def main() -> None:
    """Run all tests and report pass/fail summary."""
    print("--- RRFFuser Test Suite (pure math, no model, no DB) ---")

    tests = [
        test_hand_computed_scores,
        test_chunk_in_both_lists,
        test_chunk_only_in_dense,
        test_chunk_in_neither,
        test_top_k_truncation,
        test_empty_inputs,
        test_determinism,
        test_config_defaults,
        test_invalid_params,
    ]

    results = [test() for test in tests]

    print(f"\n{'='*70}")
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"✅ ALL {total} TESTS PASSED — RRFFuser is correct.")
    else:
        print(f"❌ {total - passed}/{total} TESTS FAILED — see output above.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
