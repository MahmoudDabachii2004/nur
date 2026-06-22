#!/usr/bin/env bash
# NUR V3 — Orchestrator
# Runs the 8-step V3 build pipeline.
#
# Steps 1, 2, 3, 5, 6, 8 → run locally
# Steps 4, 7            → run on Lightning AI L40S (separately)
#
# Usage:
#   ./scripts/v3/run_all.sh           # run all local steps
#   ./scripts/v3/run_all.sh --from 5  # start from step 5
set -e

cd "$(dirname "$0")/../.."
echo "Working directory: $(pwd)"
echo ""

# Color helpers
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

step() {
    echo -e "\n${GREEN}=== STEP $1: $2 ===${NC}\n"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

fail() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

START_FROM=${2:-1}

if [ "$START_FROM" -le 1 ]; then
    step 1 "Download Quran (alquran.cloud)"
    python scripts/v3/01_download_quran.py || fail "Step 1 failed"
fi

if [ "$START_FROM" -le 2 ]; then
    step 2 "Download Hadith (meetif via fawazahmed0)"
    python scripts/v3/02_download_hadith.py || fail "Step 2 failed"
fi

if [ "$START_FROM" -le 3 ]; then
    step 3 "Download Tafsirs (spa5k upstream, 4 editions)"
    python scripts/v3/03_download_tafsirs.py || fail "Step 3 failed"
fi

echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  MANUAL STEP NEEDED — LIGHTNING AI L40S${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}${NC}"
echo -e "${YELLOW}  Step 4 (Context Cards) and Step 7 (Embed+Index) need${NC}"
echo -e "${YELLOW}  an L40S GPU. Upload the repo to Lightning AI and run:${NC}"
echo -e "${YELLOW}${NC}"
echo -e "${YELLOW}    pip install vllm FlagEmbedding chromadb pyarrow${NC}"
echo -e "${YELLOW}    python scripts/v3/04_generate_context_cards.py${NC}"
echo -e "${YELLOW}    python scripts/v3/05_build_chunks.py${NC}"
echo -e "${YELLOW}    python scripts/v3/06_compute_cross_refs.py${NC}"
echo -e "${YELLOW}    python scripts/v3/07_embed_and_index.py${NC}"
echo -e "${YELLOW}${NC}"
echo -e "${YELLOW}  Then download nur_v3_indexed.zip back to local.${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [ "$START_FROM" -le 5 ]; then
    step 5 "Build V3 chunks (3-layer Quran + 2-layer Hadith)"
    python scripts/v3/05_build_chunks.py || fail "Step 5 failed"
fi

if [ "$START_FROM" -le 6 ]; then
    step 6 "Compute Quran → Hadith cross-references"
    python scripts/v3/06_compute_cross_refs.py || fail "Step 6 failed"
fi

if [ "$START_FROM" -le 8 ]; then
    step 8 "Verify pipeline (5 ground-truth examples)"
    python scripts/v3/08_verify_pipeline.py || warn "Step 8 (verification) reported issues"
fi

echo ""
echo -e "${GREEN}✅ V3 LOCAL PIPELINE COMPLETE${NC}"
echo ""
