#!/usr/bin/env bash
# NUR V3 — Prepare Lightning AI upload package
#
# Creates a single zip with the repo code + downloaded raw data,
# ready to upload to Lightning AI studio.
#
# Usage:
#   bash scripts/v3/prepare_lightning_upload.sh
set -e

cd "$(dirname "$0")/../.."
echo "Working directory: $(pwd)"

# Sanity: check that local data download was done
if [ ! -d data/quran ] || [ -z "$(ls data/quran/*.json 2>/dev/null)" ]; then
    echo "❌ data/quran is empty. Run scripts/v3/01_download_quran.py first."
    exit 1
fi
if [ ! -d data/hadith/meetif ] || [ -z "$(ls data/hadith/meetif/*.json 2>/dev/null)" ]; then
    echo "❌ data/hadith/meetif is empty. Run scripts/v3/02_download_hadith.py first."
    exit 1
fi
if [ ! -d data/tafsir/v3 ] || [ -z "$(ls data/tafsir/v3/ibn_kathir_en/*.json 2>/dev/null)" ]; then
    echo "❌ data/tafsir/v3 is empty. Run scripts/v3/03_download_tafsirs.py first."
    exit 1
fi

PACKAGE_NAME="nur_v3_lightning_upload.zip"
echo ""
echo "Creating $PACKAGE_NAME ..."

# Clean any old package
rm -f "$PACKAGE_NAME"

# Build the zip with ONLY what Lightning AI needs:
#   - All Python source (scripts/, src/, docs/)
#   - data/quran/ (5 MB)
#   - data/hadith/meetif/ (63 MB)
#   - data/tafsir/v3/ (237 MB)
#   - data/processed/quran_v3.jsonl, hadith_v3.jsonl (built locally in step 5/6)
#   - pyproject.toml, requirements.txt, README.md
# Excluded:
#   - .venv/, __pycache__/
#   - data/chroma_db/ (V1/V2 legacy, 1.4 GB)
#   - data/sparse/ (V1/V2 legacy)
#   - data/tafsir/en_ibn_kathir.parquet (replaced by V3 upstream fetch)
#   - data/tafsir/ar/ and data/tafsir/en/ (V1 legacy folders, replaced by v3/)
#   - .git/

zip -r "$PACKAGE_NAME" \
    scripts/v3/ \
    src/nur/ \
    docs/v3/ \
    data/quran/ \
    data/hadith/meetif/ \
    data/tafsir/v3/ \
    data/processed/quran_v3.jsonl \
    data/processed/hadith_v3.jsonl \
    data/processed/_summary_v3.json \
    data/processed/_cross_refs_summary.json \
    pyproject.toml \
    requirements.txt \
    README.md \
    -x "*.pyc" "__pycache__/*" ".git/*" ".venv/*" 2>&1 | tail -10

SIZE_MB=$(du -h "$PACKAGE_NAME" | cut -f1)
echo ""
echo "✅ Package created: $PACKAGE_NAME ($SIZE_MB)"
echo ""
echo "Next steps:"
echo "  1. Upload $PACKAGE_NAME to your Lightning AI L40S studio"
echo "  2. Unzip: unzip $PACKAGE_NAME"
echo "  3. Run: pip install vllm FlagEmbedding chromadb pyarrow"
echo "  4. Run: python scripts/v3/04_generate_context_cards.py"
echo "  5. Run: python scripts/v3/05_build_chunks.py  # rebuilds with context cards"
echo "  6. Run: python scripts/v3/06_compute_cross_refs.py"
echo "  7. Run: python scripts/v3/07_embed_and_index.py"
echo "  8. Download nur_v3_indexed.zip back to local"
echo "  9. Unzip locally and run python scripts/v3/08_verify_pipeline.py"
