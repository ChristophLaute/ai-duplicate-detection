#!/usr/bin/env bash
# Run the full FIVR evaluation pipeline for both the Siamese model and the
# LINE VSC baseline, then generate comparison plots.
#
# Usage:
#   cd CompactSIamese/
#   bash run_fivr_comparison.sh
#
# Set SKIP_EXTRACTION=1 to skip descriptor extraction if already done.
# Set SKIP_BUILD=1 to skip FAISS index build if already done.

set -e
cd "$(dirname "$0")"

SKIP_EXTRACTION=${SKIP_EXTRACTION:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
PLOTS_DIR="plots/fivr_comparison"

echo "========================================"
echo " Step 1: Precompute FIVR fragments"
echo "========================================"
python data/precompute_fivr_fragments.py

echo "========================================"
echo " Step 2: Compute Siamese descriptors"
echo "========================================"
python retrieval/compute_fivr_descriptors.py --model best_siamese_model.pth

echo "========================================"
echo " Step 3: Evaluate Siamese on FIVR"
echo "========================================"
BUILD_FLAG=""
if [ "$SKIP_BUILD" = "0" ]; then
    BUILD_FLAG="--build"
fi
python retrieval/evaluate_fivr.py $BUILD_FLAG \
    --output fivr_siamese_results.json \
    --task all

echo "========================================"
echo " Step 4: Evaluate LINE VSC baseline on FIVR"
echo "========================================"
SKIP_FLAG=""
if [ "$SKIP_EXTRACTION" = "1" ]; then
    SKIP_FLAG="--skip-extraction"
fi
python retrieval/evaluate_baselines_fivr.py $SKIP_FLAG

echo "========================================"
echo " Step 5: Generate comparison plots"
echo "========================================"
# Find the two most recent result dirs
SIAMESE_DIR=$(ls -td evaluation_results/fivr_* 2>/dev/null | head -1)
VSC_DIR=$(ls -td evaluation_results/vsc_baseline_fivr_* 2>/dev/null | head -1)

if [ -z "$SIAMESE_DIR" ] || [ -z "$VSC_DIR" ]; then
    echo "Could not find output dirs, skipping plots."
    echo "Run plot_results.py manually."
else
    echo "Siamese results: $SIAMESE_DIR"
    echo "VSC baseline:    $VSC_DIR"
    python plot_results.py \
        --dir "$SIAMESE_DIR" "$VSC_DIR" \
        --compare \
        --save-dir "$PLOTS_DIR"
    echo "Plots saved to $PLOTS_DIR"
fi

echo "Done."
