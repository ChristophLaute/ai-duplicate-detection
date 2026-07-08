#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
if [ -z "${VCDB_DIR:-}" ]; then
  export VCDB_DIR="$($PYTHON -c 'from runtime_config import VCDB_DIR; print(VCDB_DIR)')"
fi

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
SKIP_PAIRS=0
SKIP_FRAGMENTS=0

for arg in "$@"; do
    case $arg in
        --skip-pairs)     SKIP_PAIRS=1 ;;
        --skip-fragments) SKIP_FRAGMENTS=1 ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--skip-pairs] [--skip-fragments]"
            echo ""
            echo "  --skip-pairs      Skip recomputing positive/negative pairs and fragments."
            echo "                    Use after hyperparameter tuning: only retrains the model"
            echo "                    and recomputes descriptors/embeddings/evaluation."
            echo "  --skip-fragments  Skip fragment precomputation only (pairs are still rebuilt)."
            exit 1
            ;;
    esac
done

echo "Using VCDB_DIR=$VCDB_DIR"

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
echo "Cleaning generated experiment artifacts..."

if [ "$SKIP_PAIRS" -eq 0 ]; then
    rm -rf "$VCDB_DIR/positive_pairs"
    rm -rf "$VCDB_DIR/negative_pairs"
fi

if [ "$SKIP_FRAGMENTS" -eq 0 ] && [ "$SKIP_PAIRS" -eq 0 ]; then
    rm -rf "$VCDB_DIR/fragments"
fi

rm -rf "$VCDB_DIR/descriptors"
rm -rf "$VCDB_DIR/pooled_embeddings"

rm -f train_pairs_segment.csv
rm -f val_pairs_segment.csv
rm -f test_pairs_segment.csv
rm -f test_pairs.csv
rm -f train_pairs_video_level.csv
rm -f val_pairs_video_level.csv
rm -f test_pairs_video_level.csv
rm -rf evaluation_results

# ---------------------------------------------------------------------------
# Pairs + fragments (skipped with --skip-pairs)
# ---------------------------------------------------------------------------
if [ "$SKIP_PAIRS" -eq 0 ]; then
    if [ "$SKIP_FRAGMENTS" -eq 0 ]; then
        echo "Precomputing fragments..."
        $PYTHON data/precompute_fragments.py
    else
        echo "Skipping fragment precomputation (--skip-fragments)."
    fi

    echo "Rebuilding positive pairs..."
    $PYTHON data/precompute_vcdb_segment_pairs.py

    echo "Rebuilding negative pairs..."
    $PYTHON data/compute_negatives.py
else
    echo "Skipping pairs + fragments (--skip-pairs)."
fi

# ---------------------------------------------------------------------------
# Training + evaluation (always runs)
# ---------------------------------------------------------------------------
echo "Training model and regenerating clean splits..."
$PYTHON train.py

echo "Computing descriptors..."
$PYTHON retrieval/compute_descriptors.py

echo "Pooling embeddings and building FAISS index..."
$PYTHON retrieval/pool_embeddings.py

echo "Running segment-level evaluation..."
$PYTHON test.py --mode segment

echo "Running whole-video descriptor evaluation..."
$PYTHON test.py --mode whole_video_descriptors --test_csv test_pairs_video_level.csv

echo "Running FAISS retrieval evaluation..."
$PYTHON test.py --mode retrieval --test_csv test_pairs_video_level.csv --retrieval_backend faiss

echo "Done."
