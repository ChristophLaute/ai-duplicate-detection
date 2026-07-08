#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash run_fivr_batch.sh <batch_ids.txt> [model_path]"
    exit 1
fi

BATCH_IDS="$1"
MODEL_PATH="${2:-best_siamese_model.pth}"

FRAGMENTS_DIR="${FRAGMENTS_DIR:-/tmp/fivr_batch_fragments}"
DESCRIPTORS_DIR="${DESCRIPTORS_DIR:-$PWD/descriptors}"
VIDEO_DIR="${VIDEO_DIR:-}"

FRAGMENT_ARGS=(
    --ids-file "$BATCH_IDS"
    --save-dir "$FRAGMENTS_DIR"
)

if [ -n "$VIDEO_DIR" ]; then
    FRAGMENT_ARGS+=(--video-dir "$VIDEO_DIR")
fi

python data/precompute_fivr_fragments.py "${FRAGMENT_ARGS[@]}"

python retrieval/compute_fivr_descriptors.py \
    --model "$MODEL_PATH" \
    --ids-file "$BATCH_IDS" \
    --fragments-dir "$FRAGMENTS_DIR" \
    --descriptors-dir "$DESCRIPTORS_DIR" \
    --delete-fragments
