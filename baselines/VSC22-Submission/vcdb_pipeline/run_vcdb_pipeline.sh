#!/usr/bin/env bash
# =============================================================================
# VCDB → VSC pipeline: prepares VCDB for any VSC solution
#
# Usage:
#   bash run_vcdb_pipeline.sh
#
# What it does:
#   1. prepare_vcdb_for_VSC.py  — build metadata CSVs + video symlinks
#   2. preprocess_vcdb.py       — extract jpg frames, build LMDB
#   3. Symlink data/ in each solution directory to the vsc_adapter
#
# After this script, each solution can be run using its original commands.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON=/Users/christophlaute/anaconda3/envs/vcsl/bin/python3

# ---------------------------------------------------------------------------
# CONFIG — edit these paths for your machine
# ---------------------------------------------------------------------------
VCDB_DIR="/Volumes/LaCie/DuplicateDetectionThesis/VCDB"
ADAPTER_DIR="${VCDB_DIR}/vsc_adapter"          # output of prepare_vcdb_for_VSC.py

VSC22_DIR="${REPO_ROOT}/VSC22-Submission/VSC22-Descriptor-Track-1st"
LINE_VSC_DIR="${REPO_ROOT}/external/line_vsc/meta-vsc-descriptor-runtime"

WORKERS=8      # parallel ffmpeg workers for frame extraction
FPS=1          # frames per second to extract
# ---------------------------------------------------------------------------

echo "=========================================="
echo " VCDB → VSC pipeline"
echo "=========================================="
echo " VCDB_DIR    : ${VCDB_DIR}"
echo " ADAPTER_DIR : ${ADAPTER_DIR}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Step 1: Build metadata + video symlinks
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Building metadata and video symlinks..."
${PYTHON} "${SCRIPT_DIR}/prepare_vcdb_for_VSC.py"

# ---------------------------------------------------------------------------
# Step 2: Build jpg zips + LMDB (needed for VSC22 training)
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Preprocessing videos (jpg zips + LMDB)..."
echo "      This is the slow step — roughly 1-2s per video."
${PYTHON} "${SCRIPT_DIR}/preprocess_vcdb.py" \
    --data_dir "${ADAPTER_DIR}" \
    --fps "${FPS}" \
    --workers "${WORKERS}"

# ---------------------------------------------------------------------------
# Step 3: Symlink data/ in each solution to the adapter directory
#
# This is the key that makes any VSC solution "just work" — each solution
# references its data via a local data/ directory, and we point that at the
# same shared adapter.  Swap the symlink to switch between VCDB and VSC data.
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Linking adapter into solution directories..."

link_data() {
    local solution_dir="$1"
    local data_link="${solution_dir}/data"

    if [ -L "${data_link}" ]; then
        echo "  [update] ${data_link} -> ${ADAPTER_DIR}"
        rm "${data_link}"
    elif [ -d "${data_link}" ] && [ ! -L "${data_link}" ]; then
        echo "  [rename] ${data_link} -> ${data_link}_vsc (saving original VSC data)"
        mv "${data_link}" "${data_link}_vsc"
    fi
    ln -s "${ADAPTER_DIR}" "${data_link}"
    echo "  [ok]     ${data_link} -> ${ADAPTER_DIR}"
}

link_data "${VSC22_DIR}"
link_data "${LINE_VSC_DIR}/submission_src"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Pipeline complete."
echo ""
echo " To run VSC22 1st place training (model v106):"
echo "   cp ${SCRIPT_DIR}/configs/config_vcdb_v106.py ${VSC22_DIR}/train/train_v106/config_v106.py"
echo "   cd ${VSC22_DIR}/train/train_v106 && bash train_v106.sh"
echo ""
echo " To run LINE VSC inference:"
echo "   cd ${LINE_VSC_DIR}/submission_src"
echo "   python -m src.inference_full \\"
echo "     --accelerator cuda --processes 1 \\"
echo "     --dataset_paths ${ADAPTER_DIR}/videos/train/query ${ADAPTER_DIR}/videos/train/reference \\"
echo "     --gt_path ${ADAPTER_DIR}/meta/train/train_matching_ground_truth.csv \\"
echo "     --output_path ./outputs"
echo ""
echo " To restore original VSC data:"
echo "   rm ${VSC22_DIR}/data && mv ${VSC22_DIR}/data_vsc ${VSC22_DIR}/data"
echo "=========================================="
