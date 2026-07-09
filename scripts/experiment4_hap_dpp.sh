#!/usr/bin/env bash
# reference script for running the HAP-DPP experiment, would need model weights etc.
# not meant to be run directly but as a reference to give an idea over the pipeline

set -e
cd "$(dirname "$0")/../baselines/Frame_Selection_Hap"

# 1. Tune thresholds 
for METHOD in dpp interframe_diff; do
  for MODE in all nd; do
    python tune_threshold_hap_ddp.py --method "$METHOD" --mode "$MODE" --max_k_divider 2 --ds 50k
  done
done

# 2.update hardcoded thresholds

# 3. Detection + retrieval for all three methods
for DS in 5k 50k; do
  for METHOD in dpp uniform interframe_diff; do
    for MODE in all nd; do
      python HAP_DDP_det.py --ds "$DS" --method "$METHOD" --mode "$MODE" --max_k_divider 2
    done
  done
done
for DS in 5k 50k; do
  for METHOD in dpp uniform interframe_diff; do
    python HAP_DDP_retr.py --ds "$DS" --method "$METHOD" --max_k_divider 2
  done
done


