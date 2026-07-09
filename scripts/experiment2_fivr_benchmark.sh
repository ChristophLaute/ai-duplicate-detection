#!/usr/bin/env bash
# Reference for the second experiment. Not meant to be run as is, as it would
# need weights and descriptors/hashes of the videos to be extracted.
# in this example we do everything for the 50k dataset.

# Note: thresholds have to be inserted manually
set -e
cd "$(dirname "$0")/../baselines/Evaluations"

# HAP-VR
python ../HAP-VR/extract_fivr_subset.py --video-dir <path> --save-path <path> # precompute features
for MODE in all nd; do
  python tune_threshold_hap.py --ds 50k --mode "$MODE" # threshold tuning for detection
done
python hap_vr_fivr_eval.py --ds 50k # retrieval
for MODE in all nd; do
  python eval_hap_det.py --ds 50k --mode "$MODE" # detection
done

# VVS
python ../VVS/extract_fivr_subset.py --video-dir <path> --save-path <path> # precompute LN-iMAC features
for MODE in all nd; do
  python tune_threshold_vvs.py --ds fivr50k --mode "$MODE" # threshold tuning for detection
done
python vvs_fivr_eval.py --ds fivr50k  #retrieval
for MODE in all nd; do
  python eval_vvs_det.py --ds fivr50k --mode "$MODE" #detection
done

# LINE-VSC
python ../line_vsc/FIVR_exp/precompute_line_desc.py # precompute descriptors
for MODE in all nd; do
  python tune_threshold_line.py --ds fivr50k --mode "$MODE" # threshold tuning for detection
done
python line_vsc_fivr_eval.py --ds fivr50k # retrieval
for MODE in all nd; do
  python eval_line_det.py --ds fivr50k --mode "$MODE" # detection
done

# TMK
python ../TMK/extract_hashes_fivr.py # extract hashes for retrieval and detection
for MODE in all nd; do
  python tune_threshold_tmk.py --ds fivr50k --mode "$MODE" # threshold tuning for detection
done

python tmk_fivr_eval.py --ds fivr50k # retrieval
for MODE in all nd; do
  python eval_tmk_det.py --ds fivr50k --mode "$MODE" # detection
done

# compact siamese
cd ../../CompactSIamese # change to compact siamese directory
python data/precompute_fivr_descriptors.py  # encodes fragments via the trained model (computes fragments internally)
for MODE in all nd; do
  python tune_threshold_fivr.py --ds 50k --mode "$MODE" # threshold tuning for detection
done

python FIVR_evaluation.py --ds 50k          # retrieval
for MODE in all nd; do
  python eval_siam_detect.py --ds 50k --mode "$MODE" # detection
done
