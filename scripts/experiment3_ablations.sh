#!/usr/bin/env bash
# reference script for running the ablations not meant to be run directly but as a reference 
#to give an idea over the pipeline
set -e

# compact siamese ablations
cd "$(dirname "$0")/../CompactSIamese"

# ablation of scoring/aggregation function 
python FIVR_evaluation.py --ds 5k --scoring pred_max      # main method / thesis table
python FIVR_evaluation.py --ds 5k --scoring pred_chamfer
python FIVR_evaluation.py --ds 5k --scoring chamfer

# ablation of fragment length (T=10)
python data/precompute_fragments.py --len-frames 10 --target-size 32 32
python data/precompute_vcdb_segment_pairs.py
python data/compute_negatives.py
python train.py --T 10
python data/precompute_fivr_fragments.py --len-frames 10 --target-size 32 32
python data/precompute_fivr_descriptors.py
python FIVR_evaluation.py --ds 5k --T 10


# ablation of resolution (56x56)
python data/precompute_fragments.py --len-frames 15 --target-size 56 56
python data/precompute_vcdb_segment_pairs.py
python data/compute_negatives.py
python train.py --T 15
python data/precompute_fivr_fragments.py --len-frames 15 --target-size 56 56
python data/precompute_fivr_descriptors.py
python FIVR_evaluation.py --ds 5k


# LINE-VSC ablations of post-processing steps
cd ../baselines/Evaluations

python line_vsc_fivr_eval.py --ds fivr5k    # full model / thesis table
python line_vsc_fivr_eval.py --ds fivr5k --no_tta
python line_vsc_fivr_eval.py --ds fivr5k --aggregation_type avg

# tune thresholds and evaluate for each ablation of the descriptor generation steps
for FLAG in no-consistency-weight no-temporal-concat no-score-norm; do
  python ../line_vsc/FIVR_exp/postprocess_line_desc.py --$FLAG
  python tune_threshold_line.py --ds fivr50k --${FLAG//-/_}
  python line_vsc_fivr_eval.py --ds fivr5k --${FLAG//-/_}
done
