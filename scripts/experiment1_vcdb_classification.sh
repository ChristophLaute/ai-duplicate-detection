#!/usr/bin/env bash
# reference for the first experiment. Not meant to be run as is, as it would need the videos to be downloaded
# idea is to give an idea over the pipeline
# Note:  inserting hyperparameters is manual
set -e
cd "$(dirname "$0")/../CompactSIamese"

# 1. Data preparation 
python data/precompute_fragments.py             # writes fragment_metadata.csv
python data/precompute_vcdb_segment_pairs.py    # writes positive_pairs.csv
python data/compute_negatives.py                # writes negative_pairs.csv (hard + easy)

# 2. Hyperparameter tuning 
python tune.py --n-trials 50 --study-name siamese_fbeta   # writes tune_results

# 3. Put hyperparameters into train before running if they were tuned
python train.py

# 4. Evaluation
for MODE in segment whole_video_descriptors whole_video_fragments; do
  python test.py --mode "$MODE"
done
for BACKEND in faiss chamfer faiss_chamfer pred_head; do
  python test.py --mode retrieval --retrieval_backend "$BACKEND"
done
