# Compact Siamese Network — Pipeline Overview

Reimplementation and retrieval extension of Fojcik et al.'s extremely compact
Siamese video representation, trained and validated on VCDB, then extended to
near-duplicate video retrieval on FIVR-200K.

## Pipeline flow

### 1. Data preparation (VCDB)
1. `data/precompute_fragments.py` — slices raw VCDB videos into 15-frame,
   32x32 fragments (`data/FramePrep.py:compute_fragments_streaming`), writes
   `fragment_metadata.csv`.
2. `data/precompute_vcdb_segment_pairs.py` — reads VCDB `annotation/*.txt` +
   `fragment_metadata.csv` to build `positive_pairs.csv`.
3. `data/compute_negatives.py` — reads `positive_pairs.csv`, mines hard
   negatives (same video, non-overlapping segment) and easy negatives
   (random other video), writes `negative_pairs.csv`.
4. `data/dataset_segment_pairs.py` (`VCDBSegmentPairDataset`) — wraps the
   resulting pair CSVs as a PyTorch `Dataset` for training/evaluation.

### 2. Training
- `train.py` — trains `model/pytorch_model.py:SiameseNetwork` on
  `VCDBSegmentPairDataset`, builds video-disjoint train/val/test splits,
  logs metrics via `model/metrics.py`.
- `tune.py` — thin Optuna wrapper around `train.py` (imports `train`,
  `_subset`, `cap_negatives`, `load_pairs` directly); results in
  `tune_results/optuna_study.db`.

### 3. VCDB classification evaluation (Experiment 1)
- `test.py --mode segment` — the fragment-pair classification entry point;
  writes `metrics.json` (accuracy/precision/recall/F1/F-beta). `test.py` is
  a broader multi-mode CLI (`segment`, `whole_video_fragments`,
  `whole_video_descriptors`, `retrieval`) covering both classification and
  VCDB-side retrieval.

### 4. Retrieval extension (FIVR)
1. `data/precompute_fivr_fragments.py` → `data/precompute_fivr_descriptors.py`
   — extracts descriptors for FIVR fragments via the trained
   `SiameseNetwork.encode`.
2. Scoring is dispatched by **`FIVR_evaluation.py`** (`compute_score()`),
   the current, complete evaluation entry point supporting all three
   scoring variants reported in the thesis:
   - `pred_max` → `retrieval/two_stage_retr_pred_head.py:whole_video_logit_from_descriptors(..., aggregation="max")`
   - `pred_chamfer` → same function, `aggregation="chamfer"` (the default)
   - `chamfer` → `retrieval/chamfer_retr.py:chamfer_sim` (raw frame
     descriptors, no prediction head)
   - FAISS-backed variants: `retrieval/pool_embeddings.py`,
     `retrieval/chamfer_retr_faiss.py` (two-stage FAISS + rerank).
3. `eval_siam_detect.py` is an **older, narrower variant** of the same
   idea (only `pred_max`-style scoring, hardcoded thresholds, no
   FAISS/chamfer options) — superseded by `FIVR_evaluation.py`. Kept for
   reference but not the one to read first.

### Baselines comparison
- `retrieval/evaluate_baselines_fivr.py` — imports an external ISCNet/VSC-style
  baseline stack; this is a comparison script, not part of the Siamese
  method itself.

### Shared infrastructure
- `runtime_config.py` — path resolution and `get_device()`, used by nearly
  every script above. Not "core method," but read this first if a script
  fails to find a path.

## Files to focus on for a code review

Read in this order:
1. `model/pytorch_model.py` — the architecture itself.
2. `data/precompute_vcdb_segment_pairs.py` + `data/compute_negatives.py` — how
   positive/negative pairs are actually defined (the hard/easy negative
   distinction discussed in the thesis).
3. `train.py` — training loop and loss.
4. `test.py` (`--mode segment`) — how VCDB classification metrics are produced.
5. `FIVR_evaluation.py` — the retrieval extension and all three scoring
   variants used in the ablation.
6. `retrieval/two_stage_retr_pred_head.py` — the Chamfer-over-prediction-head
   aggregation that is the default/main scoring method.

## Not necessary to review closely
- `eval_siam_detect.py` — superseded by `FIVR_evaluation.py`, kept for reference.
- `plot_hap_ddp_comparison.py` — standalone figure generator for the thesis, hardcodes results, not part of the pipeline.
- `tune.py` — same logic as `train.py`, just wrapped in an Optuna objective.
