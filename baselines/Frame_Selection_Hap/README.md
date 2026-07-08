# HAP-DPP: Diversity-Aware Frame Selection for HAP-VR

Our extension to HAP-VR that reduces its online computational cost by
selecting a subset of frame descriptors before similarity computation,
instead of comparing every extracted frame. Compares three strategies:
uniform subsampling, interframe-difference peak selection, and a
DPP-inspired greedy diversity selection (our contribution).

## Files

- `dpp_frame_selection.py` — library only, no CLI. Implements all three
  selection strategies: `dpp_frame_selection` (ours), `uniform_frame_selection`,
  `select_frames_by_interframe_diff` (+ `detect_maxima_interframe_diff`,
  `desc_interframe_diff`, `relevance_fun` helpers).
- `HAP_DDP_retr.py` — retrieval-only evaluation (mAP/R@k) for a single
  frame-selection method. Args: `--ds` (5k/50k), `--method
  {dpp,interframe_diff,uniform}`, `--max_k_divider` (frame budget, default 2
  = 50%), `--energy_threshold`, `--gpu_id`.
- `HAP_DDP_det.py` — detection metrics (accuracy/precision/recall/F1/F-beta)
  for a single method+config, same core args plus `--mode {all,nd,ds}`.
- `tune_threshold_hap_ddp.py` — the DPP-aware threshold tuner. Distinct from
  `baselines/Evaluations/tune_threshold_hap.py` (the plain HAP-VR tuner) —
  this one applies the selected frame-selection method *before* tuning the
  detection threshold, since the threshold has to be re-tuned per
  frame-selection configuration.
- **`hap_ddp_sweep.py`** — the actual comparison driver that produced the
  thesis's results. Loops over `--methods` (default `dpp,interframe_diff,uniform`)
  × `--kdividers` (default `2,4,8,16`, i.e. frame budgets from 50% down to
  ~6%), tunes thresholds on `--tune_ds` (default 50k), evaluates retrieval +
  detection on `--ds` (default 5k), and writes
  `results_sweep_<method>_k<k>_<ds>.json` plus a combined
  `results_sweep_all_<ds>.json`. **Run this one** to reproduce the full
  uniform-vs-interframe-diff-vs-DPP comparison table.

## How it plugs into HAP-VR

These scripts insert `baselines/HAP-VR/` and `CompactSIamese/` onto
`sys.path` and reuse HAP-VR's own `SimilarityNetwork`,
`evaluation.calculate_similarities_to_queries`, and `HDF5DatasetGenerator`
directly against precomputed HAP features
(`FIVR_DIR/HAP_Features/hap_features_all.h5`) and HAP-VR's checkpoint
(`HAP-VR/ckpt/model.pth`). Frame selection is applied to the frame-level
descriptors *before* `index_video` — i.e., this doesn't modify HAP-VR's
model at all, only which of its already-computed descriptors get compared
at query time.
