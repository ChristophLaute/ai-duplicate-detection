# Thesis-Master-AI

The code for the final submission for the VU AI master thesis.

## Reproducing the Thesis's Experiments

This maps each of the thesis's four experiments to the scripts that produce
its results. All original SLURM cluster job scripts have been omitted from
this public release (code only) — the entry points below are the plain
Python scripts those jobs called.

### Experiment 1: Compact Siamese Fragment Classification on VCDB

Fully documented in [`CompactSIamese/README.md`](CompactSIamese/README.md).

### Experiment 2: FIVR-200K Benchmarking (Retrieval + Detection)

Five methods, evaluated at both the FIVR-5K and FIVR-50K reference-set
scales via a shared `--ds {5k,50k}` argument:

| Method | Retrieval (mAP / R@k) | Detection (accuracy/precision/recall/F1/F-beta) | Threshold tuning |
|---|---|---|---|
| HAP-VR | `baselines/Evaluations/hap_vr_fivr_eval.py` | `baselines/Evaluations/eval_hap_det.py` | `baselines/Evaluations/tune_threshold_hap.py` |
| VVS | `baselines/Evaluations/vvs_fivr_eval.py` | `baselines/Evaluations/eval_vvs_det.py` | `baselines/Evaluations/tune_threshold_vvs.py` |
| LINE-VSC | `baselines/Evaluations/line_vsc_fivr_eval.py` | `baselines/Evaluations/eval_line_det.py` | `baselines/Evaluations/tune_threshold_line.py` |
| TMK | `baselines/Evaluations/tmk_fivr_eval.py` (hashes first via `baselines/TMK/extract_hashes_fivr.py`) | `baselines/Evaluations/eval_tmk_det.py` | `baselines/Evaluations/tune_threshold_tmk.py` |
| Compact Siamese | `CompactSIamese/FIVR_evaluation.py` | same script | see `CompactSIamese/README.md` |

Detection thresholds are tuned once per method on held-out validation
queries against the 50K reference pool, then applied at both scales — run
the `tune_threshold_*.py` script before the corresponding `eval_*_det.py`.

### Experiment 3: Ablations

**Compact Siamese** (scoring function + preprocessing variants): documented
in [`CompactSIamese/README.md`](CompactSIamese/README.md).

**LINE-VSC** (component ablation): a single script with toggle flags, not
separate scripts per component. Both `baselines/Evaluations/line_vsc_fivr_eval.py`
and `baselines/Evaluations/eval_line_det.py` accept:
- `--no_consistency_weight`
- `--no_temporal_concat`
- `--no_score_norm`
- `--no_tta`
- `--aggregation_type {max,avg}` (`avg` = the "mean aggregation" ablation)

Each flag disables one pipeline component relative to the full
configuration reported as the baseline.

### Experiment 4: HAP-DPP Evaluation

Fully documented in
[`baselines/Frame_Selection_Hap/README.md`](baselines/Frame_Selection_Hap/README.md).
Run `baselines/Frame_Selection_Hap/hap_ddp_sweep.py` to reproduce the full
uniform-vs-interframe-diff-vs-DPP comparison.