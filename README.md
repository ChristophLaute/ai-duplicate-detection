# Thesis-Master-AI

The code for the final submission for the VU AI master thesis. It focuses on near-duplicate video detection and retrieval, comparing five methods (HAP-VR, LINE-VSC, VVS, TMK, and compact siamesse) on FIVR-200K subsets, plus anoriginal contribution: HAP-DPP, a diversity-aware frame-selection method that reduces HAP-VR's computational cost.

## Reproducing the Thesis's Experiments

This is a brief overview over the code accompanying the experiments guiding this thesis. 

### Experiment 1: Compact Siamese Fragment Classification on VCDB
This experiment focused on reproducing and validating the compact siamese algorithm on the VCDB dataset.

The full documentation can be found in [`CompactSIamese/README.md`](CompactSIamese/README.md).

### Experiment 2: FIVR-200K Benchmarking (Retrieval + Detection)

The five methods wereevaluated at both the FIVR-5K and FIVR-50K reference-set. The subset can be passed through a shared `--ds {5k,50k}` argument:

| Method | Retrieval (mAP / R@k) | Detection (accuracy/precision/recall/F1/F-beta) | Threshold tuning |
|---|---|---|---|
| HAP-VR | `baselines/Evaluations/hap_vr_fivr_eval.py` | `baselines/Evaluations/eval_hap_det.py` | `baselines/Evaluations/tune_threshold_hap.py` |
| VVS | `baselines/Evaluations/vvs_fivr_eval.py` | `baselines/Evaluations/eval_vvs_det.py` | `baselines/Evaluations/tune_threshold_vvs.py` |
| LINE-VSC | `baselines/Evaluations/line_vsc_fivr_eval.py` | `baselines/Evaluations/eval_line_det.py` | `baselines/Evaluations/tune_threshold_line.py` |
| TMK | `baselines/Evaluations/tmk_fivr_eval.py` (hashes first via `baselines/TMK/extract_hashes_fivr.py`) | `baselines/Evaluations/eval_tmk_det.py` | `baselines/Evaluations/tune_threshold_tmk.py` |
| Compact Siamese | `CompactSIamese/FIVR_evaluation.py` | same script | see `CompactSIamese/README.md` |

In order to extend to detection thresholds have to be calculated over the similarities extracted by the methods. 
The thresholds for detection were tuned once per method on held-out validation
queries against the 50K reference pool. If it is needed to be rerun run
the `tune_threshold_*.py` script before the corresponding `eval_*_det.py`.

### Experiment 3: Ablations
In order to investigate the behavior of the main methods ablation experiments were conducted. 
**Compact Siamese** (scoring function + preprocessing variants): documented
in [`CompactSIamese/README.md`](CompactSIamese/README.md).

**LINE-VSC** (component ablation): Both `baselines/Evaluations/line_vsc_fivr_eval.py`
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