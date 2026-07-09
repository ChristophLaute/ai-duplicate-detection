# Thesis-Master-AI

The code for the final submission for the VU AI master thesis. It focuses on near-duplicate video detection and retrieval, comparing five methods (HAP-VR, LINE-VSC, VVS, TMK, and compact siamesse) on FIVR-200K subsets, plus an original contribution: HAP-DPP, a diversity-aware frame-selection method that reduces HAP-VR's computational cost.

## Reproducing the Thesis's Experiments

This is a brief overview over the code accompanying the experiments guiding this thesis. 

### Experiment 1: Compact Siamese Fragment Classification on VCDB
This experiment focused on reproducing and validating the compact siamese algorithm on the VCDB dataset.

The full documentation can be found in [`CompactSIamese/README.md`](CompactSIamese/README.md).

### Experiment 2: FIVR-200K Benchmarking (Retrieval + Detection)

The five methods wereevaluated at both the FIVR-5K and FIVR-50K reference-set. The subset can be passed through a shared `--ds {5k,50k}` argument:

| Method | Precompute features/descriptors | Retrieval (mAP / R@k) | Detection (accuracy/precision/recall/F1/F-beta) | Threshold tuning |
|---|---|---|---|---|
| HAP-VR | `baselines/HAP-VR/extract_fivr_subset.py` | `baselines/Evaluations/hap_vr_fivr_eval.py` | `baselines/Evaluations/eval_hap_det.py` | `baselines/Evaluations/tune_threshold_hap.py` |
| VVS | `baselines/VVS/extract_fivr_subset.py` | `baselines/Evaluations/vvs_fivr_eval.py` | `baselines/Evaluations/eval_vvs_det.py` | `baselines/Evaluations/tune_threshold_vvs.py` |
| LINE-VSC | `baselines/line_vsc/FIVR_exp/precompute_line_desc.py` | `baselines/Evaluations/line_vsc_fivr_eval.py` | `baselines/Evaluations/eval_line_det.py` | `baselines/Evaluations/tune_threshold_line.py` |
| TMK | `baselines/TMK/extract_hashes_fivr.py` | `baselines/Evaluations/tmk_fivr_eval.py` | `baselines/Evaluations/eval_tmk_det.py` | `baselines/Evaluations/tune_threshold_tmk.py` |
| Compact Siamese | `CompactSIamese/data/precompute_fivr_descriptors.py` (computes fragments internally) | `CompactSIamese/FIVR_evaluation.py` | `CompactSIamese/eval_siam_detect.py` | `CompactSIamese/tune_threshold_fivr.py` |

Each method's detection script computes its threshold on held-out
validation queries against the 50K reference pool, but the thresholds are
hardcoded in each `eval_*_det.py`. 

### Experiment 3: Ablations
In order to investigate the behavior of the main methods ablation experiments were conducted. 
**Compact Siamese** (scoring function + preprocessing variants): documented
in [`CompactSIamese/README.md`](CompactSIamese/README.md).

**LINE-VSC** (postprocessing step ablation): Both `baselines/Evaluations/line_vsc_fivr_eval.py`
and `baselines/Evaluations/eval_line_det.py` accept:
- `--no_consistency_weight`
- `--no_temporal_concat`
- `--no_score_norm`
- `--no_tta`
- `--aggregation_type {max,avg}` 

Each flag disables one pipeline component relative to the full
configuration reported as the baseline.

### Experiment 4: HAP-DPP Evaluation

Fully documented in
[`baselines/Frame_Selection_Hap/README.md`](baselines/Frame_Selection_Hap/README.md).


### Scripts
For all experiments, shell scripts were written in [`scripts/`](scripts/) that are meant to demonstrate how the experiments could be replicated. Note: they will not work without videos and model weights. The idea is more to give an idea over the steps that are needed. 