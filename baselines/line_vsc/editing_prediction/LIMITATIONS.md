# Editing Prediction Model — Limitations & Deviations from Original

## 1. No `vstack_hstack` combined class

**Original model:** The pretrained `copy_type_pred__convnext_clip.ckpt` from LINE Corporation
was trained on internal data that included a combined `vstack_hstack` augmentation — a video
with content arranged in a 2×2 grid (both horizontal and vertical stacking simultaneously).
This triggered `TTA4ViewsTransform` (4-view TTA) during inference.

**Retrained model:** The custom training data (generated via `video_generation_AugLy.py`)
applies `HStack` and `VStack` as separate, mutually exclusive augmentations per video.
No combined `vstack_hstack` augmentation is generated, so the retrained model never learned
to predict it as a class.

**Impact:** `TTA4ViewsTransform` (4-view TTA) is never selected during inference. Videos that
happen to have both horizontal and vertical stacking will be handled by `hstack` or `vstack`
TTA instead, which is less optimal.

**Workaround applied:** `inference_impl.py` line 343 uses `.get("vstack_hstack", zeros)`
so a missing key defaults to 0 instead of raising a KeyError.

---

## 2. `rotate` instead of `rotate90_270` class

**Original model:** Had a combined `rotate90_270` class.

**Retrained model:** AugLy records all rotations as `"rotate"` regardless of degrees,
so `prepare.py`'s `AUG_LABEL_MAPPING` for `"rotate90"` → `"rotate90_270"` never fires.
The training data has a single `"rotate"` class (label 21) covering all rotation angles.

**Impact:** None — `inference_impl.py` does not use rotation predictions for TTA selection
at all. Rotation classes are purely informational and do not affect which TTA transform is applied.

---

## 3. No `InsertInBackground` class

**Original:** The first version of `video_generation_AugLy.py` included `InsertInBackground`.

**Retrained:** Removed from augmentation list (commit `a88dde6`) because it caused errors.

**Impact:** The model cannot detect `InsertInBackground` edits and will not apply appropriate
TTA for them. Minor impact since this augmentation type is rare in practice.

---

## 4. Training data size

**Original model:** Trained on LINE Corporation's internal dataset of unknown size.

**Retrained model:** Trained on 5,000 AugLy-generated samples from 10K VSC22 reference videos.
This is likely significantly smaller than LINE's internal training set.

**Impact:** The retrained model may be less accurate at predicting augmentation types,
leading to suboptimal TTA selection and lower retrieval performance compared to the original.

---

## 5. Training infrastructure

**Original model:** Unknown training setup.

**Retrained model:** Trained for 8 epochs on 1× NVIDIA RTX A4000 (16GB) on DAS-6.
Checkpoint used: `last.ckpt` (epoch 4, step 3120).
