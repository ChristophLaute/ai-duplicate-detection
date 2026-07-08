# Changes to `video_generation_AugLy.py` — LINE VSC editing-prediction training

## Summary

Three bugs prevented the editing-prediction model from being trained with all
the augmentation classes the original LINE VSC paper used.  The file was also
non-functional on DAS-6 because it relied on `decord`.  All four issues are
fixed in commit `8defd6f`.

---

## 1. `decord.VideoReader` replaced with `ffmpeg` subprocess assembly

**What changed:** Removed `import decord`, `decord_clip_reader()`, and
`torchvision.io.write_video()`.  Added `ffmpeg_assemble_video()` which uses
`subprocess.run(["ffmpeg", ...])` to (a) extract the reference clip, (b) split
the query video at the insert point, and (c) concatenate all parts with
`ffmpeg -f concat`.

**Why:** `decord.VideoReader` hangs indefinitely on DAS-6 cluster nodes.  The
root cause is entropy starvation: `decord` opens `/dev/random` internally and
blocks forever waiting for entropy that the compute nodes never provide.
Replacing it with `ffmpeg` (which does not use `/dev/random`) avoids the hang
completely.

---

## 2. Added `RotateWithLabel` wrapper class — fixes `rotate90_270` and `rotate180_vflip` labels

**What changed:** New class `RotateWithLabel(degrees, label)` wraps
`augly.video.Rotate` but appends a custom metadata entry instead of letting
AugLy write its own.  Three instances are added to `augly_method`:

```
RotateWithLabel(90,  "rotate90")
RotateWithLabel(270, "rotate270")
RotateWithLabel(180, "rotate180")
```

**Why:** AugLy's `Rotate` always records `{"name": "rotate"}` in its metadata
regardless of the angle supplied (confirmed in AugLy `functional.py` — only
one code path, no angle-specific sub_name).  The downstream `prepare.py` maps
augmentation names to training labels via `AUG_LABEL_MAPPING`:

```python
AUG_LABEL_MAPPING = {
    "rotate90":  "rotate90_270",
    "rotate270": "rotate90_270",
    "rotate180": "rotate180_vflip",
    "vflip":     "rotate180_vflip",
    ...
}
```

Because AugLy always wrote `"rotate"` (not `"rotate90"` / `"rotate270"` /
`"rotate180"`), these mapping keys never matched, so the classes `rotate90_270`
and `rotate180_vflip` were never generated in the training dataset.
`RotateWithLabel` fixes this by writing the correct name directly.

---

## 3. Added `VStackHStack` custom ffmpeg class — restores the 2×2 tile augmentation

**What changed:** New class `VStackHStack(video2, video3, video4, width, height)`
tiles four videos in a 2×2 grid using `ffmpeg -filter_complex` with `hstack` +
`vstack`.  An instance is added to `augly_method`:

```
VStackHStack(str(video_in2), str(video_in3), str(video_in4), q_w, q_h)
```

Two new random video picks `video_in3` and `video_in4` are added to `main()`.

**Why:** `augly.video` has no `VStackHStack` — importing it raises
`ImportError`.  The LINE VSC paper uses this augmentation as a training class
(`vstack_hstack`), which drives the `TTA4ViewsTransform` in inference.  Without
training samples for this class, the model cannot predict it and `TTA4ViewsTransform`
is never selected during descriptor extraction.

---

## 4. `InsertInBackground` restored

**What changed:** `InsertInBackground` import and augly_method entry are kept
as in the original LINE VSC code.

**Why:** It was removed in a previous commit (`a88dde6`) as a workaround when
it was broken.  Re-testing on DAS-6 confirms it now works correctly
(`OK: [{'name': 'insert_in_background', ...}]`).  The three artificial
`Rotate(90/180/270)` entries added in that workaround commit are replaced by
`RotateWithLabel` entries (see §2 above).

---

## 5. Skip-if-exists and per-sample `try/except`

**What changed:** Each iteration now checks whether `out_vid_path` and
`out_meta_path` already exist and skips if both are present.  The entire
per-sample body is wrapped in `try/except Exception`.

**Why:** Enables resuming a partial run without re-generating already-completed
samples.  Per-sample exception handling ensures a single bad video (e.g.
corrupt source file, ffmpeg failure) does not abort the entire generation job.

---

## Net result

After regenerating training data and retraining, the editing-prediction model
will have all classes the original paper trained on:

| Class | Source augmentation |
|---|---|
| `hstack` | `HStack` |
| `vstack` | `VStack` |
| `vstack_hstack` | `VStackHStack` ← **new** |
| `rotate90_270` | `RotateWithLabel(90)`, `RotateWithLabel(270)` ← **fixed** |
| `rotate180_vflip` | `RotateWithLabel(180)`, `VFlip` ← **fixed** |
| `insert_in_background` | `InsertInBackground` ← **restored** |
| *(all others)* | unchanged |
