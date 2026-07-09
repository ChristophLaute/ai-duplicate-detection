# HAP-DPP: Diversity-Aware Frame Selection for HAP-VR

Our extension to HAP-VR that reduces its online computational cost by
selecting a subset of frame descriptors before similarity computation,
instead of comparing every extracted frame. Compares three strategies:
uniform subsampling, interframe-difference peak selection, and a
DPP-inspired greedy diversity selection (our contribution).

## Files

- `dpp_frame_selection.py` — Implements all three
  selection strategies: `dpp_frame_selection` (ours), `uniform_frame_selection`,
  `select_frames_by_interframe_diff`.
- `tune_threshold_hap_ddp.py` — the DPP-aware threshold tuner. 
- `HAP_DDP_retr.py` — retrieval metrics for one frame-selection
  method
- `HAP_DDP_det.py` — detection metrics
  for one method at a time. Hardcoded thresholds in a
  `THRESHOLDS` dict. Adapt as needed. 

## How it merges with HAP-VR

Frame selection is applied to the frame-level
descriptors *before* `index_video` — i.e., this doesn't modify HAP-VR's
model at all, only which of its already-computed descriptors get compared
at query time.
