"""
Run the LINE VSC baseline (ISCNet) on FIVR-200K and evaluate with the FIVR protocol.

The pipeline:
  1. Extract per-frame descriptors for all FIVR videos using ISCNet
     (saves to FIVR_DIR/vsc_descriptors.npz — skip if already done)
  2. Build a FAISS index over reference descriptors
  3. For each of the 100 FIVR query videos, retrieve candidates
  4. Convert to FIVR JSON format {query_id: {video_id: score}}
  5. Compute and print mAP for DSVR / CSVR / ISVR
  6. Save results to an evaluation_results/ directory compatible with plot_results.py

Requirements:
  - timm:  pip install timm
  - Model weights at: baselines/line_vsc/meta-vsc-descriptor-runtime/submission_src/model_assets/
      isc_ft_v107.pth.tar
      copy_type_pred__convnext_clip.ckpt   (optional — used for TTA; skipped if absent)

Usage:
    # From CompactSIamese/
    python retrieval/evaluate_baselines_fivr.py

    # Skip extraction if descriptors already cached:
    python retrieval/evaluate_baselines_fivr.py --skip-extraction

    # Override model assets dir:
    python retrieval/evaluate_baselines_fivr.py --model-assets /path/to/model_assets
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Make the LINE VSC source importable
BASELINE_SRC = (
    Path(__file__).parent.parent.parent
    / "baselines/line_vsc/meta-vsc-descriptor-runtime/submission_src"
)
sys.path.insert(0, str(BASELINE_SRC))

sys.path.insert(0, str(Path(__file__).parent.parent))
from runtime_config import FIVR_DIR

ANNOTATION_FILE = Path(__file__).parent.parent.parent / "datasets/FIVR-200K/dataset/annotation.json"
DATASET_IDS_FILE = Path(__file__).parent.parent.parent / "datasets/FIVR-200K/dataset/youtube_ids.txt"
DEFAULT_MODEL_ASSETS = BASELINE_SRC / "model_assets"

FIVR_VIDEO_DIR = FIVR_DIR / "videos"
DESCRIPTORS_CACHE = FIVR_DIR / "vsc_descriptors.npz"

OUTPUT_ROOT = Path("evaluation_results")

TASK_LABELS = {
    "DSVR": ["ND", "DS"],
    "CSVR": ["ND", "DS", "CS"],
    "ISVR": ["ND", "DS", "CS", "IS"],
}

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".flv", ".mov")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_descriptors(video_dir: Path, model_assets: Path, fps: float = 1.0,
                        batch_size: int = 32, use_tta: bool = False) -> list:
    """
    Run ISCNet over all videos in video_dir.
    Returns a list of VideoFeature objects.
    """
    from src.model import create_model_in_runtime, create_copy_type_pred_model
    from src.inference_impl import VideoDataset, run_inference
    from torch.utils.data import DataLoader

    # Monkey-patch model asset paths
    import src.model as model_module
    orig_isc_path = "./model_assets/isc_ft_v107.pth.tar"
    orig_ct_path = "./model_assets/copy_type_pred__convnext_clip.ckpt"
    isc_path = model_assets / "isc_ft_v107.pth.tar"
    ct_path = model_assets / "copy_type_pred__convnext_clip.ckpt"

    if not isc_path.exists():
        raise FileNotFoundError(
            f"ISCNet weights not found at {isc_path}.\n"
            f"Download from the LINE VSC submission or VSC22 competition."
        )

    # Patch the hardcoded paths in model.py
    import unittest.mock as mock
    with mock.patch.object(model_module, '__file__', str(BASELINE_SRC / "src/model.py")):
        pass  # no-op, we patch torch.load directly below

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(str(isc_path), map_location=device)
    arch = ckpt["arch"]
    input_size = ckpt["args"].input_size

    import timm
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from src.model import ISCNet, gem, ConstDivider

    backbone = timm.create_model(arch, features_only=True)
    model = ISCNet(backbone=backbone, fc_dim=256, p=1.0, eval_p=1.0, l2_normalize=True)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    preprocessor = nn.Sequential(
        transforms.Resize((input_size, input_size)),
        ConstDivider(c=255.0),
        transforms.Normalize(
            mean=backbone.default_cfg["mean"],
            std=backbone.default_cfg["std"],
        ),
    )
    preprocessor = torch.jit.script(preprocessor).to(device)

    copytype_model = None
    if use_tta and ct_path.exists():
        from src.model import create_copy_type_pred_model as _ct
        # manually load to avoid hardcoded path
        import src.model as _m
        ct_ckpt = torch.load(str(ct_path), map_location="cpu")["state_dict"]
        arch_ct = "convnext_base_in22ft1k"
        mean_ct = (0.48145466, 0.4578275, 0.40821073)
        std_ct = (0.26862954, 0.26130258, 0.27577711)
        from src.model import CopyTypePredModel
        labels = [
            "add_noise", "blend_videos", "blur", "brightness", "change_aspect_ratio",
            "color_jitter", "contrast", "crop", "encoding_quality", "grayscale",
            "hstack", "other", "overlay", "overlay_dots", "overlay_emoji",
            "overlay_shapes", "overlay_text", "pad", "perspective_transform_and_shake",
            "pixelization", "rotate", "rotate180_vflip", "rotate90_270",
            "vstack", "vstack_hstack",
        ]
        ct_transforms = nn.Sequential(
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=mean_ct, std=std_ct),
            transforms.Resize(size=(256, 256)),
        )
        copytype_model = CopyTypePredModel(arch_ct, labels=labels, transforms=ct_transforms)
        copytype_model.load_state_dict(ct_ckpt)
        copytype_model = copytype_model.to(device).eval()

    from src.inference import VideoReaderType
    dataset = VideoDataset(
        str(video_dir),
        fps=fps,
        read_type="tensor",
        batch_size=1,
        extensions=[e.lstrip(".") for e in VIDEO_EXTS],
        video_reader=VideoReaderType.FFMPEG,
    )
    loader = DataLoader(dataset, batch_size=None, pin_memory=(device.type == "cuda"))

    video_features = []
    for vf in tqdm(
        run_inference(loader, model, transforms=preprocessor,
                      tta=use_tta, copytype_model=copytype_model, batch_size=batch_size),
        total=dataset.num_videos(),
        desc="Extracting descriptors",
    ):
        video_features.append(vf)

    return video_features


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def build_scores(query_features: list, ref_features: list,
                 candidates_per_query: int = 10000) -> dict:
    """
    Use the VSC CandidateGeneration (FAISS) to retrieve candidates,
    then return {query_id: {ref_id: score}}.
    """
    from src.vsc.candidates import CandidateGeneration, MaxScoreAggregation

    aggregation = MaxScoreAggregation()
    cg = CandidateGeneration(ref_features, aggregation)

    total_to_retrieve = candidates_per_query * len(query_features)
    print(f"Running FAISS retrieval: {len(query_features)} queries × up to {candidates_per_query} candidates...")
    candidates = cg.query(query_features, global_k=total_to_retrieve)

    results: dict[str, dict[str, float]] = {}
    for cp in candidates:
        if cp.query_id not in results:
            results[cp.query_id] = {}
        results[cp.query_id][cp.ref_id] = float(cp.score)

    return results


# ---------------------------------------------------------------------------
# FIVR evaluation
# ---------------------------------------------------------------------------

def compute_mAP(annotations, results, dataset, relevant_labels):
    mAP = []
    for query, gt_sets in annotations.items():
        if query not in results:
            continue
        query_gt = set(sum([gt_sets.get(l, []) for l in relevant_labels], []))
        query_gt = query_gt.intersection(dataset)
        if not query_gt:
            continue
        i, ri, s = 0.0, 0, 0.0
        for video, sim in sorted(results[query].items(), key=lambda x: x[1], reverse=True):
            if video != query and video in dataset:
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        mAP.append(s / len(query_gt))
    return float(np.mean(mAP)) if mAP else 0.0


def build_per_query_df(annotations, results, dataset):
    import pandas as pd
    rows = []
    all_labels = ["ND", "DS", "CS", "IS"]
    for query, gt_sets in annotations.items():
        if query not in results:
            continue
        for task, labels in TASK_LABELS.items():
            query_gt = set(sum([gt_sets.get(l, []) for l in labels], []))
            query_gt = query_gt.intersection(dataset)
            if not query_gt:
                continue
            i, ri, s = 0.0, 0, 0.0
            first_relevant = None
            ranked = sorted(results[query].items(), key=lambda x: x[1], reverse=True)
            for rank, (video, sim) in enumerate(ranked, start=1):
                if video != query and video in dataset:
                    ri += 1
                    if video in query_gt:
                        i += 1.0
                        s += i / ri
                        if first_relevant is None:
                            first_relevant = rank
            ap = s / len(query_gt)
            rows.append({
                "query_video": query,
                "task": task,
                "average_precision": ap,
                "first_relevant_rank": first_relevant,
                "num_relevant": len(query_gt),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip descriptor extraction if vsc_descriptors.npz already exists")
    parser.add_argument("--model-assets", type=str, default=str(DEFAULT_MODEL_ASSETS),
                        help="Path to directory containing isc_ft_v107.pth.tar")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to sample for extraction")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidates-per-query", type=int, default=10000,
                        help="Number of candidates to retrieve per query")
    parser.add_argument("--tta", action="store_true",
                        help="Use test-time augmentation (requires copy_type_pred model)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: auto-timestamped)")
    args = parser.parse_args()

    model_assets = Path(args.model_assets)

    # --- Step 1: Extract or load descriptors ---
    from src.vsc.storage import store_features, load_features

    if args.skip_extraction and DESCRIPTORS_CACHE.exists():
        print(f"Loading cached descriptors from {DESCRIPTORS_CACHE}...")
        all_features = load_features(str(DESCRIPTORS_CACHE))
    else:
        print(f"Extracting descriptors from {FIVR_VIDEO_DIR}...")
        all_features = extract_descriptors(
            video_dir=FIVR_VIDEO_DIR,
            model_assets=model_assets,
            fps=args.fps,
            batch_size=args.batch_size,
            use_tta=args.tta,
        )
        store_features(str(DESCRIPTORS_CACHE), all_features)
        print(f"Saved descriptors to {DESCRIPTORS_CACHE}")

    # --- Step 2: Split into queries and references ---
    with open(ANNOTATION_FILE) as f:
        annotations = json.load(f)
    dataset = set(np.loadtxt(str(DATASET_IDS_FILE), dtype=str))
    query_ids = set(annotations.keys())

    feature_map = {vf.video_id: vf for vf in all_features}
    query_features = [feature_map[q] for q in query_ids if q in feature_map]
    ref_features = list(feature_map.values())

    missing_queries = query_ids - set(feature_map.keys())
    if missing_queries:
        print(f"WARNING: {len(missing_queries)} query videos have no descriptor (not yet downloaded?)")

    print(f"Queries with descriptors: {len(query_features)}")
    print(f"Reference videos:         {len(ref_features)}")

    # --- Step 3: Retrieval ---
    results = build_scores(query_features, ref_features,
                           candidates_per_query=args.candidates_per_query)

    # --- Step 4: Output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / f"vsc_baseline_fivr_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save results JSON
    results_path = out_dir / "fivr_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f)

    # --- Step 5: mAP ---
    metrics = {}
    print("\n=== LINE VSC Baseline — FIVR-200K Zero-Shot ===")
    for task, labels in TASK_LABELS.items():
        mAP = compute_mAP(annotations, results, dataset, labels)
        metrics[f"mAP_{task}"] = mAP
        print(f"  {task}:  mAP = {mAP:.4f}")

    # Save metrics.json (compatible with plot_results.py)
    metrics["model"] = "LINE_VSC_ISCNet"
    metrics["fps"] = args.fps
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save per-query metrics CSV (compatible with plot_results.py)
    import pandas as pd
    per_query_df = build_per_query_df(annotations, results, dataset)
    per_query_df.to_csv(out_dir / "per_query_metrics.csv", index=False)

    print(f"\nResults saved to {out_dir}")
