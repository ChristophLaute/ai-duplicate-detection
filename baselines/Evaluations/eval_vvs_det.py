import sys
import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "CompactSIamese"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "VVS"))

from runtime_config import FIVR_DIR
from extract_fivr_subset import extract_features
from model.feature_extractor import FeatureExtractor
from model.model import SuppressionWeightGenerationStage, EasyDistractorEliminationStage


IDS_DIR   = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
DESC_DIR  = FIVR_DIR / "vvs_video_descriptors"
ANN_FILE  = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"
VIDEO_DIR = FIVR_DIR / "videos"
CHECKPOINT = Path(__file__).resolve().parent.parent / "VVS/jobs/table_benchmark_dim_3840/model/m00080000.pth"
THRESHOLDS = {
    "all": 0.3283,
    "nd":  0.4411,
}


def find_video_path(video_id):
    """Find the path to a video file given its ID, checking for common video file extensions."""
    for ext in [".mp4", ".avi", ".mkv"]:
        p = VIDEO_DIR / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def compute_query_descriptor(video_path, device, feature_extractor, tsm_tgm_model, ddm_model):
    """Compute the video descriptor for a query video using the VVS model."""
    frame_feats = extract_features(video_path, feature_extractor, device)
    if frame_feats is None:
        return None
    frame_feats = frame_feats.cuda().unsqueeze(0)
    with torch.no_grad():
        in_data = {"anchor": frame_feats}
        ddm_out = ddm_model(in_data["anchor"])
        if (torch.sigmoid(ddm_out["confidence"]) > 0.5).float().sum() != 0:
            mask = (torch.sigmoid(ddm_out["confidence"]) > 0.5)
            in_data["anchor"] = in_data["anchor"][:, mask]
        while in_data["anchor"].shape[1] < 4:
            in_data["anchor"] = torch.cat([in_data["anchor"], in_data["anchor"]], dim=1)
        feats_out = tsm_tgm_model(in_data, is_anc_processed=False)
    return feats_out["features"].squeeze().cpu().numpy()


def load_descriptor(vid):
    """Load a video descriptor from a .pt file, returning None if the file does not exist."""
    p = DESC_DIR / f"{Path(vid).stem}.pt"
    if not p.exists():
        return None
    return torch.load(p).squeeze().numpy()


def compute_accuracy(predictions, labels):
    """Compute the accuracy of predictions against true labels."""
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if len(labels) else 0.0


def compute_f1_beta(predictions, labels, beta=0.5):
    """Compute the F-beta score of predictions against true labels."""
    return fbeta_score(labels, predictions, beta=beta, average='binary')


def evaluate_pairs(pairs, threshold):
    """Evaluate a set of (score, label) pairs against a threshold, computing accuracy, F1-beta, F1, precision, and recall."""
    scores = np.array([s for s, _ in pairs])
    labels = np.array([l for _, l in pairs])
    predictions = scores >= threshold
    acc = compute_accuracy(predictions, labels)
    f1_beta = compute_f1_beta(predictions, labels, beta=0.5)
    f1 = f1_score(labels, predictions, average='binary')
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    print(f"Best threshold: {threshold:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-beta: {f1_beta:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="fivr5k")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "nd", "ds"])
    args = parser.parse_args()
    threshold = THRESHOLDS[args.mode if args.mode != "ds" else "all"]
    SIMILAIRITES_JSON = Path(__file__).resolve().parent.parent / f"similarities_vvs_{args.ds}_{args.mode}.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vvs_args = argparse.Namespace(
        suppression="vvs", vvs_vinit="v_t", vvs_h_connect=True,
        vvs_sigmoid_T=512.0, vvs_sigmoid_T_tsm=1.0, vvs_sigmoid_T_ddm=1.0,
        vvs_tsm=True, vvs_tgm=True, vvs_ddm=True, refinement=True,
        base_normalize=True, pca_whitening=True, pca_reduction=None,
        gpu=0, feature_extract="resnet50_l4imac", feature_backbone="resnet50",
        feature_type="l4imac", retrieve_triplet=False, iterations=-1,
        cycles=1, imple_flag="eval", weight_triplet=1.0, weight_saliency=1.0,
        weight_frame=1.0, distractor_sampling_ratio="20_50",
        mag_opt=5, mag_thresh=40, neg_len=32, thresh_s=0.5,
        baseline=False, use_oracle=False, fg_gt_label=None,
    )
    VVS_DIR = Path(__file__).resolve().parent.parent / "VVS"
    os.chdir(VVS_DIR)
    loaded = torch.load(CHECKPOINT, map_location="cpu")
    feature_extractor = FeatureExtractor(network="resnet50").to(device).eval()
    tsm_tgm_model = SuppressionWeightGenerationStage(vvs_args).cuda().eval()
    ddm_model = EasyDistractorEliminationStage(vvs_args).cuda().eval()
    tsm_tgm_model.load_state_dict(loaded["tsm_tgm_model"], strict=False)
    ddm_model.load_state_dict(loaded["ddm_model"], strict=False)
    os.chdir(Path(__file__).parent)

    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr_5k = json.load(f)
    query_ids = [Path(v).stem for v in fivr_5k["query"]]
    if args.ds == "fivr5k":
        ref_ids = [Path(v).stem for v in fivr_5k["database"]]
    else:
        with open(IDS_DIR / "fivr_reference_ids.txt") as f:
            ref_ids = [l.strip() for l in f if l.strip()]

    refs = {r: load_descriptor(r) for r in ref_ids}
    refs = {k: v for k, v in refs.items() if v is not None}
    ref_ids_keys = list(refs.keys())
    ref_matrix = np.stack([refs[r] for r in ref_ids_keys])

    similarities = {}
    detections = {}
    online_times = []
    for qid in tqdm(query_ids, desc="Computing similarities"):
        video_path = find_video_path(qid)
        if video_path is None:
            continue
        t0 = time.perf_counter()
        qdesc = compute_query_descriptor(video_path, device, feature_extractor, tsm_tgm_model, ddm_model)
        if qdesc is None:
            continue
        sim = ref_matrix @ qdesc
        t1 = time.perf_counter()
        similarities[qid] = {rid: float(s) for rid, s in zip(ref_ids_keys, sim)}
        detections[qid] = {rid: float(s) >= threshold for rid, s in zip(ref_ids_keys, sim)}
        online_times.append(t1 - t0)

    print(f"Online: {sum(online_times):.2f}s total, {np.mean(online_times):.4f}s per query")
    with open(ANN_FILE) as f:
        annotations = json.load(f)

    if args.mode == "nd":
        relevant_labels = ["ND"]
    else:
        relevant_labels = ["ND", "DS"]

    pairs = []
    nd_pairs = []
    ds_pairs = []
    neg_pairs = []
    for q in similarities:
        gt = set()
        for l in relevant_labels:
            gt.update(annotations[q].get(l, []))
        for ref, score in similarities[q].items():
            if ref != q:
                pairs.append((score, ref in gt))
                if ref in annotations[q].get("ND", []):
                    nd_pairs.append((score, ref in gt))
                elif ref in annotations[q].get("DS", []):
                    ds_pairs.append((score, ref in gt))
                else:
                    neg_pairs.append((score, ref in gt))

    if args.mode == "nd":
        print("ND performance (ND vs negatives):")
        evaluate_pairs(nd_pairs + neg_pairs, threshold)
    else:
        print("Overall performance (ND+DS vs negatives):")
        evaluate_pairs(pairs, threshold)

    with open(SIMILAIRITES_JSON, "w") as f:
        json.dump({
            "similarities": similarities,
            "detections": detections,
            "threshold": threshold,
            "task": args.mode,
            "db": args.ds,
        }, f, indent=2)

if __name__ == "__main__":
    main()
