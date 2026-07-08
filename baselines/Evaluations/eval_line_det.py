import sys
import os
import json
import time
import faiss
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "line_vsc/meta-vsc-descriptor-runtime/submission_src"))
from src.model import create_model_in_runtime, create_copy_type_pred_model
from src.postproc import sliding_pca
from src.vsc.index import VideoFeature
from src.vsc.candidates import CandidateGeneration, MaxScoreAggregation, ScoreAggregation
from src.vsc.storage import load_features

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "CompactSIamese"))
from runtime_config import FIVR_DIR, get_device

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "line_vsc/FIVR_exp"))
from precompute_line_desc import compute_descriptors_line


IDS_DIR  = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
DESC_DIR = FIVR_DIR / "line_descriptors_postprocessed" / "50k"
ANN_FILE = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"
EXTENSIONS = ["mp4", "webm", "mkv", "flv", "mov", "avi"]

THRESHOLDS = {
    "all": 0.8061,
    "nd":  0.8112,
}



class AvgScoreAggregation(ScoreAggregation):
    def aggregate(self, match):
        return np.mean([m.score for m in match.matches])

def compute_query_descriptor_online(video_path, model, device, copytype_model, preprocessor, pca_mat, low_var_dim, score_norm_index, beta=1.0, no_consistency_weight=False, no_temporal_concat=False, no_score_norm=False, not_tta=False):
    result = compute_descriptors_line(video_path, model, device, tta=not not_tta, copytype_model=copytype_model, preprocessor=preprocessor)
    if result is None:
        return None
    desc = result["descriptor"]
    timestamps = result["timestamps"]
    if no_consistency_weight:   
        c = 1.0
    else:
        c = float((desc @ desc.T).mean())
    if c == 0:
        return None
    vf = VideoFeature(video_id=video_path.stem, feature=(desc / c).astype(np.float32), timestamps=timestamps)
    if not no_temporal_concat:
        processed = sliding_pca([vf], pca_mat, stride=3)
        if not processed:
            return None
        vf = processed[0]
    if not no_score_norm and not no_temporal_concat:
        feat = np.delete(vf.feature, low_var_dim, axis=1)
        feat = normalize(feat)
        sim, _ = score_norm_index.search(feat.astype(np.float32), 1)
        feat = np.concatenate([feat, -beta * sim[:, :1]], axis=1)
    else:
        feat = normalize(vf.feature)
    feat = feat * c
    return VideoFeature(video_id=vf.video_id, timestamps=vf.timestamps, feature=feat)


def compute_accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if len(labels) else 0.0

def compute_f1_beta(predictions, labels, beta=0.5):
    return fbeta_score(labels, predictions, beta=beta, average='binary')

def evaluate_pairs(pairs, threshold):
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
    parser.add_argument("--no_consistency_weight", action="store_true", default=False, help="Disable consistency weight application to descriptors")
    parser.add_argument("--no_temporal_concat", action="store_true", default=False, help="Disable temporal concatenation of descriptors before PCA")
    parser.add_argument("--no_score_norm", action="store_true", default=False, help="Disable score normalization using noise reference stats")
    parser.add_argument("--no_tta", action="store_true", default=False, help="Disable test-time augmentation for query descriptor computation")
    parser.add_argument("--aggregation_type", type=str, default="max", choices=["max", "avg"], help="Type of score aggregation for candidate generation")
    parser.add_argument("--desc-dir", type=str,
    default=str(FIVR_DIR / "line_descriptors_postprocessed" / "50k"),
    help="Directory containing preprocessed descriptors and transform parameters")
    parser.add_argument("--threshold-all", type=float, default=None,
        help="Override detection threshold for mode=all/ds (default: tune_threshold_line.py's global value)")
    parser.add_argument("--threshold-nd", type=float, default=None,
        help="Override detection threshold for mode=nd (default: tune_threshold_line.py's global value)")

    args = parser.parse_args()
    if args.mode == "nd":
        threshold = args.threshold_nd if args.threshold_nd is not None else THRESHOLDS["nd"]
    else:
        threshold = args.threshold_all if args.threshold_all is not None else THRESHOLDS["all"]

    variant = ""
    if args.no_score_norm: variant += "_nosn"
    if args.no_consistency_weight: variant += "_nocw"
    if args.no_temporal_concat: variant += "_notc"
    if args.no_tta: variant += "_notta"
    if args.aggregation_type == "avg": variant += "_avg"
    if not variant: variant = "_full"
    SIMILAIRITES_JSON = Path(__file__).resolve().parent.parent / f"similarities_line_{args.ds}_{args.mode}{variant}.json"

    DESC_DIR = Path(args.desc_dir)
    VIDEO_DIR = FIVR_DIR / "videos"

    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr5k = json.load(f)

    with open(DESC_DIR / "references.npz", "rb") as f:
        refs = load_features(f)

    # offline setup
    BASELINE_SRC = Path(__file__).resolve().parent.parent / "line_vsc/meta-vsc-descriptor-runtime/submission_src"
    os.chdir(BASELINE_SRC)
    device = get_device()
    model, preprocessor = create_model_in_runtime(transforms_device=device)
    model = model.to(device).eval().half()

    copytype_model = create_copy_type_pred_model(transforms_device=device)
    copytype_model = copytype_model.to(device).eval()
    os.chdir(Path(__file__).parent)
    if not args.no_temporal_concat:
        pca_mat = faiss.read_VectorTransform(str(DESC_DIR / "pca_matrix.faiss"))
    else:
        pca_mat = None
    low_var_dim = int(np.load(DESC_DIR / "low_var_dim.npy")[0])
    if not args.no_score_norm:
        score_norm_index = faiss.read_index(str(DESC_DIR / "score_norm_index.faiss"))
    else:
        score_norm_index = None

    n_ref = sum(len(r) for r in refs)
    ti1 = time.perf_counter()
    if args.aggregation_type == "max":
        cg = CandidateGeneration(refs, MaxScoreAggregation())
    else:
     
        cg = CandidateGeneration(refs, AvgScoreAggregation())
    ti2 = time.perf_counter()
    print(f"Index built in {ti2 - ti1:.2f}s")

    all_refs = {vf.video_id for vf in refs}
    if args.ds == "fivr5k":
        db5k = {Path(v).stem for v in fivr5k["database"]}
        db = all_refs & db5k
    else:
        db = all_refs

    online_times = []
    similarities = {}
    detections = {}
    for q in tqdm(fivr5k["query"], desc="Evaluating queries"):
        stem = Path(q).stem
        video_path = next((VIDEO_DIR / f"{stem}.{ext}" for ext in EXTENSIONS if (VIDEO_DIR / f"{stem}.{ext}").exists()), None)
        if video_path is None:
            continue
        t0 = time.perf_counter()
        q_desc = compute_query_descriptor_online(video_path, model, device, copytype_model, preprocessor, pca_mat, low_var_dim, score_norm_index, beta=1.0, no_consistency_weight=args.no_consistency_weight, no_temporal_concat=args.no_temporal_concat, no_score_norm=args.no_score_norm, not_tta=args.no_tta)
        if q_desc is None:
            continue
        candidates = cg.query([q_desc], n_ref)
        t1 = time.perf_counter()
        similarities[q_desc.video_id] = {cand.ref_id: float(cand.score) for cand in candidates}
        detections[q_desc.video_id] = {cand.ref_id: bool(cand.score >= threshold) for cand in candidates}

        online_times.append(t1 - t0)

    print(f"Online: {sum(online_times):.2f}s total, {np.mean(online_times):.2f}s per query")

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
        if args.mode == "nd":
            exclude = set(annotations[q].get("DS", []))
        else:
            exclude = set()
        for ref, score in similarities[q].items():
            if ref != q and ref in db and ref not in exclude:
                pairs.append((score, ref in gt))
                if ref in annotations[q].get("ND", []):
                    nd_pairs.append((score, ref in gt))
                elif ref in annotations[q].get("DS", []):
                    ds_pairs.append((score, ref in gt))
                else:
                    neg_pairs.append((score, ref in gt))

    with open(SIMILAIRITES_JSON, "w") as f:
        json.dump({
            "similarities": similarities,
            "detections": detections,
            "threshold": threshold,
            "task": args.mode,
            "db": args.ds,
        }, f, indent=2)

    if args.mode == "nd":
        print("ND performance (ND vs negatives):")
        evaluate_pairs(nd_pairs + neg_pairs, threshold)
    else:
        print("Overall performance (ND+DS vs negatives):")
        evaluate_pairs(pairs, threshold)


if __name__ == "__main__":
    main()
