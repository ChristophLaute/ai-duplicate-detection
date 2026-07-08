import sys
import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "line_vsc/meta-vsc-descriptor-runtime/submission_src"))
from src.model import create_model_in_runtime, create_copy_type_pred_model
from src.vsc.candidates import CandidateGeneration, MaxScoreAggregation, ScoreAggregation
from src.vsc.storage import load_features

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "CompactSIamese"))
from runtime_config import FIVR_DIR, get_device

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "line_vsc/FIVR_exp"))

from eval_line_det import compute_query_descriptor_online


IDS_DIR  = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
ANN_FILE = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"
EXTENSIONS = ["mp4", "webm", "mkv", "flv", "mov", "avi"]


class AvgScoreAggregation(ScoreAggregation):
    """aggregate scores by averaging the scores of all matches for a given reference video."""
    def aggregate(self, match):
        return np.mean([m.score for m in match.matches])


def compute_accuracy(predictions, labels):
    """Compute the accuracy of predictions against true labels."""
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if len(labels) else 0.0


def best_threshold_for(pairs):
    """Determine the best threshold for binary classification based on F1-beta score."""
    scores = np.array([s for s, _ in pairs])
    labels = np.array([l for _, l in pairs])
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(bool)
    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(~sorted_labels)
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (np.sum(sorted_labels) + 1e-10)
    f1_beta = (1 + 0.5**2) * (precision * recall) / (0.5**2 * precision + recall + 1e-10)
    best_index = np.argmax(f1_beta)
    best_threshold = sorted_scores[best_index]
    best_f1_beta = f1_beta[best_index]
    best_accuracy = compute_accuracy((scores >= best_threshold).astype(int), labels)
    return best_threshold, best_f1_beta, best_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc-dir", type=str,
        default=str(FIVR_DIR / "line_descriptors_postprocessed" / "50k"),
        help="Directory containing references.npz / val_queries.npz (and transform params)")
    parser.add_argument("--no_tta", action="store_true", default=False,
        help="Recompute val query descriptors online without test-time augmentation "
             "(only meaningful for the full-model desc-dir, which has pca_matrix.faiss "
             "and score_norm_index.faiss)")
    parser.add_argument("--aggregation_type", type=str, default="max", choices=["max", "avg"],
        help="Score aggregation used for candidate generation")
    args = parser.parse_args()

    DESC_DIR = Path(args.desc_dir)

    with open(DESC_DIR / "references.npz", "rb") as f:
        refs = load_features(f)

    if args.no_tta:
        VIDEO_DIR = FIVR_DIR / "videos"
        device = get_device()
        model, preprocessor = create_model_in_runtime(transforms_device=device)
        model = model.to(device).eval().half()
        copytype_model = create_copy_type_pred_model(transforms_device=device).eval().to(device)
        pca_mat = faiss.read_VectorTransform(str(DESC_DIR / "pca_matrix.faiss"))
        low_var_dim = int(np.load(DESC_DIR / "low_var_dim.npy")[0])
        score_norm_index = faiss.read_index(str(DESC_DIR / "score_norm_index.faiss"))

        with open(IDS_DIR / "fivr_val_query_ids.txt") as f:
            val_query_ids = [l.strip() for l in f if l.strip()]

        queries = []
        for vid in tqdm(val_query_ids, desc="Computing val query descriptors (no TTA)"):
            video_path = next((VIDEO_DIR / f"{vid}.{ext}" for ext in EXTENSIONS if (VIDEO_DIR / f"{vid}.{ext}").exists()), None)
            if video_path is None:
                continue
            vf = compute_query_descriptor_online(
                video_path, model, device, copytype_model, preprocessor,
                pca_mat, low_var_dim, score_norm_index, beta=1.0, not_tta=True,
            )
            if vf is not None:
                queries.append(vf)
    else:
        with open(DESC_DIR / "val_queries.npz", "rb") as f:
            queries = load_features(f)

    if args.aggregation_type == "max":
        cg = CandidateGeneration(refs, MaxScoreAggregation())
    else:
        cg = CandidateGeneration(refs, AvgScoreAggregation())

    similarities = {}
    for qf in tqdm(queries, desc="Computing similarities"):
        candidates = cg.query([qf], sum(len(vf) for vf in refs))
        similarities[qf.video_id] = {c.ref_id: c.score for c in candidates}

    with open(ANN_FILE) as f:
        annotations = json.load(f)

    for mode, relevant_labels, env_name in [("all", ["ND", "DS"], "THRESHOLD_ALL"), ("nd", ["ND"], "THRESHOLD_ND")]:
        pairs = []
        for q in similarities:
            gt = set()
            for l in relevant_labels:
                gt.update(annotations.get(q, {}).get(l, []))
            if mode == "nd":
                exclude = set(annotations.get(q, {}).get("DS", []))
            else:
                exclude = set()
            for ref, score in similarities[q].items():
                if ref != q and ref not in exclude:
                    pairs.append((score, ref in gt))
        if not pairs:
            continue
        best_threshold, best_f1_beta, best_accuracy = best_threshold_for(pairs)
        print(f"=== {mode.upper()} ===")
        print(f"Best threshold: {best_threshold:.4f} with F1-beta: {best_f1_beta:.4f}, Accuracy: {best_accuracy:.4f}")
        print(f"{env_name}={best_threshold:.4f}")


if __name__ == "__main__":
    main()
