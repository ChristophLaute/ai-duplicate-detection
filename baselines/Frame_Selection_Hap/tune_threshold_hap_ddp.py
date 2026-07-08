import sys, json, argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "CompactSIamese"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "HAP-VR"))

from runtime_config import FIVR_DIR
from datasets.generators import HDF5DatasetGenerator
from model.similarity_network import SimilarityNetwork
from evaluation import calculate_similarities_to_queries
from sklearn.metrics import fbeta_score
from dpp_frame_selection import dpp_frame_selection, uniform_frame_selection, select_frames_by_interframe_diff


HDF5_FILE = FIVR_DIR / "HAP_Features/hap_features_all.h5"
MODEL_PATH = Path(__file__).resolve().parent.parent / "HAP-VR/ckpt/model.pth"
IDS_DIR    = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
ANN_FILE   = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"


def compute_accuracy(predictions, labels):
    """Compute the accuracy of predictions against true labels."""
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if len(labels) else 0.0


def compute_f1_beta(predictions, labels, beta=0.5):
    """Compute the F-beta score of predictions against true labels."""
    return fbeta_score(labels, predictions, beta=beta, average='binary')


def load_id_list(path):
    """Load a list of video IDs from a text file, returning only the stem of each path."""
    with open(path) as f:
        return [Path(line.strip()).stem for line in f if line.strip()]


def select_frames(descriptors, video_tensor, method, max_k_divider, energy_threshold=1e-7):
    """Determine which frames to select from a video based on the specified method and parameters."""
    if method == "dpp":
        return dpp_frame_selection(descriptors, max_k_divider=max_k_divider,
                                    energy_threshold=energy_threshold, min_frames=4)
    elif method == "interframe_diff":
        return select_frames_by_interframe_diff(descriptors, k_divider=max_k_divider, min_frames=4)
    else:
        return uniform_frame_selection(video_tensor.shape[1], k_divider=max_k_divider, min_frames=4)


def best_threshold_for(pairs):
    """Determine the best threshold for a set of (score, label) pairs by maximizing the F1-beta score."""
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


def tune_thresholds(sim_network, method, max_k_divider, energy_threshold=1e-7, ds="5k", gpu_id=0):
    """Tune the threshold for video similarity scores by evaluating a set of query-reference pairs and selecting the threshold that maximizes the F1-beta score."""
    sim_args = argparse.Namespace(features_only=False, gpu_id=gpu_id)

    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr_5k = json.load(f)

    query_ids = load_id_list(IDS_DIR / "fivr_val_query_ids.txt")
    if ds == "5k":
        ref_ids = [Path(v).stem for v in fivr_5k["database"]]
    else:
        with open(IDS_DIR / "fivr_reference_ids.txt") as f:
            ref_ids = [l.strip() for l in f if l.strip()]

    query_generator = HDF5DatasetGenerator(str(HDF5_FILE), query_ids)
    query_loader = DataLoader(query_generator, batch_size=1, num_workers=4)
    queries, query_ids_loaded = [], []
    for (video_tensor,), (video_id,) in tqdm(query_loader, desc=f"[tune {method} k={max_k_divider}] queries"):
        if video_id and video_tensor.shape[0] > 0:
            selected = select_frames(video_tensor.squeeze(), video_tensor, method, max_k_divider, energy_threshold)
            if len(selected) < 2:
                if video_tensor.shape[0] < 2:
                    continue
                selected = list(range(video_tensor.shape[0]))
            feat = sim_network.index_video(video_tensor[selected].cuda())
            queries.append(feat)
            query_ids_loaded.append(video_id)

    ref_gen = HDF5DatasetGenerator(str(HDF5_FILE), ref_ids)
    ref_loader = DataLoader(ref_gen, batch_size=1, num_workers=4)
    similarities = {q: {} for q in query_ids_loaded}
    for (video_tensor,), (video_id,) in tqdm(ref_loader, desc=f"[tune {method} k={max_k_divider}] refs"):
        if video_id and video_tensor.shape[0] > 0:
            selected = select_frames(video_tensor.squeeze(), video_tensor, method, max_k_divider, energy_threshold)
            if len(selected) < 2:
                if video_tensor.shape[0] < 2:
                    continue
                selected = list(range(video_tensor.shape[0]))
            feat = sim_network.index_video(video_tensor[selected].cuda())
            sims = calculate_similarities_to_queries(sim_network, queries, feat, sim_args)
            for i, qid in enumerate(query_ids_loaded):
                similarities[qid][video_id] = float(sims[i])

    with open(ANN_FILE) as f:
        annotations = json.load(f)

    results = {}
    for mode, relevant_labels in [("all", ["ND", "DS"]), ("nd", ["ND"])]:
        pairs = []
        for q in similarities:
            gt = set()
            for l in relevant_labels:
                gt.update(annotations[q].get(l, []))
            exclude = set(annotations[q].get("DS", [])) if mode == "nd" else set()
            for ref, score in similarities[q].items():
                if ref != q and ref not in exclude:
                    pairs.append((score, ref in gt))
        if not pairs:
            results[mode] = None
            continue
        best_threshold, best_f1_beta, best_accuracy = best_threshold_for(pairs)
        print(f"=== {mode.upper()} ===")
        print(f"Best threshold: {best_threshold:.4f} with F1-beta: {best_f1_beta:.4f}, Accuracy: {best_accuracy:.4f}")
        results[mode] = float(best_threshold)

    return results["all"], results["nd"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="5k")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--method", type=str, default="dpp", choices=["dpp", "uniform", "interframe_diff"])
    parser.add_argument("--max_k_divider", type=int, default=2, help="Divider for max_k in frame selection (e.g., 2 means max_k = num_frames // 2)")
    parser.add_argument("--energy_threshold", type=float, default=1e-7, help="Energy threshold for DPP frame selection")
    args = parser.parse_args()

    m = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    sim_network = SimilarityNetwork[m["args"].similarity_network].get_model(**vars(m["args"]))
    sim_network.load_state_dict(m["model"])
    sim_network = sim_network.cuda().eval()

    threshold_all, threshold_nd = tune_thresholds(
        sim_network, args.method, args.max_k_divider, args.energy_threshold, args.ds, args.gpu_id
    )
    if threshold_all is not None:
        print(f"THRESHOLD_ALL={threshold_all:.4f}")
    if threshold_nd is not None:
        print(f"THRESHOLD_ND={threshold_nd:.4f}")


if __name__ == "__main__":
    main()
