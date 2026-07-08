import sys, json, time, argparse
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
# + compute_mAP and compute_recall_at_k (copy from vvs script)

HDF5_FILE = FIVR_DIR / "HAP_Features/hap_features_all.h5"
MODEL_PATH = Path(__file__).resolve().parent.parent / "HAP-VR/ckpt/model.pth"
IDS_DIR    = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
ANN_FILE   = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"


def compute_accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if len(labels) else 0.0

def compute_f1_beta(predictions, labels, beta=0.5):
    return fbeta_score(labels, predictions, beta=beta, average='binary')


def load_id_list(path):
    with open(path) as f:
        return [Path(line.strip()).stem for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="5k")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--features_only", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "nd", "ds"])
    args = parser.parse_args()

    m = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    sim_network = SimilarityNetwork[m["args"].similarity_network].get_model(**vars(m["args"]))

    sim_network.load_state_dict(m["model"])
    sim_network = sim_network.cuda().eval()


    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr_5k = json.load(f)

    query_ids = load_id_list(IDS_DIR / "fivr_val_query_ids.txt")

    if args.ds == "5k":
        ref_ids = [Path(v).stem for v in fivr_5k["database"]]
    else:
        with open(IDS_DIR / "fivr_reference_ids.txt") as f:
            ref_ids = [l.strip() for l in f if l.strip()]
    query_generator = HDF5DatasetGenerator(str(HDF5_FILE), query_ids)
    query_loader = DataLoader(query_generator, batch_size=1,  num_workers=4)
    queries,query_ids_loaded = [], []
    for (video_tensor,),(video_id,) in tqdm(query_loader, desc="Loading queries"):
        if video_id and video_tensor.shape[0] > 0:
            feat = sim_network.index_video(video_tensor.cuda())
            queries.append(feat)
            query_ids_loaded.append(video_id)
    ref_gen = HDF5DatasetGenerator(str(HDF5_FILE), ref_ids)
    ref_loader = DataLoader(ref_gen, batch_size=1, num_workers=4)
    similarities = {q:{} for q in query_ids_loaded}
    for (video_tensor,),(video_id,) in tqdm(ref_loader, desc="Processing references"):
        if video_id and video_tensor.shape[0] > 0:
            feat = sim_network.index_video(video_tensor.cuda())
            sims = calculate_similarities_to_queries(sim_network, queries, feat,args)
            for i, qid in enumerate(query_ids_loaded):
                similarities[qid][video_id] = float(sims[i])
    with open(ANN_FILE) as f:
        annotations = json.load(f)
    
    if args.mode == "nd":
        relevant_labels = ["ND"]
    elif args.mode == "ds":
        relevant_labels = ["DS"]
    else:
        relevant_labels = ["ND", "DS"]
    print(f"=== Tuning mode: {args.mode.upper()} ===")
    pairs = []
    for q in similarities:
        gt = set()
        for l in relevant_labels:
            gt.update(annotations[q].get(l, []))
        # Exclude the complementary category per query so the threshold is tuned
        # on target-vs-true-negatives only (not target-vs-other-duplicates).
        if args.mode == "nd":
            exclude = set(annotations[q].get("DS", []))
        elif args.mode == "ds":
            exclude = set(annotations[q].get("ND", []))
        else:
            exclude = set()
        for ref, score in similarities[q].items():
            if ref != q and ref not in exclude:
                pairs.append((score, ref in gt))
    scores = np.array([s for s, _ in pairs])
    labels = np.array([l for _, l in pairs])
    order = np.argsort(scores) [::-1]  # Sort in descending order
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
    print(f"Best threshold: {best_threshold:.4f} with F1-beta: {best_f1_beta:.4f}, Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
