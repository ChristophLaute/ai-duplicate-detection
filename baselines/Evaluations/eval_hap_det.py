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
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score
# + compute_mAP and compute_recall_at_k (copy from vvs script)

HDF5_FILE = FIVR_DIR / "HAP_Features/hap_features_all.h5"
MODEL_PATH = Path(__file__).resolve().parent.parent / "HAP-VR/ckpt/model.pth"
IDS_DIR    = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
ANN_FILE   = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"
THRESHOLDS = {
    "all": -0.2084,
    "nd":   0.3290,
}


def compute_accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if len(labels) else 0.0

def compute_f1_beta(predictions, labels, beta=0.5):
    return fbeta_score(labels, predictions, beta=beta, average='binary')


def load_id_list(path):
    with open(path) as f:
        return [Path(line.strip()).stem for line in f if line.strip()]
    
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
    parser.add_argument("--ds", type=str, default="5k")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--features_only", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="all", choices=["all", "nd", "ds"])
    args = parser.parse_args()
    threshold = THRESHOLDS[args.mode if args.mode != "ds" else "all"]
    SIMILAIRITES_JSON = Path(__file__).resolve().parent.parent / f"similarities_hap_{args.ds}_{args.mode}.json"

    m = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    sim_network = SimilarityNetwork[m["args"].similarity_network].get_model(**vars(m["args"]))

    sim_network.load_state_dict(m["model"])
    sim_network = sim_network.cuda().eval()


    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr_5k = json.load(f)

    query_ids = [Path(v).stem for v in fivr_5k["query"]]

    if args.ds == "5k":
        ref_ids = [Path(v).stem for v in fivr_5k["database"]]
    else:
        with open(IDS_DIR / "fivr_reference_ids.txt") as f:
            ref_ids = [l.strip() for l in f if l.strip()]
    query_generator = HDF5DatasetGenerator(str(HDF5_FILE), query_ids)
    query_loader = DataLoader(query_generator, batch_size=1, num_workers=4)
    queries, query_ids_loaded = [], []
    offline_time = 0.0
    online_time  = 0.0

    for (video_tensor,),(video_id,) in tqdm(query_loader, desc="Loading queries"):
        if video_id and video_tensor.shape[0] > 0:
            t = time.perf_counter()
            feat = sim_network.index_video(video_tensor.cuda())
            online_time += time.perf_counter() - t
            queries.append(feat)
            query_ids_loaded.append(video_id)

    ref_gen = HDF5DatasetGenerator(str(HDF5_FILE), ref_ids)
    ref_loader = DataLoader(ref_gen, batch_size=1, num_workers=4)
    similarities = {q:{} for q in query_ids_loaded}
    detections = {q:{} for q in query_ids_loaded}

    for (video_tensor,),(video_id,) in tqdm(ref_loader, desc="Processing references"):
        if video_id and video_tensor.shape[0] > 0:
            t_off = time.perf_counter()
            feat = sim_network.index_video(video_tensor.cuda())
            offline_time += time.perf_counter() - t_off

            t_on = time.perf_counter()
            sims = calculate_similarities_to_queries(sim_network, queries, feat, args)
            online_time += time.perf_counter() - t_on

            for i, qid in enumerate(query_ids_loaded):
                similarities[qid][video_id] = float(sims[i])
                detections[qid][video_id] = float(sims[i]) >= threshold
    print(f"Offline time (ref indexing): {offline_time:.2f}s")
    print(f"Online time (query indexing + comparator): {online_time:.2f}s | per query: {online_time/max(len(query_ids_loaded),1):.3f}s")
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
        for ref,score in similarities[q].items():
            if ref !=q:
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
