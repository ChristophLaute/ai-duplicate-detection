import argparse
import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm
import time
import torch
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score


sys.path.insert(0, str(Path(__file__).parent.parent))


from model.pytorch_model import SiameseNetwork
from retrieval.two_stage_retr_pred_head import whole_video_logit_from_descriptors
from runtime_config import FIVR_DIR, get_device

FIVR_DESCR_DIR = FIVR_DIR / "descriptors"
FIVR_INDEX_DIR = FIVR_DIR / "frame_index"
FIVR_INDEX_DIR.mkdir(parents=True, exist_ok=True)

FIVR_INDEX_PATH = FIVR_INDEX_DIR / "faiss_frame_index.bin"
FIVR_METADATA_PATH = FIVR_INDEX_DIR / "frame_metadata.pkl"
FIVR_CACHE_PATH = FIVR_INDEX_DIR / "descriptor_cache.pkl"

ANNOTATION_FILE = Path(__file__).parent / "dataset/annotation.json"
DATASET_IDS_FILE = Path(__file__).parent.parent.parent / "datasets/FIVR-200K/dataset/youtube_ids.txt"

RELEVANT_LABELS = ["ND", "DS"]
THRESHOLDS = {
    "all": 1.4628,
    "nd":  1.4987,
}


def load_descriptor(path):
    """Load a descriptor from a .npy file and return the 'descriptors' key."""
    obj = np.load(path, allow_pickle=True).item()
    return obj["descriptors"]
def evaluate_pairs(pairs, threshold):
    """Evaluate pairs of scores and labels against a threshold, computing accuracy, F1-beta, F1, precision, and recall."""      
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

def compute_accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if len(labels) else 0.0

def compute_f1_beta(predictions, labels, beta=0.5):
    return fbeta_score(labels, predictions, beta=beta, average="binary")

def load_id_list(path):
    with open(path) as f:
        return [Path(line.strip()).stem for line in f if line.strip()]

def compute_mAP(annotations, results, dataset, relevant_labels):
    """Compute mean Average Precision (mAP) for the given annotations and detection results."""
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

def compute_recall_at_k(annotations, results, dataset, relevant_labels, k):
    """Compute Recall@k for the given annotations and detection results."""
    recalls = []
    for query, gt_sets in annotations.items():
        if query not in results:
            continue
        query_gt = set(sum([gt_sets.get(l, []) for l in relevant_labels], []))
        query_gt = query_gt.intersection(dataset)
        if not query_gt:
            continue
        top_k_videos = [video for video, _ in sorted(results[query].items(), key=lambda x: x[1], reverse=True)
                if video != query and video in dataset][:k]

        recall = len(set(top_k_videos).intersection(query_gt)) / len(query_gt)
        recalls.append(recall)
    return float(np.mean(recalls)) if recalls else 0.0

def main(args):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(args.model_path, map_location=get_device()))
    model = model.to(get_device())
    model.eval()
    SIMILAIRITES_JSON = Path(__file__).parent.parent / f"similarities_siamese_{args.ds}_{args.mode}.json"
    with open(args.fivr_vid_json, "r") as f:
        split = json.load(f)
    query_ids = [Path(vid).stem for vid in split["query"]]

    if args.ds == "5k":
        db_ids = set([Path(vid).stem for vid in split["database"]])
    else:
        with open(Path(args.fivr_vid_json).parent / "fivr_reference_ids.txt") as f:
            db_ids = set(l.strip() for l in f if l.strip())

    with open(args.annotation, "r") as f:
        annotations = json.load(f)

    descriptor_cache = {}
    for video_id in tqdm(db_ids, desc="Loading descriptors"):
        descriptor_path = args.descriptors_dir / f"{video_id}.npy"
        if descriptor_path.exists():
            descriptor = load_descriptor(descriptor_path)
            if descriptor is not None:
                descriptor_cache[video_id] = descriptor
    print(f"Loaded  {len(descriptor_cache)} descriptors out of {len(db_ids)} database videos.")
    results = {}
    detections = {}
    retrieval_times = []
    for query_id in tqdm(query_ids, desc="Evaluating queries"):
        query_path = args.descriptors_dir / f"{query_id}.npy"
        if not query_path.exists():
            continue
        query_descriptor = load_descriptor(query_path)
        if query_descriptor is None:
            continue
       
        scores = {}
        for video_id, db_descriptor in descriptor_cache.items():
            if video_id == query_id:
                continue
            score = whole_video_logit_from_descriptors(query_descriptor, db_descriptor, model, get_device())

            scores[video_id] = score
        detections[query_id] = {video_id: score >= THRESHOLDS[args.mode] for video_id, score in scores.items()}
            
        
        
        
        results[query_id] = scores
    if args.mode == "nd":
        relevant_labels = ["ND"]
    else:
        relevant_labels = ["ND", "DS"]
    threshold = THRESHOLDS[args.mode if args.mode != "ds" else "all"]
    pairs = []
    nd_pairs = []
    ds_pairs = []
    neg_pairs = []
    for q in results:
        gt = set()
        for l in relevant_labels:
            gt.update(annotations[q].get(l, []))
        for ref,score in results[q].items():
            if ref !=q:
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
            "similarities": results,
            "detections": detections,
            "threshold": threshold,
            "task": args.mode,
        }, f, indent=2)




if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=Path, default=Path("best_siamese_model.pth"))
        parser.add_argument("--fivr_vid_json", type=Path, default=Path("fivr_subset_50k/fivr5k_vid.json"))
        parser.add_argument("--annotation", type=Path, default=ANNOTATION_FILE)
        parser.add_argument("--descriptors_dir", type=Path, default=FIVR_DIR / "descriptors")
        parser.add_argument("--output_dir", type=Path, default=Path("evaluation_results"))
        parser.add_argument("--val_query_ids", type=Path, default=None)
        parser.add_argument("--ds", type=str, default="5k", choices=["5k", "50k"])
        parser.add_argument("--mode", type=str, default="all", choices=["all", "nd", "ds"])
        args = parser.parse_args()
        main(args)
        
