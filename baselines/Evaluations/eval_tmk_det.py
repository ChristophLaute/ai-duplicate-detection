import sys
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import fbeta_score, f1_score, precision_score, recall_score
import tmkpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "CompactSIamese"))

from runtime_config import FIVR_DIR




IDS_DIR  = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
DESC_DIR = FIVR_DIR / "tmk_descriptors"
ANN_FILE = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"
VIDEO_DIR = FIVR_DIR / "videos"
EXTENSIONS = ["mp4", "webm", "mkv", "flv", "mov", "avi"]
THRESHOLDS = {
    "all": 0.7383,
    "nd":  0.8540,
}


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


def query_batched(q_path, ref_tmk_files, batch_size=5000):
    """
    Wrapper around tmkpy.query that processes references in chunks.
    """
    
    all_scores = []
    for i in range(0, len(ref_tmk_files), batch_size):
        batch = ref_tmk_files[i:i + batch_size]
        all_scores.extend(tmkpy.query(q_path, batch, len(batch)))
    return all_scores


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
    SIMILAIRITES_JSON = Path(__file__).resolve().parent.parent / f"similarities_tmk_{args.ds}_{args.mode}.json"
    

    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr_5k = json.load(f)
    query_ids = [Path(v).stem for v in fivr_5k["query"]]
    if args.ds == "fivr5k":
        ref_ids = [Path(v).stem for v in fivr_5k["database"]]
    else:
        with open(IDS_DIR / "fivr_reference_ids.txt") as f:
            ref_ids = [l.strip() for l in f if l.strip()]
        
    ref_tmk_files = [str(DESC_DIR / f"{rid}.tmk") for rid in ref_ids
                 if (DESC_DIR / f"{rid}.tmk").exists()]
    ref_ids_vid = [rid for rid in ref_ids if (DESC_DIR / f"{rid}.tmk").exists()]
    similarities = {}
    detections = {}
    online_times = []
    query_ids = [qid for qid in query_ids if any((VIDEO_DIR / f"{qid}.{ext}").exists() for ext in EXTENSIONS)]
    for qid in tqdm(query_ids, desc="Evaluating queries"):
        video_path = next((VIDEO_DIR / f"{qid}.{ext}" for ext in EXTENSIONS if (VIDEO_DIR / f"{qid}.{ext}").exists()), None)
        if video_path is None:
            continue
        t0 = time.perf_counter()
        query = tmkpy.hashVideo(str(video_path), "ffmpeg")
        q_path = str(DESC_DIR / f"{qid}.tmk")
        query.writeToOutputFile(q_path, "fivr_hashing")
        scores = query_batched(q_path, ref_tmk_files)
        t1 = time.perf_counter()
        similarities[qid] = {rid: float(s) for rid, s in zip(ref_ids_vid, scores)}
        detections[qid] = {rid: float(s) >= threshold for rid, s in zip(ref_ids_vid, scores)}
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
        for ref,score in similarities[q].items():
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
            "similarities": similarities,
            "detections": detections,
            "threshold": threshold,
            "task": args.mode,
            "db" :args.ds,
        }, f, indent=2)



if __name__ == "__main__":
    main()
