import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import fbeta_score
import tmkpy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "CompactSIamese"))

from runtime_config import FIVR_DIR



IDS_DIR  = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
DESC_DIR = FIVR_DIR / "tmk_descriptors"
ANN_FILE = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"

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




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="fivr5k")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "nd", "ds"])
    args = parser.parse_args()
    

    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr_5k = json.load(f)
    query_ids = load_id_list(IDS_DIR / "fivr_val_query_ids.txt")
    if args.ds == "fivr5k":
        ref_ids = [Path(v).stem for v in fivr_5k["database"]]
    else:
        with open(IDS_DIR / "fivr_reference_ids.txt") as f:
            ref_ids = [l.strip() for l in f if l.strip()]
   
    ref_tmk_files = [str(DESC_DIR / f"{rid}.tmk") for rid in ref_ids
                 if (DESC_DIR / f"{rid}.tmk").exists()]
    ref_ids_vid = [rid for rid in ref_ids
                if (DESC_DIR / f"{rid}.tmk").exists()]
    query_ids = [qid for qid in query_ids
                if (DESC_DIR / f"{qid}.tmk").exists()]
    print(f"Loaded {len(query_ids)} queries and {len(ref_ids_vid)} references")

    similarities = {}
    for qid in tqdm(query_ids, desc="Computing similarities"):
        q_path = str(DESC_DIR / f"{qid}.tmk")
        scores = query_batched(q_path, ref_tmk_files)
        similarities[qid] = {rid: float(s) for rid, s in zip(ref_ids_vid, scores)}


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
    
    
    order = np.argsort(scores) [::-1]  
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(bool)
    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(~sorted_labels)
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (np.sum(sorted_labels) + 1e-10)
    f1_beta = (1 + 0.5**2) * (precision * recall) / (0.5**2 * precision + recall + 1e-10) # fbeta is maximised and calculated directly from precision and recall
    best_index = np.argmax(f1_beta)
    best_threshold = sorted_scores[best_index] # the best score is the one that maximizes the F1-beta score
    best_f1_beta = f1_beta[best_index]
    best_accuracy = compute_accuracy((scores >= best_threshold).astype(int), labels)
    print(f"Best threshold: {best_threshold:.4f} with F1-beta: {best_f1_beta:.4f}, Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
