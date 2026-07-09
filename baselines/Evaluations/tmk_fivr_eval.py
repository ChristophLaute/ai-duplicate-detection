import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import tmkpy
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "CompactSIamese"))

from runtime_config import FIVR_DIR




IDS_DIR  = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
DESC_DIR = FIVR_DIR / "tmk_descriptors"
ANN_FILE = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"
VIDEO_DIR = FIVR_DIR / "videos"
EXTENSIONS = ["mp4", "webm", "mkv", "flv", "mov", "avi"]


def load_id_list(path):
    with open(path) as f:
        return [Path(line.strip()).stem for line in f if line.strip()]


def query_batched(q_path, ref_tmk_files, batch_size=5000):
    """
    Wrapper around tmkpy.query that processes references in chunks. Solves ram issues
    """
    all_scores = []
    for i in range(0, len(ref_tmk_files), batch_size):
        batch = ref_tmk_files[i:i + batch_size]
        all_scores.extend(tmkpy.query(q_path, batch, len(batch))) # compute similarities for each query video with all reference videos
    return all_scores


def compute_recall_at_k(annotations, results, dataset, relevant_labels, k):
    """Compute the recall at k for a set of query results against ground truth annotations."""

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


def compute_mAP(annotations, results, database, relevant_labels):
    """Compute the mean Average Precision (mAP) for a set of query results against ground truth annotations."""
    APs = []
    for query, gt_sets in annotations.items():
        if query not in results:
            continue
        query_gt = set(sum([gt_sets.get(l, []) for l in relevant_labels], []))
        query_gt = query_gt.intersection(database)
        if not query_gt:
            continue
        i, ri, s = 0.0, 0, 0.0
        for video, _ in sorted(results[query].items(), key=lambda x: x[1], reverse=True):
            if video != query and video in database:
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        APs.append(s / len(query_gt))
    return float(np.mean(APs)) if APs else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="fivr5k")
    args = parser.parse_args()
    

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
    query_ids = [qid for qid in query_ids if any((VIDEO_DIR / f"{qid}.{ext}").exists() for ext in EXTENSIONS)]

    t0 = time.perf_counter()
    for qid in tqdm(query_ids, desc="Evaluating queries"):
        video_path = next((VIDEO_DIR / f"{qid}.{ext}" for ext in EXTENSIONS if (VIDEO_DIR / f"{qid}.{ext}").exists()), None)
        if video_path is None:
            continue
        #query hash computation
        query = tmkpy.hashVideo(str(video_path), "ffmpeg")
        q_path = str(DESC_DIR / f"{qid}.tmk")
        query.writeToOutputFile(q_path,"fivr_hashing")
        scores = query_batched(q_path, ref_tmk_files) # compute similarities for each query video with all reference videos
        similarities[qid] = {rid: float(s) for rid, s in zip(ref_ids_vid, scores)}
    t1 = time.perf_counter()
    print(f"Online: {t1-t0:.2f}s total, {(t1-t0)/len(similarities):.4f}s per query ({len(similarities)} queries against {len(ref_ids_vid)} references)")
    with open(ANN_FILE) as f:
        annotations = json.load(f)
    db = set(ref_ids_vid)
    print(f"Evaluating on {len(query_ids)} queries against {len(ref_ids_vid)} references")

    metrics = {
        "baseline": "tmk",
        "subset": args.ds,
        "retrieval_time_total": round(t1 - t0, 3),
        "retrieval_time_per_query": round((t1 - t0) / len(query_ids), 4),
    }
    #metric computation loop for each task
    for task, labels in [('DSVR', ['ND', 'DS']), ('CSVR', ['ND', 'DS', 'CS']), ('ISVR', ['ND', 'DS', 'CS', 'IS'])]:
        task_metrics = {"mAP": round(compute_mAP(annotations, similarities, db, labels), 4)}
        for k in [1, 5, 10, 20, 50, 100]:
            task_metrics[f"R@{k}"] = round(compute_recall_at_k(annotations, similarities, db, labels, k), 4)
        metrics[task] = task_metrics
        print(f"\n{task} mAP: {task_metrics['mAP']:.4f}")
        for k in [1, 5, 10, 20, 50, 100]:
            print(f"  R@{k}: {task_metrics[f'R@{k}']:.4f}")

    out_path = Path(__file__).resolve().parent / f"results_tmk_{args.ds}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_path}")

if __name__ == "__main__":
    main()
