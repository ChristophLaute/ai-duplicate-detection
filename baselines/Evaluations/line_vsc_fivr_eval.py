import sys
import os
import json
import time
import faiss
import dataclasses
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import normalize

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
EXTENSIONS = ["mp4", "webm", "mkv","flv","mov","avi"]

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
    if not no_score_norm:
        feat = np.delete(vf.feature, low_var_dim, axis=1)
        feat = normalize(feat)
        sim, _ = score_norm_index.search(feat.astype(np.float32), 1)
        feat = np.concatenate([feat, -beta * sim[:, :1]], axis=1)
    else:
        feat = normalize(vf.feature)
    feat = feat * c
    return VideoFeature(video_id=vf.video_id, timestamps=vf.timestamps, feature=feat)
    

def compute_recall_at_k(annotations, results, dataset, relevant_labels, k):
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
    APs = []
    for query, gt_sets in annotations.items():
        if query not in results:
            continue
        query_gt = set(sum([gt_sets.get(l, []) for l in relevant_labels], []))
        query_gt = query_gt.intersection(database)
        if not query_gt:
            continue
        i, ri, s = 0.0, 0, 0.0
        for video, sim in sorted(results[query].items(), key=lambda x: x[1], reverse=True):
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
    parser.add_argument("--mode", type=str, default="all", choices=["all", "nd", "ds"])
    parser.add_argument("--no_consistency_weight", action="store_true", default=False, help="Disable consistency weight application to descriptors")
    parser.add_argument("--no_temporal_concat", action="store_true", default=False, help="Disable temporal concatenation of descriptors before PCA")
    parser.add_argument("--no_score_norm", action="store_true", default=False, help="Disable score normalization using noise reference stats")
    parser.add_argument("--no_tta", action="store_true", default=False, help="Disable test-time augmentation for query descriptor computation")
    parser.add_argument("--aggregation_type", type=str, default="max", choices=["max", "avg"], help="Type of score aggregation for candidate generation")
    parser.add_argument("--desc-dir", type=str,
    default=str(FIVR_DIR / "line_descriptors_postprocessed" / "50k"),
    help="Directory containing preprocessed descriptors and transform parameters")

    args = parser.parse_args()
    

    

    DESC_DIR = Path(args.desc_dir)
    VIDEO_DIR = FIVR_DIR / "videos"

    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr5k = json.load(f)

    #with open(DESC_DIR / "queries.npz", "rb") as f:
        #queries = load_features(f)
    with open(DESC_DIR / "references.npz", "rb") as f:
        refs = load_features(f)
    BASELINE_SRC = Path(__file__).resolve().parent.parent / "line_vsc/meta-vsc-descriptor-runtime/submission_src"
    os.chdir(BASELINE_SRC)
    device = get_device()
    model,preprocessor = create_model_in_runtime(transforms_device=device)
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
    online_times = []
    similarities = {}
    ti1 = time.perf_counter()
    if args.aggregation_type == "max":
        cg = CandidateGeneration(refs, MaxScoreAggregation())
    else:
        cg = CandidateGeneration(refs, AvgScoreAggregation())
    ti2 = time.perf_counter()
    print(f"Index built in {ti2 - ti1:.2f}s")
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
        similarities[q_desc.video_id] = {cand.ref_id: cand.score for cand in candidates}
        online_times.append(t1 - t0)

    print(f"Online: {sum(online_times):.2f}s total, {np.mean(online_times):.2f}s per query")
    with open(ANN_FILE) as f:
        annotations = json.load(f)
    all_refs = {vf.video_id for vf in refs}
    if args.ds == "fivr5k":
        db5k = {Path(v).stem for v in fivr5k["database"]}
        db = all_refs & db5k
    else:
        db = all_refs
    print(f"Evaluating on {len(similarities)} queries against {len(refs)} references")

    metrics = {
    "baseline": "line_vsc",
    "subset": args.ds,
    "index_build_time (offline)": round(ti2 - ti1, 3),
    "online_time_total": round(sum(online_times), 3),
    "online_time_per_query": round(np.mean(online_times), 4),
    "aggregation_type": args.aggregation_type,
    "no_consistency_weight": args.no_consistency_weight,
    "no_temporal_concat": args.no_temporal_concat,
    "no_score_norm": args.no_score_norm,
    "no_tta": args.no_tta,
    }
    for task, labels in [('DSVR', ['ND', 'DS']), ('CSVR', ['ND', 'DS', 'CS']), ('ISVR', ['ND', 'DS', 'CS', 'IS'])]:
        task_metrics = {"mAP": round(compute_mAP(annotations, similarities, db, labels), 4)}
        for k in [1, 5, 10, 20, 50, 100]:
            task_metrics[f"R@{k}"] = round(compute_recall_at_k(annotations, similarities, db, labels, k), 4)
        metrics[task] = task_metrics
        print(f"\n{task} mAP: {task_metrics['mAP']:.4f}")
        for k in [1, 5, 10, 20, 50, 100]:
            print(f"  R@{k}: {task_metrics[f'R@{k}']:.4f}")

    variant = Path(args.desc_dir).name
    out_path = Path(__file__).resolve().parent / f"results_line_vsc_{args.ds}_{variant}.json"


    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_path}")

if __name__ == "__main__":
    main()
