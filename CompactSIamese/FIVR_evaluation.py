import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import time
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))


from model.pytorch_model import SiameseNetwork
from retrieval.two_stage_retr_pred_head import whole_video_logit_from_descriptors
from retrieval.chamfer_retr import chamfer_sim
from runtime_config import FIVR_DIR, get_device
from retrieval.pool_embeddings import (
    pool_all_fragments,
    construct_faiss_index,
    load_index_and_metadata,
    video_level_retrieval,
)
from retrieval.chamfer_retr_faiss import retrieve_similar_videos_chamfer_faiss
from retrieval.two_stage_retr_pred_head import retrieve_similar_videos_pred_head
from data.precompute_fivr_descriptors import compute_descriptors




FIVR_DESCR_DIR = FIVR_DIR / "descriptors"
FIVR_INDEX_DIR = FIVR_DIR / "frame_index"
FIVR_INDEX_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATION_FILE = Path(__file__).parent / "dataset/annotation.json"
DATASET_IDS_FILE = Path(__file__).parent.parent.parent / "datasets/FIVR-200K/dataset/youtube_ids.txt"

RELEVANT_LABELS = ["ND", "DS"]
FIVR_POOL_DIR = FIVR_DIR / "pooled_embeddings"
VIDEO_DIR = FIVR_DIR / "videos"
VIDEO_EXTS = ["mp4", "webm", "mkv","flv","mov","avi"]



def load_descriptor(path):
    """Load a descriptor from a .npy file and return the 'descriptors' key."""
    obj = np.load(path, allow_pickle=True).item()
    return obj["descriptors"]


def compute_mAP(annotations, results, dataset, relevant_labels):
    """Compute mean Average Precision (mAP) for the given annotations and retrieval results."""
    mAP = []
    for query, gt_sets in annotations.items():
        if query not in results:
            continue
        query_gt = set(sum([gt_sets.get(l, []) for l in relevant_labels], []))
        query_gt = query_gt.intersection(dataset)
        if not query_gt:
            continue
        i, ri, s = 0.0, 0, 0.0
        for video in results[query]:          # results[query] is a ranked list
            if video != query and video in dataset:
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        mAP.append(s / len(query_gt))
    return float(np.mean(mAP)) if mAP else 0.0


def compute_recall_at_k(annotations, results, dataset, relevant_labels, k):
    """Compute Recall@k for the given annotations and retrieval results."""
    recalls = []
    for query, gt_sets in annotations.items():
        if query not in results:
            continue
        query_gt = set(sum([gt_sets.get(l, []) for l in relevant_labels], []))
        query_gt = query_gt.intersection(dataset)
        if not query_gt:
            continue
        top_k_videos = [video for video in results[query]
                        if video != query and video in dataset][:k]
        recall = len(set(top_k_videos).intersection(query_gt)) / len(query_gt)
        recalls.append(recall)
    return float(np.mean(recalls)) if recalls else 0.0


def compute_score(q_desc, db_desc, scoring, model=None, device=None):
    """Single-dispatch scoring function for brute-force siamese backend."""
    if scoring == "pred_chamfer":
        return whole_video_logit_from_descriptors(q_desc, db_desc, model, device, aggregation="chamfer")
    elif scoring == "pred_max":
        return whole_video_logit_from_descriptors(q_desc, db_desc, model, device, aggregation="max")
    elif scoring == "chamfer":
        q_frames = q_desc.reshape(-1, q_desc.shape[-1])
        db_frames = db_desc.reshape(-1, db_desc.shape[-1])
        return chamfer_sim(q_frames, db_frames)
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")


def main(args):
    SIM_JSON = Path(__file__).parent.parent / f"results_siamese_retrieval_{args.ds}_{args.scoring}_{args.backend}.json"
    device = get_device()
    model = None
    pool_index = None
    pool_metadata = None
    pool_vid_array = None
    # try different score aggregation methods for the siamese
    if args.scoring in ["pred_chamfer", "pred_max"] or args.backend in ["siamese", "faiss_siam"]:
        model = SiameseNetwork(T=args.T)
        model.load_state_dict(torch.load(args.model_path, map_location=get_device()))
        model = model.to(get_device())
        model.eval()
    # different similarity search backends
    if args.backend in ["faiss", "faiss_siam", "faiss_chamfer"]:
        if not (FIVR_POOL_DIR / "faiss_index.bin").exists():
            pool_all_fragments(descr_dir=FIVR_DESCR_DIR, save_dir=FIVR_POOL_DIR)
            construct_faiss_index(save_dir=FIVR_POOL_DIR)
            print("FAISS index constructed.")
        pool_index, pool_metadata, pool_vid_array = load_index_and_metadata(save_dir=FIVR_POOL_DIR)
        print("Index and metadata loaded.")

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
    # cache all database descriptors for faster retrieval
    descriptor_cache = {}
    for video_id in tqdm(db_ids, desc="Loading descriptors"):
        descriptor_path = args.descriptors_dir / f"{video_id}.npy"
        if descriptor_path.exists():
            descriptor = load_descriptor(descriptor_path)
            if descriptor is not None:
                descriptor_cache[video_id] = descriptor
    print(f"Loaded {len(descriptor_cache)} descriptors out of {len(db_ids)} database videos.")

    results = {}
    similarities = {query_id: {} for query_id in query_ids}
    retrieval_times = []

    for query_id in tqdm(query_ids, desc="Evaluating queries"):
        video_path = next((VIDEO_DIR / f"{query_id}.{ext}" for ext in VIDEO_EXTS if (VIDEO_DIR / f"{query_id}.{ext}").exists()), None)

        if video_path is None:
            continue
        t0 = time.perf_counter()
        query_descriptor = compute_descriptors(video_path, model, device, T=args.T, target_size=tuple(args.target_size))

        query_path = args.descriptors_dir / f"{query_id}.npy"
        if query_descriptor is False or query_descriptor is None:
            print(f"Warning: Could not compute descriptor for query {query_id}, skipping")
            continue
        

        if args.backend == "siamese":
            scores = {}
            for video_id, db_descriptor in descriptor_cache.items():
                if video_id == query_id:
                    continue
                scores[video_id] = compute_score(query_descriptor, db_descriptor, args.scoring, model, get_device())
                similarities[query_id][video_id] = scores[video_id]

            ranked = sorted(scores, key=scores.get, reverse=True)

        elif args.backend == "faiss":
            result = video_level_retrieval(
                query_id, top_k_per_fragment=pool_index.ntotal,
                index=pool_index, metadata=pool_metadata, video_id_array=pool_vid_array)
            ranked = [r["video_id"] for r in result if r["video_id"] != query_id]

        elif args.backend == "faiss_chamfer":
            ranked = retrieve_similar_videos_chamfer_faiss(
                query_path, descriptor_cache,
                top_k_videos=args.top_k_rerank,
                top_k_per_fragment=pool_index.ntotal,
                index=pool_index, metadata=pool_metadata, video_id_array=pool_vid_array,
                descr_dir=FIVR_DESCR_DIR, computed_descriptor=query_descriptor)
            ranked = [v for v in ranked if v != query_id]

        elif args.backend == "faiss_siam":
            agg = "chamfer" if args.scoring == "pred_chamfer" else "max"
            ranked = retrieve_similar_videos_pred_head(
                query_path, descriptor_cache, model, get_device(),
                top_k_videos=args.top_k_rerank,
                top_k_per_fragment=pool_index.ntotal,
                index=pool_index, metadata=pool_metadata, video_id_array=pool_vid_array,
                descr_dir=FIVR_DESCR_DIR, aggregation=agg,
                computed_descriptor=query_descriptor)
            ranked = [v for v in ranked if v != query_id]

        retrieval_times.append(time.perf_counter() - t0)
        results[query_id] = ranked

    avg_retrieval_time = float(np.mean(retrieval_times))
    metrics = {
        "baseline": "siamese",
        "subset": args.ds,
        "online_time_per_query": round(avg_retrieval_time, 4),
        "online_time_total": round(sum(retrieval_times), 3),
        "queries_evaluated": len(results),
        "scoring_method": args.scoring,
        "backend": args.backend,
    }
    for task, labels in [('DSVR', ['ND', 'DS']), ('CSVR', ['ND', 'DS', 'CS']), ('ISVR', ['ND', 'DS', 'CS', 'IS'])]:
        task_metrics = {"mAP": round(compute_mAP(annotations, results, db_ids, labels), 4)}
        for k in [1, 5, 10, 20, 50, 100]:
            task_metrics[f"R@{k}"] = round(compute_recall_at_k(annotations, results, db_ids, labels, k), 4)
        metrics[task] = task_metrics
        print(f"\n{task} mAP: {task_metrics['mAP']:.4f}")
        for k in [1, 5, 10, 20, 50, 100]:
            print(f"  R@{k}: {task_metrics[f'R@{k}']:.4f}")

    print(f"Online: {sum(retrieval_times):.2f}s total, {avg_retrieval_time:.4f}s per query")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"results_siamese_{args.ds}_{args.backend}_{args.scoring}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {out_path}")
    with open(SIM_JSON, "w") as f:
        json.dump({
            "similarities": similarities,
            "db": args.ds,
            "scoring": args.scoring,
            "backend": args.backend,
        }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=Path("best_siamese_model.pth"))
    parser.add_argument("--fivr_vid_json", type=Path, default=Path("fivr_subset_50k/fivr5k_vid.json"))
    parser.add_argument("--annotation", type=Path, default=ANNOTATION_FILE)
    parser.add_argument("--descriptors_dir", type=Path, default=FIVR_DIR / "descriptors")
    parser.add_argument("--output_dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--ds", type=str, default="5k", choices=["5k", "50k"])
    parser.add_argument("--backend", choices=["faiss", "faiss_chamfer", "siamese", "faiss_siam"], default="siamese")
    parser.add_argument("--scoring", choices=["pred_chamfer", "pred_max", "chamfer"], default="pred_max")
    parser.add_argument("--top_k_rerank", type=int, default=50)
    parser.add_argument("--T", type=int, default=15, help="Frames per fragment (must match training)")
    parser.add_argument("--target_size", type=int, nargs=2, default=[32, 32], help="Frame resolution (must match training)")

    args = parser.parse_args()
    main(args)
