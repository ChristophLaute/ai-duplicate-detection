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

HDF5_FILE = FIVR_DIR / "HAP_Features/hap_features_all.h5"
MODEL_PATH = Path(__file__).resolve().parent.parent / "HAP-VR/ckpt/model.pth"
IDS_DIR    = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
ANN_FILE   = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"


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
    parser.add_argument("--ds", type=str, default="5k")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--features_only", action="store_true", default=False)
    args = parser.parse_args()

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
    query_loader = DataLoader(query_generator, batch_size=1,  num_workers=4)
    queries, query_ids_loaded = [], []
    offline_time = 0.0
    online_time  = 0.0

    # Query loop — all online (index_video per query)
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

    # Reference loop — index_video is offline; comparator is online
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

    print(f"Offline time (ref indexing): {offline_time:.2f}s")
    print(f"Online time (query indexing + comparator): {online_time:.2f}s | per query: {online_time/max(len(query_ids_loaded),1):.3f}s")
    with open(ANN_FILE) as f:
        annotations = json.load(f)
    db = set(ref_ids)
    print(f"Evaluating on {len(query_ids_loaded)} queries against {len(ref_ids)} references")

    metrics = {
        "baseline": "hap_vr",
        "subset": args.ds,
        "offline_time_total": round(offline_time, 3),
        "online_time_total": round(online_time, 3),
        "online_time_per_query": round(online_time / max(len(query_ids_loaded), 1), 4),
    }
    for task, labels in [('DSVR', ['ND', 'DS']), ('CSVR', ['ND', 'DS', 'CS']), ('ISVR', ['ND', 'DS', 'CS', 'IS'])]:
        task_metrics = {"mAP": round(compute_mAP(annotations, similarities, db, labels), 4)}
        for k in [1, 5, 10, 20, 50, 100]:
            task_metrics[f"R@{k}"] = round(compute_recall_at_k(annotations, similarities, db, labels, k), 4)
        metrics[task] = task_metrics
        print(f"\n{task} mAP: {task_metrics['mAP']:.4f}")
        for k in [1, 5, 10, 20, 50, 100]:
            print(f"  R@{k}: {task_metrics[f'R@{k}']:.4f}")

    out_path = Path(__file__).resolve().parent / f"results_hap_vr_{args.ds}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_path}")

if __name__ == "__main__":
    main()



