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

from tune_threshold_hap_ddp import tune_thresholds, select_frames
from HAP_DDP_retr import compute_recall_at_k, compute_mAP


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


def evaluate_pairs(pairs, threshold):
    """Evaluate a set of (score, label) pairs against a threshold, computing accuracy, F1-beta, F1, precision, and recall."""
    scores = np.array([s for s, _ in pairs])
    labels = np.array([l for _, l in pairs])
    predictions = scores >= threshold
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(compute_accuracy(predictions, labels)),
        "f1_beta": float(compute_f1_beta(predictions, labels, beta=0.5)),
        "f1": float(f1_score(labels, predictions, average='binary')),
        "precision": float(precision_score(labels, predictions, average='binary')),
        "recall": float(recall_score(labels, predictions, average='binary')),
    }
    print(", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="5k", help="Reference set for the main eval pass (detection + retrieval)")
    parser.add_argument("--tune_ds", type=str, default="50k", help="Reference set used for threshold tuning")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--energy_threshold", type=float, default=1e-7, help="Energy threshold for DPP frame selection")
    parser.add_argument("--methods", type=str, nargs="+", default=["dpp", "interframe_diff", "uniform"],
        choices=["dpp", "interframe_diff", "uniform"], help="Frame selection methods to sweep over")
    parser.add_argument("--kdividers", type=int, nargs="+", default=[2, 4, 8, 16],
        help="max_k_divider values to sweep over")
    args = parser.parse_args()
    sim_args = argparse.Namespace(features_only=False, gpu_id=args.gpu_id)

    m = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    sim_network = SimilarityNetwork[m["args"].similarity_network].get_model(**vars(m["args"]))
    sim_network.load_state_dict(m["model"])
    sim_network = sim_network.cuda().eval()

    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr_5k = json.load(f)
    with open(ANN_FILE) as f:
        annotations = json.load(f)

    query_ids = [Path(v).stem for v in fivr_5k["query"]]
    if args.ds == "5k":
        ref_ids = [Path(v).stem for v in fivr_5k["database"]]
    else:
        with open(IDS_DIR / "fivr_reference_ids.txt") as f:
            ref_ids = [l.strip() for l in f if l.strip()]
    db = set(ref_ids)

    all_results = []

    for method in args.methods:
        for k in args.kdividers:
            print(f"\n===== method={method} max_k_divider={k} =====")

        
            threshold_all, threshold_nd = tune_thresholds(
                sim_network, method, k, args.energy_threshold, args.tune_ds, args.gpu_id
            )


            offline_time = 0.0
            online_time = 0.0

            query_generator = HDF5DatasetGenerator(str(HDF5_FILE), query_ids)
            query_loader = DataLoader(query_generator, batch_size=1, num_workers=4)
            queries, query_ids_loaded = [], []
            for (video_tensor,), (video_id,) in tqdm(query_loader, desc=f"[{method} k={k}] queries"):
                if video_id and video_tensor.shape[0] > 0:
                    t = time.perf_counter()
                    selected = select_frames(video_tensor.squeeze(), video_tensor, method, k, args.energy_threshold)
                    if len(selected) < 2:
                        if video_tensor.shape[0] < 2:
                            continue
                        selected = list(range(video_tensor.shape[0]))
                    feat = sim_network.index_video(video_tensor[selected].cuda())
                    online_time += time.perf_counter() - t
                    queries.append(feat)
                    query_ids_loaded.append(video_id)

            ref_gen = HDF5DatasetGenerator(str(HDF5_FILE), ref_ids)
            ref_loader = DataLoader(ref_gen, batch_size=1, num_workers=4)
            similarities = {q: {} for q in query_ids_loaded}
            for (video_tensor,), (video_id,) in tqdm(ref_loader, desc=f"[{method} k={k}] refs"):
                if video_id and video_tensor.shape[0] > 0:
                    t_off = time.perf_counter()
                    selected = select_frames(video_tensor.squeeze(), video_tensor, method, k, args.energy_threshold)
                    if len(selected) < 2:
                        if video_tensor.shape[0] < 2:
                            continue
                        selected = list(range(video_tensor.shape[0]))
                    feat = sim_network.index_video(video_tensor[selected].cuda())
                    offline_time += time.perf_counter() - t_off

                    t_on = time.perf_counter()
                    sims = calculate_similarities_to_queries(sim_network, queries, feat, sim_args)
                    online_time += time.perf_counter() - t_on
                    for i, qid in enumerate(query_ids_loaded):
                        similarities[qid][video_id] = float(sims[i])

            print(f"Offline time (frame sel + ref indexing): {offline_time:.2f}s")
            print(f"Online time (query indexing + comparator): {online_time:.2f}s | per query: {online_time / max(len(query_ids_loaded), 1):.3f}s")

            # --- detection metrics (using thresholds tuned for this config) ---
            detection = {}
            for mode, relevant_labels, threshold in [("all", ["ND", "DS"], threshold_all), ("nd", ["ND"], threshold_nd)]:
                if threshold is None:
                    continue
                pairs = []
                for q in similarities:
                    gt = set()
                    for l in relevant_labels:
                        gt.update(annotations[q].get(l, []))
                    exclude = set(annotations[q].get("DS", [])) if mode == "nd" else set()
                    for ref, score in similarities[q].items():
                        if ref != q and ref not in exclude:
                            pairs.append((score, ref in gt))
                print(f"[detection {mode}]", end=" ")
                detection[mode] = evaluate_pairs(pairs, threshold)

    
            retrieval = {}
            for task, labels in [('DSVR', ['ND', 'DS']), ('CSVR', ['ND', 'DS', 'CS']), ('ISVR', ['ND', 'DS', 'CS', 'IS'])]:
                task_metrics = {"mAP": round(compute_mAP(annotations, similarities, db, labels), 4)}
                for kk in [1, 5, 10, 20, 50, 100]:
                    task_metrics[f"R@{kk}"] = round(compute_recall_at_k(annotations, similarities, db, labels, kk), 4)
                retrieval[task] = task_metrics
                print(f"[retrieval {task}] mAP={task_metrics['mAP']:.4f} " +
                      " ".join(f"R@{kk}={task_metrics[f'R@{kk}']:.4f}" for kk in [1, 5, 10, 20, 50, 100]))

            result = {
                "method": method,
                "max_k_divider": k,
                "ds": args.ds,
                "tune_ds": args.tune_ds,
                "threshold_all": threshold_all,
                "threshold_nd": threshold_nd,
                "offline_time_total": round(offline_time, 3),
                "online_time_total": round(online_time, 3),
                "online_time_per_query": round(online_time / max(len(query_ids_loaded), 1), 4),
                "detection": detection,
                "retrieval": retrieval,
            }
            all_results.append(result)

            out_path = Path(__file__).resolve().parent / f"results_sweep_{method}_k{k}_{args.ds}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Saved to {out_path}")

    combined_path = Path(__file__).resolve().parent / f"results_sweep_all_{args.ds}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results to {combined_path}")


if __name__ == "__main__":
    main()
