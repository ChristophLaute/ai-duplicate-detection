import sys
import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "CompactSIamese"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "VVS"))

from runtime_config import FIVR_DIR

from extract_fivr_subset import extract_features
from model.feature_extractor import FeatureExtractor
from model.model import SuppressionWeightGenerationStage, EasyDistractorEliminationStage
from arg import arg_func



IDS_DIR  = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"
DESC_DIR = FIVR_DIR / "vvs_video_descriptors"
ANN_FILE = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/dataset/annotation.json"
VIDEO_DIR = FIVR_DIR / "videos"
CHECKPOINT = Path(__file__).resolve().parent.parent / "VVS/jobs/table_benchmark_dim_3840/model/m00080000.pth"


def find_video_path(video_id):
    video_path = None
    for ext in [".mp4", ".avi", ".mkv"]:
        p = VIDEO_DIR / f"{video_id}{ext}"
        if p.exists():
            video_path = p
            return video_path

    
    return None
def compute_query_descriptor(video_path,device, feature_extractor, tsm_tgm_model, ddm_model):
    stem = Path(video_path).stem
    frame_feats = extract_features(video_path, feature_extractor, device)
    if frame_feats is None:
        return None
    frame_feats = frame_feats.cuda().unsqueeze(0)
    with torch.no_grad():
            in_data = {"anchor": frame_feats}
            
            ddm_out = ddm_model(in_data["anchor"])
            if (torch.sigmoid(ddm_out["confidence"]) > 0.5).float().sum() != 0:
                    mask = (torch.sigmoid(ddm_out["confidence"]) > 0.5)
                    in_data["anchor"] = in_data["anchor"][:,mask]
                
            while in_data["anchor"].shape[1] < 4:
                in_data["anchor"] = torch.cat([in_data["anchor"], in_data["anchor"]], dim=1)

            feats_out = tsm_tgm_model(in_data, is_anc_processed = False)
            return feats_out["features"].squeeze().cpu().numpy()
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

def load_descriptor(vid):
    p = DESC_DIR / f"{Path(vid).stem}.pt"
    if not p.exists():
        return None
    return torch.load(p).squeeze().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="fivr5k")
    args = parser.parse_args()
    vvs_args = argparse.Namespace(
    suppression="vvs", vvs_vinit="v_t", vvs_h_connect=True,
    vvs_sigmoid_T=512.0, vvs_sigmoid_T_tsm=1.0, vvs_sigmoid_T_ddm=1.0,
    vvs_tsm=True, vvs_tgm=True, vvs_ddm=True, refinement=True,
    base_normalize=True, pca_whitening=True, pca_reduction=None,
    gpu=0, feature_extract="resnet50_l4imac", feature_backbone="resnet50",
    feature_type="l4imac", retrieve_triplet=False, iterations=-1,
    cycles=1, imple_flag="eval", weight_triplet=1.0, weight_saliency=1.0,
    weight_frame=1.0, distractor_sampling_ratio="20_50",
    mag_opt=5, mag_thresh=40, neg_len=32, thresh_s=0.5,
    baseline=False, use_oracle=False, fg_gt_label=None,
)

    VVS_DIR = Path(__file__).resolve().parent.parent / "VVS"
    os.chdir(VVS_DIR)
    loaded = torch.load(CHECKPOINT, map_location="cpu")
    feature_extractor = FeatureExtractor(network="resnet50").to(device).eval()
    tsm_tgm_model = SuppressionWeightGenerationStage(vvs_args).cuda().eval()
    ddm_model = EasyDistractorEliminationStage(vvs_args).cuda().eval()
    tsm_tgm_model.load_state_dict(loaded["tsm_tgm_model"], strict=False)
    ddm_model.load_state_dict(loaded["ddm_model"], strict=False)
    os.chdir(Path(__file__).parent)
        

    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr_5k = json.load(f)
    if args.ds == "fivr5k":
        ref_ids = [Path(v).stem for v in fivr_5k["database"]]
        query_ids = [Path(v).stem for v in fivr_5k["query"]]
    else:
        with open(IDS_DIR / "fivr_reference_ids.txt") as f:
            ref_ids = [l.strip() for l in f if l.strip()]
        query_ids = [Path(v).stem for v in fivr_5k["query"]]
        
    #queries = {q:load_descriptor(q) for q in query_ids}
    refs = {r:load_descriptor(r) for r in ref_ids}
    #queries = {k:v for k,v in queries.items() if v is not None}
    refs = {k:v for k,v in refs.items() if v is not None}
    #print(f"Loaded {len(queries)} queries and {len(refs)} references")
    ref_ids_keys = list(refs.keys())
    ref_matrix = np.stack([refs[r] for r in ref_ids_keys])
    t0 =time.perf_counter()
    similarities = {}
    for qid in tqdm(query_ids, desc="Computing similarities"):
        video_path = find_video_path(qid)
        if video_path is None:
            continue
        qdesc = compute_query_descriptor(video_path, device, feature_extractor, tsm_tgm_model, ddm_model)
        if qdesc is None:
            print(f"Warning: Could not compute descriptor for query {qid}, skipping")
            continue
        sim = ref_matrix @ qdesc
        similarities[qid] = {rid: float(s) for rid, s in zip(ref_ids_keys, sim)}
    t1 = time.perf_counter()
    print(f"Online: {t1-t0:.2f}s total, {(t1-t0)/len(similarities):.4f}s per query")
    with open(ANN_FILE) as f:
        annotations = json.load(f)
    db = set(refs)
    #print(f"Evaluating on {len(queries)} queries against {len(refs)} references")

    metrics = {
        "baseline": "vvs",
        "subset": args.ds,
        "online_time_total": round(t1 - t0, 3),
        "online_time_per_query": round((t1 - t0) / len(similarities), 4),
    }
    for task, labels in [('DSVR', ['ND', 'DS']), ('CSVR', ['ND', 'DS', 'CS']), ('ISVR', ['ND', 'DS', 'CS', 'IS'])]:
        task_metrics = {"mAP": round(compute_mAP(annotations, similarities, db, labels), 4)}
        for k in [1, 5, 10, 20, 50, 100]:
            task_metrics[f"R@{k}"] = round(compute_recall_at_k(annotations, similarities, db, labels, k), 4)
        metrics[task] = task_metrics
        print(f"\n{task} mAP: {task_metrics['mAP']:.4f}")
        for k in [1, 5, 10, 20, 50, 100]:
            print(f"  R@{k}: {task_metrics[f'R@{k}']:.4f}")

    out_path = Path(__file__).resolve().parent / f"results_vvs_{args.ds}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_path}")

if __name__ == "__main__":
    main()