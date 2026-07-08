#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from data.dataset_segment_pairs import VCDBSegmentPairDataset
from model.metrics import calculate_metrics, mean_bce_with_logits,average_precision_at_k
from model.pytorch_model import SiameseNetwork
from runtime_config import VCDB_DIR, get_device, resolve_video_path, video_path_to_id
from train import validate, validate_per_pair_type
from data.whole_video_dataset_creation import construct_dataloaders
from retrieval.pool_embeddings import (
    pool_all_fragments,
    construct_faiss_index,
    video_level_retrieval,
)
from retrieval.chamfer_retr import (
    construct_video_db,
    retrieve_similar_videos_chamfer,
)
from retrieval.chamfer_retr_faiss import (
    retrieve_similar_videos_chamfer_faiss,
    INDEX_PATH
)
from utils.seeds import set_seed
from retrieval.two_stage_retr_pred_head import whole_video_logit_from_descriptors, retrieve_similar_videos_pred_head, retrieve_similar_videos_pred_only  
from retrieval.pool_embeddings import load_index_and_metadata



device = get_device()
DESCRIPTORS_DIR = VCDB_DIR / "descriptors"
ANNOTATION_DIR = VCDB_DIR / "annotation"

DEFAULT_MODEL_PATH = Path("best_siamese_model.pth")
DEFAULT_SEGMENT_TEST_CSV = Path("test_pairs_segment.csv")
DEFAULT_VIDEO_LEVEL_TEST_CSV = Path("test_pairs_video_level.csv")
DEFAULT_OUTPUT_ROOT = Path("evaluation_results")
SAVE_DIR = VCDB_DIR / "pooled_embeddings"


def to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def make_output_dir(root, mode, retrieval_backend = None):
    """Creates unique output directory per run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{retrieval_backend}" if mode == "retrieval" and retrieval_backend is not None else ""
    out_dir = root / f"{mode}{suffix}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def subsample_pairs(df,max_pairs = None,balanced_pairs = False,sample_seed = 42) :
    """Subsamples a dataframe of video pairs with a balanced and random sampled mode"""
    if max_pairs is None or max_pairs <= 0 or len(df) <= max_pairs:
        return df

    if not balanced_pairs:
        return df.sample(n=max_pairs, random_state=sample_seed).reset_index(drop=True)
    
    #positive and negative samples 
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]

    # target is to have half
    target_each = max_pairs // 2
    n_pos = min(len(pos_df), target_each)
    n_neg = min(len(neg_df), target_each)
    # samples from each half by taking classes into account
    sampled_pos = pos_df.sample(n=n_pos, random_state=sample_seed)
    sampled_neg = neg_df.sample(n=n_neg, random_state=sample_seed)
    sampled = pd.concat([sampled_pos, sampled_neg], axis=0)
    # if there is still not enough samples to fill the target 
    remaining = max_pairs - len(sampled)
    if remaining > 0:
        print(f"Warning: Not enough samples to reach target of {max_pairs} with balanced sampling. Sampled {len(sampled)} pairs. Sampling additional {remaining} pairs randomly from the whole dataset.")

    return sampled.sample(frac=1.0, random_state=sample_seed).reset_index(drop=True)


def load_descriptor_file(descriptor_path):
    obj = np.load(descriptor_path, allow_pickle=True).item()
    if "descriptors" not in obj:
        raise KeyError(f"'descriptors' key not found in {descriptor_path}")
    return obj["descriptors"]


def build_descriptor_index():
    """creates difctionary of descriptors to load at the start"""
    index = defaultdict(list)
    for p in DESCRIPTORS_DIR.rglob("*.npy"):
        rel_id = p.relative_to(DESCRIPTORS_DIR).with_suffix("").as_posix()
        index[rel_id].append(p)
    return index


def resolve_descriptor_path(descriptor_index, video_id):
    """flexible resolution of video_id to descriptor file path, with some tolerance for mismatches
    """
    candidates = descriptor_index.get(video_id, [])
    # if no exact match tries just matching the stem of the video and searching for
    if len(candidates) == 0:
        stem_matches = [
            path
            for rel_id, paths in descriptor_index.items()
            if Path(rel_id).name == video_id
            for path in paths
        ]
        if len(stem_matches) == 1:
            return stem_matches[0]
        return None
    if len(candidates) > 1:
        print(f"Warning: multiple descriptor files found for '{video_id}', using first: {candidates[0]}")
    return candidates[0]










    


def validate_whole_video_pairs_from_descriptors(model, test_csv):
    """evaluates whole video classification from precomputed descriptors by comparing all fragment pairs and taking the maximum logit as the final score for the video pair"""
    all_labels = []
    all_logits = []
    pair_rows = []

    descriptor_index = build_descriptor_index()
    unique_pairs = test_csv[["video1", "video2", "label"]].drop_duplicates()

    print(f"Unique whole-video pairs to evaluate: {len(unique_pairs)}")
    print(f"Indexed descriptor stem keys: {len(descriptor_index)}")

    missing_rows = []
    empty_rows = []

    for _, row in tqdm(
        unique_pairs.iterrows(),
        total=len(unique_pairs),
        desc="Whole-video descriptor eval",
    ):
        video1_id = row["video1"]
        video2_id = row["video2"]
        label = int(row["label"])

        path1 = resolve_descriptor_path(descriptor_index, video1_id)
        path2 = resolve_descriptor_path(descriptor_index, video2_id)

        if path1 is None or path2 is None:
            missing_rows.append(
                {
                    "video1": video1_id,
                    "video2": video2_id,
                    "label": label,
                    "path1_found": path1 is not None,
                    "path2_found": path2 is not None,
                }
            )
            continue

        frags1 = load_descriptor_file(path1)
        frags2 = load_descriptor_file(path2)

        if len(frags1) == 0 or len(frags2) == 0:
            empty_rows.append(
                {
                    "video1": video1_id,
                    "video2": video2_id,
                    "label": label,
                    "len_frags1": len(frags1),
                    "len_frags2": len(frags2),
                }
            )
            continue
        t0 = time.perf_counter()
        logit = whole_video_logit_from_descriptors(frags1, frags2, model, device = get_device())
        inference_duration = time.perf_counter() - t0
        pred = 1 if logit >= 0.0 else 0

        all_labels.append(label)
        all_logits.append(float(logit))

        pair_rows.append(
            {
                "video1": video1_id,
                "video2": video2_id,
                "label": label,
                "logit": float(logit),
                "pred": pred,
                "correct": int(pred == label),
                "latency_ms": inference_duration * 1000,
            }
        )

    print(f"Pairs skipped because descriptor file missing: {len(missing_rows)}")
    print(f"Pairs skipped because descriptor arrays were empty: {len(empty_rows)}")

    if len(all_labels) == 0:
        raise ValueError(
            "No valid video pairs were evaluated. "
            "Check descriptor indexing, saved descriptor files, and CSV video IDs."
        )

    all_labels_t = torch.tensor(all_labels, dtype=torch.float32)
    all_logits_t = torch.tensor(all_logits, dtype=torch.float32)
    accuracy, precision, recall, f1,fbeta = calculate_metrics(all_labels_t, all_logits_t)
    latency_mean = np.mean([row["latency_ms"] for row in pair_rows]) if pair_rows else None
    latency_median = np.median([row["latency_ms"] for row in pair_rows]) if pair_rows else None
    latency_p95 = np.percentile([row["latency_ms"] for row in pair_rows], 95) if pair_rows else None
    loss = mean_bce_with_logits(all_labels_t, all_logits_t)

    metrics = {
        "loss": to_float(loss),
        "accuracy": to_float(accuracy),
        "precision": to_float(precision),
        "recall": to_float(recall),
        "f1": to_float(f1),
        "fbeta": to_float(fbeta),
        "num_evaluated_pairs": len(pair_rows),
        "num_missing_pairs": len(missing_rows),
        "num_empty_pairs": len(empty_rows),
        "latency_mean": to_float(latency_mean),
        "latency_median": to_float(latency_median),
        "latency_p95": to_float(latency_p95),

    }

    return metrics, pd.DataFrame(pair_rows), pd.DataFrame(missing_rows), pd.DataFrame(empty_rows)


def is_duplicate_in_fragment_sequences(frags1, frags2, model, threshold_logit=0.0, batch_size=128):
    """computes whether there is at least one duplicate fragment pair between two sequences of fragments by comparing all pairs and taking the maximum logit"""
    pairs = [(f1, f2) for f1 in frags1 for f2 in frags2]
    outputs = []
    with torch.no_grad():
        for i in range (0,len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            f1_batch = torch.stack([b[0] for b in batch]).to(device)
            f2_batch = torch.stack([b[1] for b in batch]).to(device)
            out,_,_= model(f1_batch, f2_batch)
            outputs.extend(out.reshape(-1).tolist())


    max_output = max(outputs) if outputs else None
    max_prob = torch.sigmoid(torch.tensor(max_output)).item() if max_output is not None else None
    is_dup = max_output >= threshold_logit if max_output is not None else False
    
    return is_dup, max_output, max_prob


def validate_whole_video_pairs_from_fragments(model, test_loader):
    """"evaluates whole video classification from fragments without precomputing descriptors by comparing all fragment pairs and taking the maximum logit as the final score for the video pair"""
    all_labels = []
    all_logits = []
    pair_rows = []
    empty_rows = []

    for batch in tqdm(test_loader, desc="Whole-video fragment eval"):
        frags1, frags2, label, video1_id, video2_id = batch[0]

        if len(frags1) == 0 or len(frags2) == 0:
            empty_rows.append(
                {
                    "video1": video1_id,
                    "video2": video2_id,
                    "label": int(label.item()),
                    "len_frags1": len(frags1),
                    "len_frags2": len(frags2),
                }
            )
            continue

        label = label.item()
        t0 = time.perf_counter()
        is_duplicate, max_output, max_output_sigmoid = is_duplicate_in_fragment_sequences(frags1, frags2, model)
        inference_duration = time.perf_counter() - t0
        all_labels.append(float(label))
        all_logits.append(float(max_output))

        pair_rows.append(
            {
                "video1": video1_id,
                "video2": video2_id,
                "label": int(label),
                "max_output": None if max_output is None else float(max_output),
                "max_output_sigmoid": None if max_output_sigmoid is None else float(max_output_sigmoid),
                "pred": int(is_duplicate),
                "correct": int(int(is_duplicate) == int(label)),
                "latency_ms": inference_duration * 1000,
            }
        )

    if len(all_labels) == 0:
        raise ValueError("No valid fragment-based whole-video pairs were evaluated.")

    all_labels_t = torch.tensor(all_labels, dtype=torch.float32)
    all_logits_t = torch.tensor(all_logits, dtype=torch.float32)
    accuracy, precision, recall, f1, fbeta = calculate_metrics(all_labels_t, all_logits_t)
    loss = mean_bce_with_logits(all_labels_t, all_logits_t)
    latency_mean = np.mean([row["latency_ms"] for row in pair_rows]) if pair_rows else None
    latency_median = np.median([row["latency_ms"] for row in pair_rows]) if pair_rows else None
    latency_p95 = np.percentile([row["latency_ms"] for row in pair_rows], 95) if pair_rows else None

    metrics = {
        "loss": to_float(loss),
        "accuracy": to_float(accuracy),
        "precision": to_float(precision),
        "recall": to_float(recall),
        "f1": to_float(f1),
        "fbeta": to_float(fbeta),
        "num_evaluated_pairs": len(pair_rows),
        "num_empty_pairs": len(empty_rows),
        "latency_mean": latency_mean,
        "latency_median": latency_median,
        "latency_p95": latency_p95,

    }

    return metrics, pd.DataFrame(pair_rows), pd.DataFrame(empty_rows)


def iter_vcdb_annotation_pairs():
    """builds an iterator over all annotated positive pairs in the VCDB annotation files, yielding tuples of (video1_id, video2_id)"""
    if not ANNOTATION_DIR.exists():
        raise FileNotFoundError(f"VCDB annotation directory not found: {ANNOTATION_DIR}")

    for ann_file in sorted(ANNOTATION_DIR.glob("*.txt")):
        # Skip AppleDouble metadata files that can appear on macOS volumes.
        if ann_file.name.startswith("._"):
            continue

        with open(ann_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue

                p1 = resolve_video_path(row[0].strip())
                p2 = resolve_video_path(row[1].strip())
                if p1 is None or p2 is None:
                    continue
                v1 = video_path_to_id(p1)
                v2 = video_path_to_id(p2)
                if not v1 or not v2 or v1 == v2:
                    continue

                yield v1, v2


def build_retrieval_ground_truth(test_csv):
    query_videos = set(test_csv["video1"]).union(set(test_csv["video2"]))
    indexed_videos = set(build_descriptor_index().keys())

    gt = defaultdict(set)
    all_positive_pairs = set()
    indexed_positive_pairs = set()

    for v1, v2 in iter_vcdb_annotation_pairs():
        pair = tuple(sorted((v1, v2)))
        if pair in all_positive_pairs:
            continue
        all_positive_pairs.add(pair)

        if v1 not in indexed_videos or v2 not in indexed_videos:
            continue

        indexed_positive_pairs.add(pair)
        gt[v1].add(v2)
        gt[v2].add(v1)

    csv_positive_pairs = {
        tuple(sorted((v1, v2)))
        for v1, v2 in test_csv[test_csv["label"] == 1][["video1", "video2"]].itertuples(index=False, name=None)
    }
    query_videos = sorted(v for v in query_videos if len(gt.get(v, set())) > 0)
    missing_csv_positives = indexed_positive_pairs - csv_positive_pairs

    print(f"VCDB annotation positive pairs (all): {len(all_positive_pairs)}")
    print(f"VCDB annotation positive pairs within indexed corpus: {len(indexed_positive_pairs)}")
    print(f"Positive pairs in CSV: {len(csv_positive_pairs)}")
    print(f"Annotation positives missing from CSV positives: {len(missing_csv_positives)}")
    print(f"Queries with at least one annotated positive: {len(query_videos)}")

    return gt, query_videos





def evaluate_retrieval_faiss(test_csv, top_k_per_fragment=None, ranking_save_k=100, max_queries=None):
    pool_all_fragments()
    construct_faiss_index()
    
    pool_index, pool_metadata, pool_vid_array = load_index_and_metadata()
    top_k_per_fragment = top_k_per_fragment if top_k_per_fragment is not None else pool_index.ntotal

    gt, query_videos = build_retrieval_ground_truth(test_csv)
    if max_queries is not None and max_queries > 0:
        query_videos = query_videos[:max_queries]

    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    ap_scores = []

    query_rows = []
    ranking_rows = []
    empty_query_rows = []

    print(f"Queries to evaluate: {len(query_videos)}")
    print("Example GT query videos:", query_videos[:10])

    pooled_ids = sorted({p.stem.rsplit("__frag", 1)[0] for p in (VCDB_DIR / "pooled_embeddings").glob("*.npy")})
    print("Number of pooled video IDs:", len(pooled_ids))
    print("Example pooled video IDs:", pooled_ids[:10])

    for query_video in tqdm(query_videos, desc="FAISS retrieval eval"):
        relevant = gt[query_video]
        if len(relevant) == 0:
            continue
        t0 = time.perf_counter()
        ranked = video_level_retrieval(
            query_video_id=query_video,
            top_k_per_fragment=top_k_per_fragment,
            index=pool_index,
            metadata=pool_metadata,
            video_id_array=pool_vid_array,
        )
        retrieval_duration = time.perf_counter() - t0

        ranked_ids = [r["video_id"] for r in ranked]

        if len(ranked_ids) == 0:
            empty_query_rows.append(
                {
                    "query_video": query_video,
                    "num_relevant": len(relevant),

                }
            )
            continue

        r1 = int(any(v in relevant for v in ranked_ids[:1]))
        r5 = int(any(v in relevant for v in ranked_ids[:5]))
        r10 = int(any(v in relevant for v in ranked_ids[:10]))
        ap = average_precision_at_k(ranked_ids, relevant, k=None)

        recall_at_1.append(r1)
        recall_at_5.append(r5)
        recall_at_10.append(r10)
        if ap is not None:
            ap_scores.append(ap)

        first_relevant_rank = None
        for i, vid in enumerate(ranked_ids, start=1):
            if vid in relevant:
                first_relevant_rank = i
                break

        query_rows.append(
            {
                "query_video": query_video,
                "num_relevant": len(relevant),
                "recall@1": r1,
                "recall@5": r5,
                "recall@10": r10,
                "average_precision": None if ap is None else float(ap),
                "first_relevant_rank": first_relevant_rank,
                "retrieval_time_sec": retrieval_duration,
            }
        )

        top_ranked = ranked[:ranking_save_k] if ranking_save_k is not None else ranked
        for rank, r in enumerate(top_ranked, start=1):
            ranking_rows.append(
                {
                    "query_video": query_video,
                    "rank": rank,
                    "candidate_video": r["video_id"],
                    "score": r["combined_score"],
                    "max_score": r.get("max_score"),
                    "coverage": r.get("coverage"),
                    "num_hits": r.get("num_hits"),
                    "is_relevant": int(r["video_id"] in relevant),
                }
            )

    if len(recall_at_1) == 0:
        raise ValueError("No retrieval rankings were successfully evaluated.")

    results = {
        "queries_evaluated": len(recall_at_1),
        "queries_with_no_results": len(empty_query_rows),
        "recall@1": float(np.mean(recall_at_1)),
        "recall@5": float(np.mean(recall_at_5)),
        "recall@10": float(np.mean(recall_at_10)),
        "mAP": float(np.mean(ap_scores)) if len(ap_scores) > 0 else float("nan"),
        "top_k_per_fragment": top_k_per_fragment,
        "retrieval_backend": "faiss",
        "retrieval_time_sec_mean": float(np.mean([row["retrieval_time_sec"] for row in query_rows])),
        "retrieval_time_sec_median": float(np.median([row["retrieval_time_sec"] for row in query_rows])),
        "retrieval_time_sec_p95": float(np.percentile([row["retrieval_time_sec"] for row in query_rows], 95)),
    }

    return (
        results,
        pd.DataFrame(query_rows),
        pd.DataFrame(ranking_rows),
        pd.DataFrame(empty_query_rows),
    )


def evaluate_retrieval_chamfer(test_csv, ranking_save_k=100, max_queries=None):
    """evaluates retrieval using the chamfer distance-based retrieval backend"""
    gt, query_videos = build_retrieval_ground_truth(test_csv)
    if max_queries is not None and max_queries > 0:
        query_videos = query_videos[:max_queries]

    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    ap_scores = []

    query_rows = []
    ranking_rows = []
    empty_query_rows = []

    #frag_db = construct_fragment_db()
    video_db = construct_video_db()
    
    descriptor_index = build_descriptor_index()

    print(f"Queries to evaluate: {len(query_videos)}")
    #print(f"Number of database fragments: {len(frag_d)}")

    for query_video in tqdm(query_videos, desc="Chamfer retrieval eval"):
        relevant = gt[query_video]
        if len(relevant) == 0:
            continue

        query_desc_path = resolve_descriptor_path(descriptor_index, query_video)
        if query_desc_path is None:
            empty_query_rows.append(
                {
                    "query_video": query_video,
                    "num_relevant": len(relevant),
                    "reason": "missing_query_descriptor",
                }
            )
            continue
        
        t0 = time.perf_counter()
        ranked = retrieve_similar_videos_chamfer(
            query_desc_path=query_desc_path,
            video_db=video_db,
        )
        t1 = time.perf_counter()
        retrieval_duration = t1 - t0

        ranked = [(vid, stats) for vid, stats in ranked if vid != query_video]
        ranked_ids = [vid for vid, _ in ranked]

        if len(ranked_ids) == 0:
            empty_query_rows.append(
                {
                    "query_video": query_video,
                    "num_relevant": len(relevant),
                    "reason": "no_results",
                }
            )
            continue

        r1 = int(any(v in relevant for v in ranked_ids[:1]))
        r5 = int(any(v in relevant for v in ranked_ids[:5]))
        r10 = int(any(v in relevant for v in ranked_ids[:10]))
        ap = average_precision_at_k(ranked_ids, relevant, k=None)

        recall_at_1.append(r1)
        recall_at_5.append(r5)
        recall_at_10.append(r10)
        if ap is not None:
            ap_scores.append(ap)

        first_relevant_rank = None
        for i, vid in enumerate(ranked_ids, start=1):
            if vid in relevant:
                first_relevant_rank = i
                break

        query_rows.append(
            {
                "query_video": query_video,
                "num_relevant": len(relevant),
                "recall@1": r1,
                "recall@5": r5,
                "recall@10": r10,
                "average_precision": None if ap is None else float(ap),
                "first_relevant_rank": first_relevant_rank,
                "retrieval_time_sec": retrieval_duration,
            }
        )

        top_ranked = ranked[:ranking_save_k] if ranking_save_k is not None else ranked
        for rank, (candidate_video, stats) in enumerate(top_ranked, start=1):
            ranking_rows.append(
                {
                    "query_video": query_video,
                    "rank": rank,
                    "candidate_video": candidate_video,
                    "score": stats,
                    #"max_score": stats.get("max_score"),
                    "is_relevant": int(candidate_video in relevant),
                }
            )

    if len(recall_at_1) == 0:
        raise ValueError("No retrieval rankings were successfully evaluated.")

    results = {
        "queries_evaluated": len(recall_at_1),
        "queries_with_no_results": len(empty_query_rows),
        "recall@1": float(np.mean(recall_at_1)),
        "recall@5": float(np.mean(recall_at_5)),
        "recall@10": float(np.mean(recall_at_10)),
        "mAP": float(np.mean(ap_scores)) if len(ap_scores) > 0 else float("nan"),
        "retrieval_backend": "chamfer",
        "retrieval_time_sec_mean": float(np.mean([row["retrieval_time_sec"] for row in query_rows])),
        "retrieval_time_sec_median": float(np.median([row["retrieval_time_sec"] for row in query_rows])),
        "retrieval_time_sec_p95": float(np.percentile([row["retrieval_time_sec"] for row in query_rows], 95)),
    }

    return (
        results,
        pd.DataFrame(query_rows),
        pd.DataFrame(ranking_rows),
        pd.DataFrame(empty_query_rows),
    )




def save_segment_results(model, test_csv_path, output_dir):
    """evaluates the model on segment pairs and saves metrics and predictions to output_dir"""
    from train import cap_negatives
    test_df = pd.read_csv(test_csv_path)
    test_df = cap_negatives(test_df, neg_pos_ratio=1.0, random_state=42)
    test_dataset = VCDBSegmentPairDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    av_g_loss, accuracy, precision, recall, f1, fbeta = validate(model, test_loader, device)

    summary_metrics = {
        "loss": to_float(av_g_loss),
        "accuracy": to_float(accuracy),
        "precision": to_float(precision),
        "recall": to_float(recall),
        "f1": to_float(f1),
        "fbeta": to_float(fbeta),
        
    }

    type_metrics = validate_per_pair_type(model, test_loader, device)
    type_metrics_clean = {}
    for pair_type, metrics in type_metrics.items():
        type_metrics_clean[pair_type] = {k: to_float(v) for k, v in metrics.items()}

    save_json(summary_metrics, output_dir / "metrics.json")
    save_json(type_metrics_clean, output_dir / "pair_type_metrics.json")

    print(
        f"Overall - Loss: {summary_metrics['loss']:.4f}, Accuracy: {summary_metrics['accuracy']:.4f}, "
        f"Precision: {summary_metrics['precision']:.4f}, Recall: {summary_metrics['recall']:.4f}, "
        f"F1: {summary_metrics['f1']:.4f}"
        f""
    )
    print(f"Saved summary metrics to {output_dir}")


def save_whole_video_fragments_results(model, output_dir):
    test_loader = construct_dataloaders()
    metrics, pairs_df, empty_df = validate_whole_video_pairs_from_fragments(model, test_loader)

    save_json(metrics, output_dir / "metrics.json")
    pairs_df.to_csv(output_dir / "pair_predictions.csv", index=False)
    empty_df.to_csv(output_dir / "empty_pairs.csv", index=False)

    print(
        f"Whole Video (fragments) - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f},"
        f" F-beta: {metrics['fbeta']:.4f},"
        f" Latency (ms) - mean: {metrics['latency_mean']:.2f}, median: {metrics['latency_median']:.2f}, p95: {metrics['latency_p95']:.2f}"

    )
    print(f"Saved results to {output_dir}")


def save_whole_video_descriptors_results(
    model,
    test_csv_path,
    output_dir,
    max_pairs = None,
    balanced_pairs = False,
    sample_seed = 42,
):
    """evaluates the model on whole video pairs using precomputed descriptors and saves metrics and predictions to output_dir"""
    from train import cap_negatives
    test_df = pd.read_csv(test_csv_path)
    test_df = cap_negatives(test_df, neg_pos_ratio=1.0, random_state=42)
    before_len = len(test_df)
    test_df = subsample_pairs(
        test_df,
        max_pairs=max_pairs,
        balanced_pairs=balanced_pairs,
        sample_seed=sample_seed,
    )
    if len(test_df) != before_len:
        print(
            f"Subsampled whole-video test pairs from {before_len} to {len(test_df)} "
            f"(balanced_pairs={balanced_pairs}, sample_seed={sample_seed})."
        )
        print("Label distribution after sampling:")
        print(test_df["label"].value_counts(dropna=False).to_string())

    metrics, pairs_df, missing_df, empty_df = validate_whole_video_pairs_from_descriptors(model, test_df)

    save_json(metrics, output_dir / "metrics.json")
    pairs_df.to_csv(output_dir / "pair_predictions.csv", index=False)
    missing_df.to_csv(output_dir / "missing_pairs.csv", index=False)
    empty_df.to_csv(output_dir / "empty_pairs.csv", index=False)

    print(
        f"Whole Video (descriptors) - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
        f" F-beta: {metrics['fbeta']:.4f},"
        f" Latency (ms) - mean: {metrics['latency_mean']:.2f}, median: {metrics['latency_median']:.2f}, p95: {metrics['latency_p95']:.2f}"
    )
    print(f"Saved results to {output_dir}")


def evaluate_retrieval_chamfer_faiss(
    test_csv,
    top_k_frames=50,
    top_k_per_fragment=None,
    ranking_save_k=100,
    max_queries=None,

):
    """evaluates retrieval using the chamfer distance-based retrieval backend with FAISS acceleration"""
    gt, query_videos = build_retrieval_ground_truth(test_csv)
    if max_queries is not None and max_queries > 0:
        query_videos = query_videos[:max_queries]
    
    
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    ap_scores = []

    query_rows = []
    ranking_rows = []
    empty_query_rows = []

    descriptor_index = build_descriptor_index()
    print(f"Queries to evaluate: {len(query_videos)}")
    descriptor_cache = {
    p.relative_to(DESCRIPTORS_DIR).with_suffix("").as_posix(): load_descriptor_file(p)
    for p in DESCRIPTORS_DIR.rglob("*.npy")
}
    pool_index, pool_metadata, pool_vid_array = load_index_and_metadata()

    for query_video in tqdm(query_videos, desc="Chamfer-FAISS retrieval eval"):
        relevant = gt[query_video]
        if len(relevant) == 0:
            continue

        query_desc_path = resolve_descriptor_path(descriptor_index, query_video)
        if query_desc_path is None:
            empty_query_rows.append({
                "query_video": query_video,
                "num_relevant": len(relevant),
                "reason": "missing_query_descriptor",
            })
            continue
        t0 = time.perf_counter()
        ranked = retrieve_similar_videos_chamfer_faiss(
            query_desc_path=query_desc_path,
            top_k_videos=top_k_frames,
            top_k_per_fragment=top_k_per_fragment,
            descriptor_cache=descriptor_cache,
            index=pool_index,
            metadata=pool_metadata,
            video_id_array=pool_vid_array,
        )
        retrieval_duration = time.perf_counter() - t0
        ranked_ids = [vid for vid in ranked if vid != query_video]


        if len(ranked_ids) == 0:
            empty_query_rows.append({
                "query_video": query_video,
                "num_relevant": len(relevant),
                "reason": "no_results",
            })
            continue

        r1 = int(any(v in relevant for v in ranked_ids[:1]))
        r5 = int(any(v in relevant for v in ranked_ids[:5]))
        r10 = int(any(v in relevant for v in ranked_ids[:10]))
        ap = average_precision_at_k(ranked_ids, relevant, k=None)

        recall_at_1.append(r1)
        recall_at_5.append(r5)
        recall_at_10.append(r10)
        if ap is not None:
            ap_scores.append(ap)

        first_relevant_rank = None
        for i, vid in enumerate(ranked_ids, start=1):
            if vid in relevant:
                first_relevant_rank = i
                break

        query_rows.append({
            "query_video": query_video,
            "num_relevant": len(relevant),
            "recall@1": r1,
            "recall@5": r5,
            "recall@10": r10,
            "average_precision": None if ap is None else float(ap),
            "first_relevant_rank": first_relevant_rank,
            "retrieval_time_sec": retrieval_duration,
        })

        top_ranked = ranked_ids[:ranking_save_k] if ranking_save_k is not None else ranked
        for rank, candidate_video in enumerate(top_ranked, start=1):
            ranking_rows.append({
                "query_video": query_video,
                "rank": rank,
                "candidate_video": candidate_video,
                #"max_score": stats.get("max_score"),
                #"coverage": stats.get("coverage"),
                "is_relevant": int(candidate_video in relevant),
                
            })

    if len(recall_at_1) == 0:
        raise ValueError("No retrieval rankings were successfully evaluated.")

    results = {
        "queries_evaluated": len(recall_at_1),
        "queries_with_no_results": len(empty_query_rows),
        "recall@1": float(np.mean(recall_at_1)),
        "recall@5": float(np.mean(recall_at_5)),
        "recall@10": float(np.mean(recall_at_10)),
        "mAP": float(np.mean(ap_scores)) if ap_scores else float("nan"),
        "retrieval_time_sec_mean": float(np.mean([row["retrieval_time_sec"] for row in query_rows])),
        "retrieval_time_sec_median": float(np.median([row["retrieval_time_sec"] for row in query_rows])),
        "retrieval_time_sec_p95": float(np.percentile([row["retrieval_time_sec"] for row in query_rows], 95)),
        "top_k_frames": top_k_frames,
        "retrieval_backend": "faiss_chamfer",
    }

    return (
        results,
        pd.DataFrame(query_rows),
        pd.DataFrame(ranking_rows),
        pd.DataFrame(empty_query_rows),
    )

def evaluate_retrieval_pred_head(
    test_csv,
    model,
    top_k_frames=50,
    top_k_per_fragment=None,
    ranking_save_k=100,
    max_queries=None,
    device = get_device(),
    pool_index = None,
    pool_metadata = None,
    pool_vid_array = None,
):
    """evaluates retrieval using the prediction head of the model with precomputed descriptors and FAISS acceleration"""
    gt, query_videos = build_retrieval_ground_truth(test_csv)
    if max_queries is not None and max_queries > 0:
        query_videos = query_videos[:max_queries]
    
    

    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    ap_scores = []

    query_rows = []
    ranking_rows = []
    empty_query_rows = []

    descriptor_index = build_descriptor_index()
    print(f"Queries to evaluate: {len(query_videos)}")
    descriptor_cache = {
    p.relative_to(DESCRIPTORS_DIR).with_suffix("").as_posix(): load_descriptor_file(p)
    for p in DESCRIPTORS_DIR.rglob("*.npy")
}
    for query_video in tqdm(query_videos, desc="Chamfer-FAISS retrieval eval"):
        relevant = gt[query_video]
        if len(relevant) == 0:
            continue

        query_desc_path = resolve_descriptor_path(descriptor_index, query_video)
        if query_desc_path is None:
            empty_query_rows.append({
                "query_video": query_video,
                "num_relevant": len(relevant),
                "reason": "missing_query_descriptor",
            })
            continue
        t0 = time.perf_counter()
        ranked = retrieve_similar_videos_pred_head(
            query_desc_path=query_desc_path,
            descriptor_cache=descriptor_cache,
            model=model,
            device=device,
            top_k_videos=top_k_frames,
            top_k_per_fragment=top_k_per_fragment,
            index=pool_index,
            metadata=pool_metadata,
            video_id_array=pool_vid_array,
        )
        retrieval_duration = time.perf_counter() - t0
        ranked_ids = [vid for vid in ranked if vid != query_video]

        if len(ranked_ids) == 0:
            empty_query_rows.append({
                "query_video": query_video,
                "num_relevant": len(relevant),
                "reason": "no_results",
            })
            continue

        r1 = int(any(v in relevant for v in ranked_ids[:1]))
        r5 = int(any(v in relevant for v in ranked_ids[:5]))
        r10 = int(any(v in relevant for v in ranked_ids[:10]))
        ap = average_precision_at_k(ranked_ids, relevant, k=None)

        recall_at_1.append(r1)
        recall_at_5.append(r5)
        recall_at_10.append(r10)
        if ap is not None:
            ap_scores.append(ap)

        first_relevant_rank = None
        for i, vid in enumerate(ranked_ids, start=1):
            if vid in relevant:
                first_relevant_rank = i
                break

        query_rows.append({
            "query_video": query_video,
            "num_relevant": len(relevant),
            "recall@1": r1,
            "recall@5": r5,
            "recall@10": r10,
            "average_precision": None if ap is None else float(ap),
            "first_relevant_rank": first_relevant_rank,
            "retrieval_time_sec": retrieval_duration,
        })

        top_ranked = ranked[:ranking_save_k] if ranking_save_k is not None else ranked
        for rank, candidate_video in enumerate(top_ranked, start=1):
            ranking_rows.append({
                "query_video": query_video,
                "rank": rank,
                "candidate_video": candidate_video,
                #"max_score": stats.get("max_score"),
                #"coverage": stats.get("coverage"),
                "is_relevant": int(candidate_video in relevant),
                
            })

    if len(recall_at_1) == 0:
        raise ValueError("No retrieval rankings were successfully evaluated.")

    results = {
        "queries_evaluated": len(recall_at_1),
        "queries_with_no_results": len(empty_query_rows),
        "recall@1": float(np.mean(recall_at_1)),
        "recall@5": float(np.mean(recall_at_5)),
        "recall@10": float(np.mean(recall_at_10)),
        "mAP": float(np.mean(ap_scores)) if ap_scores else float("nan"),
        "retrieval_time_sec_mean": float(np.mean([row["retrieval_time_sec"] for row in query_rows])),
        "retrieval_time_sec_median": float(np.median([row["retrieval_time_sec"] for row in query_rows])),
        "retrieval_time_sec_p95": float(np.percentile([row["retrieval_time_sec"] for row in query_rows], 95)),
        "top_k_frames": top_k_frames,
        "retrieval_backend": "pred_head",
    }

    return (
        results,
        pd.DataFrame(query_rows),
        pd.DataFrame(ranking_rows),
        pd.DataFrame(empty_query_rows),
    )

def evaluate_retrieval_pred_only(
    test_csv,
    model,
    top_k_frames=50,
    ranking_save_k=100,
    max_queries=None,
    device = get_device()):
    """evaluates retrieval using the prediction head of the model with precomputed descriptors and FAISS acceleration"""
    gt, query_videos = build_retrieval_ground_truth(test_csv)
    if max_queries is not None and max_queries > 0:
        query_videos = query_videos[:max_queries]
    
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    ap_scores = []

    query_rows = []
    ranking_rows = []
    empty_query_rows = []

    descriptor_index = build_descriptor_index()
    print(f"Queries to evaluate: {len(query_videos)}")
    descriptor_cache = {
    p.relative_to(DESCRIPTORS_DIR).with_suffix("").as_posix(): load_descriptor_file(p)
    for p in DESCRIPTORS_DIR.rglob("*.npy")
}

    for query_video in tqdm(query_videos, desc="Chamfer-FAISS retrieval eval"):
        relevant = gt[query_video]
        if len(relevant) == 0:
            continue

        query_desc_path = resolve_descriptor_path(descriptor_index, query_video)
        if query_desc_path is None:
            empty_query_rows.append({
                "query_video": query_video,
                "num_relevant": len(relevant),
                "reason": "missing_query_descriptor",
            })
            continue
        t0 = time.perf_counter()
        ranked = retrieve_similar_videos_pred_only(
            query_desc_path=query_desc_path,
            descriptor_cache=descriptor_cache,
            model=model,
            device = device)
        retrieval_duration = time.perf_counter() - t0
        ranked = sorted([(vid, stats) for vid, stats in ranked.items() if vid != query_video], key=lambda x: x[1], reverse=True)
        ranked_ids = [vid for vid, _ in ranked]

        if len(ranked_ids) == 0:
            empty_query_rows.append({
                "query_video": query_video,
                "num_relevant": len(relevant),
                "reason": "no_results",
            })
            continue

        r1 = int(any(v in relevant for v in ranked_ids[:1]))
        r5 = int(any(v in relevant for v in ranked_ids[:5]))
        r10 = int(any(v in relevant for v in ranked_ids[:10]))
        ap = average_precision_at_k(ranked_ids, relevant, k=None)

        recall_at_1.append(r1)
        recall_at_5.append(r5)
        recall_at_10.append(r10)
        if ap is not None:
            ap_scores.append(ap)

        first_relevant_rank = None
        for i, vid in enumerate(ranked_ids, start=1):
            if vid in relevant:
                first_relevant_rank = i
                break

        query_rows.append({
            "query_video": query_video,
            "num_relevant": len(relevant),
            "recall@1": r1,
            "recall@5": r5,
            "recall@10": r10,
            "average_precision": None if ap is None else float(ap),
            "first_relevant_rank": first_relevant_rank,
            "retrieval_time_sec": retrieval_duration,
        })

        top_ranked = ranked[:ranking_save_k] if ranking_save_k is not None else ranked
        for rank, (candidate_video, stats) in enumerate(top_ranked, start=1):
            ranking_rows.append({
                "query_video": query_video,
                "rank": rank,
                "candidate_video": candidate_video,
                "score": stats,
                #"max_score": stats.get("max_score"),
                #"coverage": stats.get("coverage"),
                "is_relevant": int(candidate_video in relevant),
                
            })

    if len(recall_at_1) == 0:
        raise ValueError("No retrieval rankings were successfully evaluated.")

    results = {
        "queries_evaluated": len(recall_at_1),
        "queries_with_no_results": len(empty_query_rows),
        "recall@1": float(np.mean(recall_at_1)),
        "recall@5": float(np.mean(recall_at_5)),
        "recall@10": float(np.mean(recall_at_10)),
        "mAP": float(np.mean(ap_scores)) if ap_scores else float("nan"),
        "retrieval_time_sec_mean": float(np.mean([row["retrieval_time_sec"] for row in query_rows])),
        "retrieval_time_sec_median": float(np.median([row["retrieval_time_sec"] for row in query_rows])),
        "retrieval_time_sec_p95": float(np.percentile([row["retrieval_time_sec"] for row in query_rows], 95)),
        "top_k_frames": top_k_frames,
        "retrieval_backend": "pred_only",
    }

    return (
        results,
        pd.DataFrame(query_rows),
        pd.DataFrame(ranking_rows),
        pd.DataFrame(empty_query_rows),
    )
def save_retrieval_results(
    test_csv_path,
    output_dir,
    top_k_per_fragment,
    top_k_rerank,
    top_k_rankings,
    max_queries,
    model = None,
    retrieval_backend = "faiss",
    device = device,
    ):
    """evaluates retrieval using the specified backend and saves metrics and rankings to output_dir"""

    test_df = pd.read_csv(test_csv_path)

    if retrieval_backend == "faiss":
        results, query_df, rankings_df, empty_query_df = evaluate_retrieval_faiss(
            test_df,
            top_k_per_fragment=top_k_per_fragment,
            ranking_save_k=top_k_rankings,
            max_queries=max_queries,
        )
    elif retrieval_backend == "chamfer":
        results, query_df, rankings_df, empty_query_df = evaluate_retrieval_chamfer(
            test_df,
            ranking_save_k=top_k_rankings,
            max_queries=max_queries,
        )
    elif retrieval_backend == "faiss_chamfer":
        results, query_df, rankings_df, empty_query_df = evaluate_retrieval_chamfer_faiss(
            test_df,
            top_k_frames=top_k_rerank,
            top_k_per_fragment=top_k_per_fragment,
            ranking_save_k=top_k_rankings,
            max_queries=max_queries,)
    elif retrieval_backend == "pred_head":
        pool_index, pool_metadata, pool_vid_array = load_index_and_metadata()
        results, query_df, rankings_df, empty_query_df = evaluate_retrieval_pred_head(
            test_df,
            model=model,
            top_k_frames=top_k_rerank,
            top_k_per_fragment=top_k_per_fragment,
            ranking_save_k=top_k_rankings,
            max_queries=max_queries,
            device=device,
            pool_index=pool_index,
            pool_metadata=pool_metadata,
            pool_vid_array=pool_vid_array,
        )
    elif retrieval_backend == "pred_only":
        results, query_df, rankings_df, empty_query_df = evaluate_retrieval_pred_only(
            test_df,
            model =model,
            top_k_frames=top_k_per_fragment,
            ranking_save_k=top_k_rankings,
            max_queries=max_queries,
            device=device,
        )
    else:
        raise ValueError(f"Unknown retrieval backend: {retrieval_backend}")

    save_json(results, output_dir / "metrics.json")
    query_df.to_csv(output_dir / "per_query_metrics.csv", index=False)
    rankings_df.to_csv(output_dir / "rankings.csv", index=False)
    empty_query_df.to_csv(output_dir / "queries_with_no_results.csv", index=False)

    print(f"Retrieval backend: {results['retrieval_backend']}")
    print(f"Queries evaluated: {results['queries_evaluated']}")
    print(f"Recall@1:  {results['recall@1']:.4f}")
    print(f"Recall@5:  {results['recall@5']:.4f}")
    print(f"Recall@10: {results['recall@10']:.4f}")
    print(f"mAP:       {results['mAP']:.4f}")
    print(f"Mean retrieval time (sec): {results['retrieval_time_sec_mean']:.4f}")
    print(f"Median retrieval time (sec): {results['retrieval_time_sec_median']:.4f}")
    print(f"P95 retrieval time (sec): {results['retrieval_time_sec_p95']:.4f}")

    print(f"Saved results to {output_dir}")


def resolve_test_csv_for_mode(mode, test_csv_arg):
    if test_csv_arg is not None:
        return Path(test_csv_arg)

    if mode == "segment":
        return DEFAULT_SEGMENT_TEST_CSV
    if mode in {"whole_video_descriptors", "retrieval"}:
        return DEFAULT_VIDEO_LEVEL_TEST_CSV
    return None


def main(args):
    set_seed()
    output_dir = make_output_dir(
        Path(args.output_root),
        args.mode,
        retrieval_backend=args.retrieval_backend,
    )
    resolved_test_csv = resolve_test_csv_for_mode(args.mode, args.test_csv)

    config = {
        "mode": args.mode,
        "retrieval_backend": args.retrieval_backend,
        "model_path": str(args.model_path),
        "test_csv": None if resolved_test_csv is None else str(resolved_test_csv),
        "descriptors_dir": str(DESCRIPTORS_DIR),
        "device": str(device),
        "top_k_rankings": args.top_k_rankings,
        "top_k_per_fragment": args.top_k_per_fragment,
        "top_k_rerank": args.top_k_rerank,
        "max_queries": args.max_queries,
        "max_pairs": args.max_pairs,
        "balanced_pairs": args.balanced_pairs,
        "sample_seed": args.sample_seed,
    }
    save_json(config, output_dir / "config.json")

    model = None
    if args.mode in ["segment", "whole_video_fragments", "whole_video_descriptors"] or \
        (args.mode == "retrieval" and args.retrieval_backend in ["pred_head", "pred_only"]):

        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

    if args.mode == "segment":
        if resolved_test_csv is None:
            raise ValueError("No test CSV resolved for segment mode.")
        save_segment_results(model, resolved_test_csv, output_dir)

    elif args.mode == "whole_video_fragments":
        save_whole_video_fragments_results(model, output_dir)

    elif args.mode == "whole_video_descriptors":
        if resolved_test_csv is None:
            raise ValueError("No test CSV resolved for whole_video_descriptors mode.")
        save_whole_video_descriptors_results(
            model,
            resolved_test_csv,
            output_dir,
            max_pairs=args.max_pairs,
            balanced_pairs=args.balanced_pairs,
            sample_seed=args.sample_seed,
        )

    elif args.mode == "retrieval":
        if resolved_test_csv is None:
            raise ValueError("No test CSV resolved for retrieval mode.")
        save_retrieval_results(
            resolved_test_csv,
            output_dir,
            top_k_per_fragment=args.top_k_per_fragment,
            top_k_rerank=args.top_k_rerank,
            top_k_rankings=args.top_k_rankings,
            max_queries=args.max_queries,
            retrieval_backend=args.retrieval_backend,
            model=model,
            device=device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Siamese Network on VCDB")

    parser.add_argument(
        "--mode",
        choices=["segment", "whole_video_fragments", "whole_video_descriptors", "retrieval"],
        default="whole_video_descriptors",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--retrieval_backend",
        choices=["faiss", "chamfer", "faiss_chamfer", "pred_head","pred_only"],
        default="faiss",
        help="Which retrieval backend to use when --mode retrieval. "
             "faiss_chamfer = two-stage: FAISS candidate retrieval + Chamfer re-ranking.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help=(
            "Path to test CSV. If omitted: "
            "segment->test_pairs_segment.csv, "
            "whole_video_descriptors/retrieval->test_pairs_video_level.csv"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where evaluation outputs will be saved",
    )
    parser.add_argument(
        "--top_k_rankings",
        type=int,
        default=100,
        help="How many ranked candidates per query to save in retrieval mode. Use -1 to save all.",
    )
    parser.add_argument(
        "--top_k_per_fragment",
        type=int,
        default=-1,
        help="How many nearest neighbours to retrieve per query fragment in FAISS retrieval mode. Use -1 to retrieve all (ntotal).",
    )
    parser.add_argument(
        "--top_k_rerank",
        type=int,
        default=50,
        help="How many FAISS candidates to pass to the reranking stage (pred_head or chamfer). Only used for faiss_chamfer and pred_head backends.",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="If set, evaluate only the first N retrieval queries for faster sanity checks.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional cap on number of pairs (mainly for whole_video_descriptors quick evaluation).",
    )
    parser.add_argument(
        "--balanced_pairs",
        action="store_true",
        help="If set with --max_pairs, sample pairs with approximately balanced labels.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Random seed for pair subsampling.",
    )

    args = parser.parse_args()

    if args.top_k_rankings == -1:
        args.top_k_rankings = None
    if args.top_k_per_fragment == -1:
        args.top_k_per_fragment = None
    if args.top_k_rerank == -1:
        args.top_k_rerank = None

    main(args)
