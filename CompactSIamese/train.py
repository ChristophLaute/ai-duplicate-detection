import argparse

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset_segment_pairs import VCDBSegmentPairDataset
from model.metrics import calculate_metrics
from model.pytorch_model import SiameseNetwork
from runtime_config import VCDB_DIR, get_device
from torch.cuda.amp import autocast, GradScaler
from utils.seeds import set_seed
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

# Derived per-run in main() from args.train_csv to avoid cross-ablation contamination
BALANCED_CSV = None
def _subset(df,video_ids):
        video_ids = set(video_ids)
        return df[
            df["video1"].isin(video_ids)
            & df["video2"].isin(video_ids)
        ].copy().reset_index(drop=True)
def build_video_level_pairs(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert segment-level rows to video-level labels.

    Rules:
    - Remove self pairs.
    - Treat (video1, video2) and (video2, video1) as the same unordered pair.
    - If any segment row for a pair is positive, video-level label is positive.
    """
    required = {"video1", "video2", "label"}
    missing = required - set(pairs_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for video-level aggregation: {sorted(missing)}")

    df = pairs_df[["video1", "video2", "label"]].copy()
    df["label"] = df["label"].astype(float).round().astype(int)

    df = df[df["video1"] != df["video2"]].copy()
    df["a"] = df[["video1", "video2"]].min(axis=1)
    df["b"] = df[["video1", "video2"]].max(axis=1)

    video_df = (
        df.groupby(["a", "b"], as_index=False)
        .agg(label=("label", "max"))
        .rename(columns={"a": "video1", "b": "video2"})
        .sort_values(["video1", "video2"])
        .reset_index(drop=True)
    )

    return video_df


def split_by_video_ids(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
):
    df = df[df["video1"] != df["video2"]].copy().reset_index(drop=True)

    all_videos = np.array(sorted(set(df["video1"]).union(set(df["video2"]))))
    train_videos, temp_videos = train_test_split(
        all_videos,
        test_size=val_ratio + test_ratio,
        random_state=random_state,
    )

    temp_ratio = test_ratio / (val_ratio + test_ratio)
    val_videos, test_videos = train_test_split(
        temp_videos,
        test_size=temp_ratio,
        random_state=random_state,
    )

    return _subset(df, train_videos), _subset(df, val_videos), _subset(df, test_videos)


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for x1, x2, y, _pair_type in val_loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().reshape(-1)
            with autocast(enabled=(device.type=="cuda")):
                logits, _, _ = model(x1, x2)
                logits = logits.reshape(-1)

                loss = criterion(logits, y)

            y_true_all.append(y.cpu())
            y_pred_all.append(logits.cpu())
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    y_true_all = torch.cat(y_true_all)
    y_pred_all = torch.cat(y_pred_all)
    accuracy, precision, recall, f1, fbeta = calculate_metrics(y_true_all, y_pred_all)

    return avg_loss, accuracy, precision, recall, f1, fbeta


def validate_per_pair_type(model, val_loader, device):
    model.eval()
    pair_type_metrics = {}

    with torch.no_grad():
        for x1, x2, y, pair_types in val_loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().reshape(-1)

            with autocast(enabled=(device.type=="cuda")):
                logits, _, _ = model(x1, x2)
                logits = logits.reshape(-1)

                losses = nn.functional.binary_cross_entropy_with_logits(
                    logits, y, reduction="none"
                )

            for i, pt in enumerate(pair_types):
                if pt not in pair_type_metrics:
                    pair_type_metrics[pt] = {"y_true": [], "y_pred": [], "loss": []}

                pair_type_metrics[pt]["y_true"].append(y[i].cpu().unsqueeze(0))
                pair_type_metrics[pt]["y_pred"].append(logits[i].cpu().unsqueeze(0))
                pair_type_metrics[pt]["loss"].append(losses[i].item())

    results = {}
    for pt, metrics in pair_type_metrics.items():
        y_true_all = torch.cat(metrics["y_true"])
        y_pred_all = torch.cat(metrics["y_pred"])
        avg_loss = sum(metrics["loss"]) / len(metrics["loss"])
        accuracy, precision, recall, f1, fbeta = calculate_metrics(y_true_all, y_pred_all)

        results[pt] = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fbeta": fbeta,
        }

    return results

def cap_negatives(df, neg_pos_ratio=1.0, random_state=42):
    pos_df = df[df["label"] == 1]
    neg_df = df[df["label"] == 0]

    n_pos = len(pos_df)
    n_neg_to_keep = int(n_pos * neg_pos_ratio)

    if n_neg_to_keep >= len(neg_df):
        return df

    neg_df_sampled = neg_df.sample(n=n_neg_to_keep, random_state=random_state)
    balanced_df = pd.concat([pos_df, neg_df_sampled], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df

def train(train_df, val_df, learning_rate=1e-4, wd=1e-5,epochs=150,batch_size=16, patience=7,dropout=0.5, best_metric="fbeta",save_model=True,model_save_path="best_siamese_model.pth",log_csv_path="training_log.csv", train_mode = True, trial = None, T=15):
    device = get_device()
    if trial is not None:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        patience = trial.suggest_int("patience", 3, 10)
        dropout = trial.suggest_float("dropout", 0.1, 0.9)

    train_dataset = VCDBSegmentPairDataset(train_df)
    val_dataset = VCDBSegmentPairDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True if device.type == "cuda" else False, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True if device.type == "cuda" else False, persistent_workers=True)

    model = SiameseNetwork(dropout_p=dropout, T=T).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=wd,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_accuracy = 0.0
    best_recall = 0.0
    best_precision = 0.0
    best_fbeta = 0.0
    tolerance = patience
    val_per_pair_type = {}
    best_metric_value = float("inf") if best_metric == "loss" else 0.0
    log_rows = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x1, x2, y, _pair_type in train_loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().reshape(-1)

            optimizer.zero_grad()
            with autocast(enabled=(device.type=="cuda")):
                logits, _, _ = model(x1, x2)
                logits = logits.reshape(-1)

                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        print("epoch", epoch, "loss", train_loss)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_fbeta = validate(model, val_loader, device)
        metric_name = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "fbeta": "fbeta",
            "loss": "loss",
        }[best_metric]
        metric_value = {
            "accuracy": val_accuracy,
            "precision": val_precision,
            "recall": val_recall,
            "f1": val_f1,
            "fbeta": val_fbeta,
            "loss": val_loss,
        }[best_metric]
        if best_metric == "loss":
            improved = metric_value < best_metric_value if best_metric_value != 0.0 else True
        else:
            improved = metric_value > {
                "accuracy": best_accuracy,
                "precision": best_precision,
                "recall": best_recall,
                "f1": best_f1,
                "fbeta": best_fbeta,
            }[best_metric]
        if improved:
            best_f1 = val_f1
            best_accuracy = val_accuracy
            best_precision = val_precision
            best_recall = val_recall
            best_fbeta = val_fbeta
            best_metric_value = metric_value
            val_per_pair_type = validate_per_pair_type(model, val_loader, device)
            
            tolerance = patience
            if save_model:  
                torch.save(model.state_dict(), model_save_path)
        else:
            print("epoch", epoch, f"validation {metric_name} did not improve")
            tolerance -= 1
            if tolerance <= 0:
                print("Early stopping triggered")
                log_rows.append({
                    "epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "val_accuracy": float(val_accuracy),
                    "val_precision": float(val_precision), "val_recall": float(val_recall),
                    "val_f1": float(val_f1), "val_fbeta": float(val_fbeta), "best": False,
                })
                break

        if train_mode:

            print("epoch", epoch, "validation loss", val_loss)
            print("epoch", epoch, "validation accuracy", val_accuracy)
            print("epoch", epoch, "validation precision", val_precision)
            print("epoch", epoch, "validation recall", val_recall)
            print("epoch", epoch, "validation f1", val_f1)

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_accuracy": float(val_accuracy),
            "val_precision": float(val_precision), "val_recall": float(val_recall),
            "val_f1": float(val_f1), "val_fbeta": float(val_fbeta), "best": improved,
        })

    # Save training log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_csv_path, index=False)
    if train_mode:
        print(f"Training log saved to {log_csv_path}")
        print("Best F1 at best precision checkpoint:", best_f1)
        print("Best validation accuracy at best precision checkpoint:", best_accuracy)
        print("Best validation precision:", best_precision)
        print("Best validation recall at best precision checkpoint:", best_recall)
        print("Validation metrics per pair type:")
        for pt, metrics in val_per_pair_type.items():
            print(f"  Pair type: {pt}")
            print(f"    Loss: {metrics['loss']:.4f}")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1 Score: {metrics['f1']:.4f}")
    return { f"{metric_name}": best_metric_value}
    

def load_pairs(pos_pairs_csv=None, neg_pairs_csv=None):
    if pos_pairs_csv is None:
        pos_pairs_csv = VCDB_DIR / "positive_pairs" / "positive_pairs.csv"
    if neg_pairs_csv is None:
        neg_pairs_csv = VCDB_DIR / "negative_pairs" / "negative_pairs.csv"
    pos_df = pd.read_csv(pos_pairs_csv)
    neg_df = pd.read_csv(neg_pairs_csv)
    df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return df
def main():
    set_seed()  
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=15, help="Number of frames per fragment (default: 15)")
    parser.add_argument("--model-save-path", type=str, default="best_siamese_model.pth", help="Path to save best model checkpoint")
    parser.add_argument("--train-csv", type=str, default="train_pairs_segment.csv", help="Path to training pairs CSV")
    parser.add_argument("--val-csv", type=str, default="val_pairs_segment.csv", help="Path to validation pairs CSV")
    parser.add_argument("--pos-pairs-csv", type=Path, default=None, help="Path to positive_pairs.csv (overrides default)")
    parser.add_argument("--neg-pairs-csv", type=Path, default=None, help="Path to negative_pairs.csv (overrides default)")
    args = parser.parse_args()

    device = get_device()

    train_split = Path(args.train_csv)
    val_split = Path(args.val_csv)
    test_split = Path("test_pairs_segment.csv")

    # Derive balanced CSV path from train_csv so each ablation gets its own file
    # e.g. train_pairs_T10.csv -> balanced_pairs_T10.csv
    balanced_csv = train_split.parent / train_split.name.replace("train_pairs", "balanced_pairs")

    if train_split.exists() and val_split.exists() and test_split.exists():
        print("Loading existing fixed splits from CSV files.")
        train_df = pd.read_csv(train_split)
        val_df = pd.read_csv(val_split)
        test_df = pd.read_csv(test_split)
    else:
        df = load_pairs(pos_pairs_csv=args.pos_pairs_csv, neg_pairs_csv=args.neg_pairs_csv)
        train_df, val_df, test_df = split_by_video_ids(df, val_ratio=0.15, test_ratio=0.15, random_state=42)

        if balanced_csv.exists():
            print(f"Using existing balanced CSV at {balanced_csv}")
            train_df = pd.read_csv(balanced_csv)
        else:
            train_df = cap_negatives(train_df, neg_pos_ratio=1.0, random_state=42)
            train_df.to_csv(balanced_csv, index=False)

        train_df.to_csv(train_split, index=False)
        val_df.to_csv(val_split, index=False)
        test_df.to_csv(test_split, index=False)
        
        

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split_df) == 0:
            raise ValueError(f"{split_name} split is empty after video-level split; adjust the split strategy.")
        if split_df["label"].nunique() < 2:
            raise ValueError(f"{split_name} split contains only one class after video-level split.")

    # Segment-level split files (for segment model evaluation)
    train_df.to_csv("train_pairs_segment.csv", index=False)
    val_df.to_csv("val_pairs_segment.csv", index=False)
    test_df.to_csv("test_pairs_segment.csv", index=False)
    # Backward-compatible alias
    test_df.to_csv("test_pairs.csv", index=False)

    # Video-level split files (for whole-video descriptor/retrieval evaluation)
    train_video_df = build_video_level_pairs(train_df)
    val_video_df = build_video_level_pairs(val_df)
    test_video_df = build_video_level_pairs(test_df)
    train_video_df.to_csv("train_pairs_video_level.csv", index=False)
    val_video_df.to_csv("val_pairs_video_level.csv", index=False)
    test_video_df.to_csv("test_pairs_video_level.csv", index=False)

    print(
        "Saved split CSVs: "
        "train/val/test segment-level + train/val/test video-level "
        "(positive-wins aggregation, self-pairs removed, split by disjoint video IDs)."
    )
    print(
        "Video-level test pairs:",
        len(test_video_df),
        "| positives:",
        int((test_video_df["label"] == 1).sum()),
        "| negatives:",
        int((test_video_df["label"] == 0).sum()),
    )
    val_df = cap_negatives(val_df, neg_pos_ratio=1.0, random_state=42)

    results = train(
        train_df,
        val_df,
        learning_rate=3.07e-4,
        wd=2.33e-6,
        batch_size=32,
        patience=15,
        dropout=0.22,
        epochs=200,
        best_metric="loss",
        T=args.T,
        model_save_path=args.model_save_path,
    )


    


if __name__ == "__main__":
    main()
