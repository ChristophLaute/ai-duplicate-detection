import os
import sys
import csv
import uuid
import pickle
import hashlib
import subprocess
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import tmkpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime_config import VCDB_DIR


BASE_DIR = Path.cwd()

POS_CSV = VCDB_DIR / "positive_pairs" / "positive_pairs.csv"
NEG_CSV = VCDB_DIR / "negative_pairs" / "negative_pairs.csv"

TMP_DIR = BASE_DIR / "_tmp_vcdb_segments"
OUT_DIR = BASE_DIR / "tmk_hashes_vcdb"
FAIL_LOG = OUT_DIR / "failed_segments.txt"

FFMPEG = "/opt/homebrew/bin/ffmpeg"  

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)



@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


@contextmanager
def suppress_subprocess_output():
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)


def make_segment_id(video_path, start, end):
    raw = f"{video_path}|{start}|{end}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def save_feature(path, feat):
    with open(path, "wb") as f:
        pickle.dump(np.asarray(feat, dtype=np.float32), f)


def load_feature(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_segment(video_path, start_time, end_time, segment_id):
    out_path = TMP_DIR / f"{segment_id}.mp4"

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    command = [
        FFMPEG,
        "-y",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", str(video_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "veryfast",
        str(out_path),
    ]

    with suppress_subprocess_output():
        subprocess.run(command, check=True)

    return out_path

def compute_tmk(video_path):
    try:
        return tmkpy.hashVideo(
            inputVideoFileName=str(video_path),
            ffmpegPath=FFMPEG
        )
    except Exception:
        return None


def hash_segment(video_path, start, end):
    segment_id = make_segment_id(video_path, start, end)

    tmk_path = OUT_DIR / f"{segment_id}.tmk"
    feat_path = OUT_DIR / f"{segment_id}.pkl"

    if tmk_path.exists() and feat_path.exists():
        return tmk_path, feat_path

    try:
        segment_path = extract_segment(video_path, start, end, segment_id)
    except Exception:
        with open(FAIL_LOG, "a") as f:
            f.write(f"{segment_id}\t{video_path}\t{start}\t{end}\textract_failed\n")
        return None, None

    try:
        with suppress_subprocess_output():
            vid_obj = compute_tmk(segment_path)

        if vid_obj is None:
            with open(FAIL_LOG, "a") as f:
                f.write(f"{segment_id}\t{video_path}\t{start}\t{end}\thash_failed\n")
            return None, None

        with suppress_subprocess_output():
            vid_obj.writeToOutputFile(str(tmk_path), "vcdb_hashing")

        feat = vid_obj.getPureAverageFeature()
        save_feature(feat_path, feat)

        return tmk_path, feat_path

    finally:
        if segment_path.exists():
            segment_path.unlink()

def build_hash_cache(df):
    existing = {
        p.stem for p in OUT_DIR.glob("*.tmk")
        if p.is_file() and p.stat().st_size > 0
    }

    failed = set()
    if FAIL_LOG.exists():
        with open(FAIL_LOG, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    failed.add(parts[0])

    seen = set()
    total_rows = len(df)

    for counter, (_, row) in enumerate(df.iterrows(), 1):
        if counter % 100 == 0 or counter == total_rows:
            print(f"Progress: {counter}/{total_rows} ({(counter/total_rows)*100:.2f}%)")

        segments = [
            (row["video1_path"], row["start1"], row["end1"]),
            (row["video2_path"], row["start2"], row["end2"])
        ]

        for video_path, start, end in segments:
            seg_id = make_segment_id(video_path, start, end)

            if seg_id in seen or seg_id in existing or seg_id in failed:
                continue

            hash_segment(video_path, start, end)
            seen.add(seg_id)


    
def score_pair_level1(row):
    seg1_id = make_segment_id(row["video1_path"], row["start1"], row["end1"])
    seg2_id = make_segment_id(row["video2_path"], row["start2"], row["end2"])

    feat1 = load_feature(OUT_DIR / f"{seg1_id}.pkl")
    feat2 = load_feature(OUT_DIR / f"{seg2_id}.pkl")

    return cosine_similarity(feat1, feat2)


def score_pair_level2(row):
    seg1_id = make_segment_id(row["video1_path"], row["start1"], row["end1"])
    seg2_id = make_segment_id(row["video2_path"], row["start2"], row["end2"])

    tmk1 = str(OUT_DIR / f"{seg1_id}.tmk")
    tmk2 = str(OUT_DIR / f"{seg2_id}.tmk")

    score = tmkpy.query(tmk1, [tmk2], 1)[0]
    return float(score)

def evaluate(df, scoring="level1", threshold=0.7):
    y_true = []
    y_score = []

    for i, row in df.iterrows():
        try:
            if scoring == "level1":
                score = score_pair_level1(row)
            elif scoring == "level2":
                score = score_pair_level2(row)
            else:
                raise ValueError("scoring must be 'level1' or 'level2'")

            y_true.append(int(row["label"]))
            y_score.append(score)

        except Exception as e:
            print(f"Failed scoring row {i}: {e}")

    y_pred = [1 if s >= threshold else 0 for s in y_score]

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "ap": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred),
    }

    return results, y_true, y_score


def find_best_threshold(y_true, y_score):
    best_t = None
    best_f1 = -1

    for t in np.linspace(0.0, 1.0, 101):
        y_pred = [1 if s >= t else 0 for s in y_score]
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1

pos_df = pd.read_csv(POS_CSV)
neg_df = pd.read_csv(NEG_CSV)

df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# same split as Siamese
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

print("train:", len(train_df))
print("val:", len(val_df))
print("test:", len(test_df))

print("\nBuilding TMK cache...")
build_hash_cache(df)

print("\nEvaluating level-1 on validation set...")
val_results_l1, y_val_l1, s_val_l1 = evaluate(val_df, scoring="level1", threshold=0.7)
print("Initial level-1 val:", val_results_l1)

best_t_l1, best_f1_l1 = find_best_threshold(y_val_l1, s_val_l1)
print("Best level-1 threshold:", best_t_l1)
print("Best level-1 val F1:", best_f1_l1)

print("\nEvaluating level-1 on test set...")
test_results_l1, _, _ = evaluate(test_df, scoring="level1", threshold=best_t_l1)
print("Final level-1 test:", test_results_l1)

print("\nEvaluating level-2 on validation set...")
val_results_l2, y_val_l2, s_val_l2 = evaluate(val_df, scoring="level2", threshold=0.7)
print("Initial level-2 val:", val_results_l2)

best_t_l2, best_f1_l2 = find_best_threshold(y_val_l2, s_val_l2)
print("Best level-2 threshold:", best_t_l2)
print("Best level-2 val F1:", best_f1_l2)

print("\nEvaluating level-2 on test set...")
test_results_l2, _, _ = evaluate(test_df, scoring="level2", threshold=best_t_l2)
print("Final level-2 test:", test_results_l2)
