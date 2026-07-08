import os
import sys
import csv
import uuid
import pickle
import hashlib
import subprocess
from pathlib import Path
from contextlib import contextmanager
import argparse

import numpy as np
import pandas as pd
import tmkpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "CompactSIamese"))
from runtime_config import FIVR_DIR



BASE_DIR = Path.cwd()

OUTPUT_DIR = FIVR_DIR / "tmk_descriptors"
FAIL_LOG   = OUTPUT_DIR / "failed.txt"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def find_video(vide_dir, vid_id):
    for ext in VIDEO_EXTS:
        candidate = vide_dir / f"{vid_id}{ext}"
        if candidate.exists():
            return candidate
    return None





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





def load_feature(path):
    with open(path, "rb") as f:
        return pickle.load(f)




def compute_tmk(video_path, ffmpeg="ffmpeg"):
    try:
        return tmkpy.hashVideo(str(video_path), ffmpeg)
    except Exception as e:
        print(f"  hashVideo error: {e}", flush=True)
        return None




    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_start", type=int, default=0)
    parser.add_argument("--chunk_end",   type=int, default=None)
    parser.add_argument("--ids_file",    type=str, default=None)
    args = parser.parse_args()

    if args.ids_file is not None:
        ids_file = Path(args.ids_file)
    else:
        ids_file = Path(__file__).resolve().parents[2] / "CompactSIamese/fivr_subset_50k/fivr_reference_ids.txt"
    video_dir = FIVR_DIR / "videos"

    with open(ids_file) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    chunk = all_ids[args.chunk_start : args.chunk_end]
    print(f"Processing {len(chunk)} videos")
    ok = 0
    failed = 0
    skipped = 0
    for vid_id in tqdm(chunk,desc="Processing videos"):
        out_path = OUTPUT_DIR / f"{vid_id}.tmk"
        if out_path.exists():
            skipped += 1
            continue
        video_path = find_video(video_dir, vid_id)
        if video_path is None:
            with open(FAIL_LOG, "a") as f:
                f.write(f"{vid_id}: video not found\n")
            failed += 1
            continue
        with suppress_subprocess_output():
            vid_obj = compute_tmk(video_path)
        if vid_obj is None:
            with open(FAIL_LOG, "a") as f:
                f.write(f"{vid_id}: hashing failed\n")
            failed += 1
            continue
        vid_obj.writeToOutputFile(str(out_path), "fivr_hashing")
        ok += 1

    print(f"Done — ok: {ok}, skipped: {skipped}, failed: {failed}")

if __name__ == "__main__":
    main()




