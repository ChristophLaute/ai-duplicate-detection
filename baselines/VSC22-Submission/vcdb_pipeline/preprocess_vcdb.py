"""
Preprocesses VCDB videos into jpg zips and LMDB for VSC22 training.

This replaces the original vid2jpg_zip.py + zip2lmdb.py workflow, but is
fully driven by the vsc_adapter directory produced by prepare_vcdb_for_VSC.py.
No hardcoded paths, no test-split assumptions.

Usage:
    python preprocess_vcdb.py --data_dir /path/to/VCDB/vsc_adapter

What it does:
    1. Reads meta/train/train_query_metadata.csv and
       meta/train/train_reference_metadata.csv to get all video IDs.
    2. For each video, extracts 1 fps frames via ffmpeg into a per-video zip file
       at jpg_zips/{last2chars}/{video_id}.zip
    3. Packs all zips into two LMDB databases:
         lmdb/vsc/       <- used as the main training LMDB
         lmdb/train_vsc/ <- used as the augmentation LMDB (RandomOverlayImages)
                            this is a second pack of the same data, kept separate
                            so the two LMDB handles can be opened independently.
"""

import argparse
import io
import lmdb
import numpy as np
import os
import shutil
import subprocess
import zipfile
from multiprocessing.pool import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Step 1: MP4 → jpg zip
# ---------------------------------------------------------------------------

def _convert_one(args):
    video_path, zip_path, fps, timeout = args

    if not os.path.exists(video_path):
        print(f"  [skip] missing video: {video_path}")
        return

    if os.path.isfile(zip_path):
        return  # already done; remove zip_path first to redo

    tmp_dir = zip_path[:-4] + "_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = (
        f'ffmpeg -nostdin -y -i "{video_path}" '
        f'-start_number 0 -q 0 -vf fps={fps} '
        f'"{tmp_dir}/%07d.jpg"'
    )
    try:
        with open(os.devnull, "w") as null:
            subprocess.call(cmd, shell=True, timeout=timeout, stderr=null)
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  [timeout] {video_path}")
        return

    jpg_paths = sorted(
        os.path.join(tmp_dir, fn)
        for fn in os.listdir(tmp_dir)
        if fn.endswith(".jpg")
    )

    if jpg_paths:
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        with zipfile.ZipFile(zip_path, "w") as zf:
            for p in jpg_paths:
                zf.write(p, arcname=os.path.basename(p))

    shutil.rmtree(tmp_dir, ignore_errors=True)


def build_jpg_zips(data_dir, fps = 1, workers = 8, timeout = 120):
    """Extract frames from every train video into jpg zips.

    Q and R IDs are symlinks to the same source file, so we deduplicate by
    resolved path. The first ID encountered gets the real zip; any later ID
    pointing to the same source gets a symlink to that zip instead of a
    second ffmpeg run.
    """
    query_meta = pd.read_csv(data_dir / "meta" / "train" / "train_query_metadata.csv")
    ref_meta   = pd.read_csv(data_dir / "meta" / "train" / "train_reference_metadata.csv")
    all_ids = pd.concat([query_meta, ref_meta], ignore_index=True)["video_id"].tolist()

    args_list = []
    # resolved source path -> Path of the zip that will be created for it
    seen = {}

    for vid_id in all_ids:
        token      = "query" if vid_id.startswith("Q") else "reference"
        video_path = data_dir / "videos" / "train" / token / f"{vid_id}.mp4"
        zip_path   = data_dir / "jpg_zips" / vid_id[-2:] / f"{vid_id}.zip"

        # resolve symlink to find the actual source file on disk
        try:
            resolved = str(video_path.resolve())
        except OSError:
            resolved = str(video_path)  # unresolvable — treat as unique

        if resolved in seen:
            # Same source already queued under a different ID.
            # Create a symlink so the LMDB builder can find this ID's zip
            # without re-extracting frames.
            canonical_zip = seen[resolved]
            if not zip_path.exists() and not zip_path.is_symlink():
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                zip_path.symlink_to(canonical_zip.resolve())
        else:
            seen[resolved] = zip_path
            args_list.append((str(video_path), str(zip_path), fps, timeout))

    print(f"Building jpg zips for {len(args_list)} unique source videos "
          f"({len(all_ids) - len(args_list)} duplicates symlinked, workers={workers}) ...")
    with Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(_convert_one, args_list), total=len(args_list)))


# ---------------------------------------------------------------------------
# Step 2: jpg zips → LMDB
# ---------------------------------------------------------------------------

def build_lmdb(data_dir, lmdb_name, vids, lmdb_size = int(1e12)):
    """
    Pack jpg zips for the given video IDs into a single LMDB.

    The resulting LMDB contains:
      key = str(global_frame_index).encode()
      value = raw JPEG bytes

    A companion meta.npz is written alongside the LMDB:
      vids      : array of video IDs in insertion order
      intervals : array of [start_frame, end_frame) for each video
    """
    lmdb_path = data_dir / "lmdb" / lmdb_name
    meta_path = lmdb_path / "meta.npz"
    lmdb_path.mkdir(parents=True, exist_ok=True)

    if meta_path.exists():
        print(f"  [skip] LMDB already exists: {lmdb_path}")
        return

    print(f"Building LMDB '{lmdb_name}' for {len(vids)} videos ...")

    global_idx = 0
    vid_list, interval_list = [], []

    with lmdb.open(str(lmdb_path), map_size=lmdb_size) as env:
        with env.begin(write=True) as txn:
            for vid_id in tqdm(vids):
                zip_path = data_dir / "jpg_zips" / vid_id[-2:] / f"{vid_id}.zip"
                if not zip_path.exists():
                    continue

                start = global_idx
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        for img_name in sorted(zf.namelist()):
                            content = zf.read(img_name)
                            txn.put(str(global_idx).encode(), content)
                            global_idx += 1
                except Exception as e:
                    print(f"  [error] {vid_id}: {e}")
                    global_idx = start  # roll back partial video
                    continue

                vid_list.append(vid_id)
                interval_list.append([start, global_idx])

    np.savez(
        str(meta_path),
        vids=np.array(vid_list),
        intervals=np.array(interval_list),
    )
    print(f"  LMDB done: {global_idx} frames, {len(vid_list)} videos → {lmdb_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess VCDB videos into jpg zips and LMDB for VSC22 training."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=Path,
        help="Path to the vsc_adapter directory (output of prepare_vcdb_for_VSC.py)",
    )
    parser.add_argument("--fps",     type=int, default=1,   help="Frames per second to extract")
    parser.add_argument("--workers", type=int, default=8,   help="Parallel workers for ffmpeg")
    parser.add_argument("--timeout", type=int, default=120, help="Per-video ffmpeg timeout (s)")
    parser.add_argument(
        "--skip_zips",
        action="store_true",
        help="Skip jpg zip creation (use if zips already exist)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    assert data_dir.exists(), f"data_dir not found: {data_dir}"

    # --- Step 1: MP4 → jpg zips -------------------------------------------
    if not args.skip_zips:
        build_jpg_zips(data_dir, fps=args.fps, workers=args.workers, timeout=args.timeout)
    else:
        print("Skipping jpg zip creation (--skip_zips).")

    # --- Step 2: jpg zips → LMDB ------------------------------------------
    # Read all video IDs from metadata
    query_ids = pd.read_csv(
        data_dir / "meta" / "train" / "train_query_metadata.csv"
    )["video_id"].tolist()
    ref_ids = pd.read_csv(
        data_dir / "meta" / "train" / "train_reference_metadata.csv"
    )["video_id"].tolist()
    all_ids = sorted(set(query_ids + ref_ids))

    # lmdb/vsc  — main training LMDB (all videos)
    build_lmdb(data_dir, "vsc", all_ids)

    # lmdb/train_vsc — same data, second handle for augmentation mixup
    # The training pipeline opens this separately from lmdb/vsc so both need
    # to exist as distinct directories, even though the content is identical.
    build_lmdb(data_dir, "train_vsc", all_ids)

    print("\nPreprocessing complete.")
    print(f"  jpg_zips  → {data_dir / 'jpg_zips'}")
    print(f"  lmdb/vsc  → {data_dir / 'lmdb' / 'vsc'}")
    print(f"  lmdb/train_vsc → {data_dir / 'lmdb' / 'train_vsc'}")


if __name__ == "__main__":
    main()
