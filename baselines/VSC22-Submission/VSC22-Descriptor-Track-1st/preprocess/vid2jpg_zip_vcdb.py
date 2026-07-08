import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import Pool
import os
import subprocess
import shutil
import zipfile

# =========================
# CONFIG: VCDB adapter paths on LaCie
# =========================
VCDB_DIR = "/Volumes/LaCie/DuplicateDetectionThesis/VCDB"
ADAPTER_DIR = f"{VCDB_DIR}/vsc_adapter"

TRAIN_META_DIR = f"{ADAPTER_DIR}/train/meta"
TRAIN_VIDEOS_DIR = f"{ADAPTER_DIR}/train/videos"
TRAIN_JPG_ZIPS_DIR = f"{ADAPTER_DIR}/train/jpg_zips"
SHARED_DIR = f"{ADAPTER_DIR}/shared"

NUM_WORKERS = 8
FPS = 1
OVERRIDE = False


def convert_mp4_to_zip_ffmpeg(
    mp4_path,
    zip_path,
    fps = 1,
    override = False,
):
    if os.path.isfile(zip_path):
        if override:
            os.remove(zip_path)
        else:
            return 0

    output_dir = os.path.dirname(zip_path)
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(output_dir, f"{os.path.basename(zip_path)[:-4]}_tmp")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = f'ffmpeg -nostdin -y -i "{mp4_path}" -start_number 0 -q 0 -vf fps={fps} "{tmp_dir}/%07d.jpg"'

    try:
        with open(os.devnull, "w") as null:
            subprocess.call(cmd, shell=True, timeout=120, stderr=null, stdout=null)
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return -1

    imgpaths = [
        os.path.join(tmp_dir, fn)
        for fn in sorted(os.listdir(tmp_dir))
        if fn.endswith(".jpg")
    ]

    if len(imgpaths) != 0:
        with zipfile.ZipFile(zip_path, "w") as wzip:
            for p in imgpaths:
                name = os.path.basename(p)
                wzip.write(p, arcname=name)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


def do_convert(args):
    token, idx, fps = args

    video_path = f"{TRAIN_VIDEOS_DIR}/{token}/{idx}.mp4"
    zip_path = f"{TRAIN_JPG_ZIPS_DIR}/{idx[-2:]}/{idx}.zip"

    if not os.path.exists(video_path):
        return (idx, "missing_video")

    status = convert_mp4_to_zip_ffmpeg(
        video_path,
        zip_path,
        fps=fps,
        override=OVERRIDE,
    )

    if status == 0:
        return (idx, "ok")
    else:
        return (idx, "failed")


def main():
    train_query_meta = f"{TRAIN_META_DIR}/train_query_metadata.csv"
    train_ref_meta = f"{TRAIN_META_DIR}/train_reference_metadata.csv"

    meta_query = pd.read_csv(train_query_meta)
    meta_ref = pd.read_csv(train_ref_meta)

    print("train_query_meta =", train_query_meta)
    print("train_ref_meta   =", train_ref_meta)
    print("num query rows   =", len(meta_query))
    print("num ref rows     =", len(meta_ref))
    print("first query ids  =", meta_query["video_id"].head().tolist())
    print("first ref ids    =", meta_ref["video_id"].head().tolist())

    meta = pd.concat([meta_query, meta_ref], ignore_index=True)

    video_list = meta["video_id"].tolist()

    args_list = []
    for idx in video_list:
        token = "query" if idx.startswith("Q") else "reference"
        args_list.append((token, idx, FPS))

    print("##### total videos to process:", len(video_list))

    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(do_convert, args_list), total=len(args_list)))

    ok = sum(1 for _, s in results if s == "ok")
    missing = [idx for idx, s in results if s == "missing_video"]
    failed = [idx for idx, s in results if s == "failed"]

    print(f"OK: {ok}")
    print(f"Missing videos: {len(missing)}")
    print(f"Failed conversions: {len(failed)}")

    if missing:
        with open(f"{SHARED_DIR}/vid2jpg_missing_videos.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(missing) + "\n")

    if failed:
        with open(f"{SHARED_DIR}/vid2jpg_failed_videos.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(failed) + "\n")


if __name__ == "__main__":
    main()