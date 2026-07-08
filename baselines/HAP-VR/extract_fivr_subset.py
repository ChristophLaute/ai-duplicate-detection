import argparse
import gc
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import time

sys.path.insert(0, str(Path(__file__).parent))

from model.feature_extractor import FeatureExtractor
import h5py
import utils



VIDEO_EXTS = ("*.mp4", "*.mkv", "*.webm", "*.flv", "*.mov")

def load_id_filter(ids_file):
    if ids_file is None:
        return None
    with ids_file.open() as f:
        return {line.strip() for line in f if line.strip()}
def validate_HDF5_file(hdf5_path):
    """checks if HDF5 is valid"""
    if not hdf5_path.exists():
        return True
    try:
        with h5py.File(hdf5_path, "r") as f:
            return True
    except OSError as e:
        return False
    

def list_video_files(video_dir, id_filter):
    video_files = []
    for ext in VIDEO_EXTS:
        video_files.extend(
            p for p in video_dir.glob(ext)
            if p.is_file() and not p.name.startswith("._")
        )
    if id_filter is not None:
        video_files = [p for p in video_files if p.stem in id_filter]
    return sorted(video_files)




def extract_features(video_path, model, device):
    frames =utils.load_video_opencv(str(video_path), fps=1, resize=256, crop=224) # T,224,224,3
    if frames is None or len(frames) == 0:
        return None
    
    frames = torch.from_numpy(frames).to(device)
    with torch.no_grad():
        features = model(frames).cpu()
    return features

def extract_all_features(video_files, model, device, hdf5_path):
    empty = 0
    saved = 0
    errors = 0
    skipped = 0
    t0 = time.perf_counter()
    hdf5_path = Path(hdf5_path)
    if not validate_HDF5_file(hdf5_path):
        print(f"Error: HDF5 file {hdf5_path} is not valid. Please fix or delete it before running the script.")
        backup = hdf5_path.with_suffix(".corr.hdf5")
        hdf5_path.rename(backup)
        print(f"The invalid file has been renamed to {backup}.")
    with h5py.File(hdf5_path, "a") as hdf5_file:
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_id = video_file.stem
            if video_id in hdf5_file:
                skipped += 1
                continue
            try:
                features = extract_features(video_file, model, device)
                if features is None:
                    empty += 1
                    continue
                hdf5_file.create_dataset(video_id, data=features.numpy(), compression="gzip")
                saved += 1
                gc.collect()
            except Exception as e:
                errors += 1
                tqdm.write(f"Error processing {video_file}: {e}")
    t1 = time.perf_counter()
    tqdm.write(f"Finished processing videos in {t1 - t0:.2f} seconds.")
    tqdm.write(f"Saved: {saved}, Skipped: {skipped}, Empty: {empty}, Errors: {errors}")
    tqdm.write(f"mean time per video: {(t1 - t0) / len(video_files):.2f} seconds.")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract HAP features for a FIVR video subset.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing FIVR video files.")
    parser.add_argument("--save-path", type=Path, required=True, help="Path to save the extracted features as an HDF5 file.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional file with one video ID per line to process.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    model =FeatureExtractor.RESNET.get_model(512).to(device)
    model.eval()

    id_filter = load_id_filter(args.ids_file)
    video_files = list_video_files(args.video_dir, id_filter)
    video_files = video_files[args.offset: args.offset + args.limit if args.limit else None]
    print(f"Processing {len(video_files)} videos (offset={args.offset}, limit={args.limit})")

    extract_all_features(video_files, model, device, args.save_path)
