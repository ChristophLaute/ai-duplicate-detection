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
from datasets.generators import ToTensorNormalize
import utils



VIDEO_EXTS = ("*.mp4", "*.mkv", "*.webm", "*.flv", "*.mov")

def load_id_filter(ids_file: Path | None) -> set[str] | None:
    if ids_file is None:
        return None
    with ids_file.open() as f:
        return {line.strip() for line in f if line.strip()}


def list_video_files(video_dir: Path, id_filter: set[str] | None) -> list[Path]:
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
    frames =utils.load_video(str(video_path), fps=1, cc_size=224, rs_size=256)
    if frames is None or len(frames) == 0:
        return None
    frames = ToTensorNormalize()(frames) # (C, T, H, W)
    frames = frames.permute(1,0,2,3).to(device)  # (T, C, H, W) for batch processing
    with torch.no_grad():
        features = model(frames).cpu()
    return features

def extract_all_features(video_files, model, device, save_dir):
    empty = 0
    saved = 0
    errors = 0
    skipped = 0
    t0 = time.perf_counter()
    for video_file in tqdm(video_files, desc="Processing videos"):
        save_path = save_dir / f"{video_file.stem}.pt"
        if save_path.exists():
            skipped += 1
            continue
        try:
            features = extract_features(video_file, model, device)
            if features is None:
                    empty += 1
                    continue
            torch.save(features, save_path)
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
    
    parser = argparse.ArgumentParser(description="Extract VVS features for a FIVR video subset.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing FIVR video files.")
    parser.add_argument("--save-path", type=Path, required=True, help="Directory to save .pt feature files.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional file with one video ID per line to process.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.save_path.mkdir(parents=True, exist_ok=True)

    model = FeatureExtractor(network="resnet50").to(device)
    model.eval()

    id_filter = load_id_filter(args.ids_file)
    video_files = list_video_files(args.video_dir, id_filter)
    video_files = video_files[args.offset: args.offset + args.limit if args.limit else None]
    print(f"Processing {len(video_files)} videos (offset={args.offset}, limit={args.limit})")

    extract_all_features(video_files, model, device, args.save_path)
