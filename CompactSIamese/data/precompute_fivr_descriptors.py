import argparse
import gc
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.FramePrep import compute_fragments_streaming
from model.pytorch_model import SiameseNetwork
from runtime_config import FIVR_DIR, get_device
from data.precompute_fivr_fragments import load_id_filter, list_video_files
from retrieval.compute_fivr_descriptors import fragments_to_tensor, encode_fragments


VIDEO_DIR = FIVR_DIR / "videos"
#FRAGMENTS_DIR = FIVR_DIR / "fragments "
DESCRIPTORS_DIR = FIVR_DIR / "descriptors"
DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("best_siamese_model.pth")
VIDEO_EXTS = ("*.mp4", "*.mkv", "*.webm", "*.flv", "*.mov")


def compute_descriptors(video_path, model, device, batch_size=128, T=15, target_size=(32, 32)):
    fragments = compute_fragments_streaming(video_path=video_path, len_frames=T, target_size=target_size)
    if len(fragments) == 0:
        return False
    
    descriptors = encode_fragments(
        fragments=fragments,
        model=model,
        device=device,
        batch_size=batch_size
    )
    del fragments
    return descriptors

def compute_all_descriptors(video_files,model,device,batch_size = 128, descriptor_save_dir = DESCRIPTORS_DIR):
    empty = 0
    saved = 0
    errors = 0
    skipped = 0
    t0 = time.perf_counter()
    for video_file in tqdm(video_files, desc="Processing videos"):
        save_path = descriptor_save_dir / f"{video_file.stem}.npy"
        if save_path.exists():
            skipped += 1
            continue
        try:
            descriptors = compute_descriptors(video_file, model, device, batch_size)
            if descriptors is False:
                    empty += 1
                    continue
            np.save(save_path, {
                "descriptors": descriptors,
                "num_fragments": int(descriptors.shape[0]),
            }, allow_pickle=True)
            saved += 1
            del descriptors
            gc.collect()
        except Exception as e:
             errors += 1
             tqdm.write(f"Error processing {video_file}: {e}")
        

    t1 = time.perf_counter()
    tqdm.write(f"Finished processing videos in {t1 - t0:.2f} seconds.")
    tqdm.write(f"Saved: {saved}, Skipped: {skipped}, Empty: {empty}, Errors: {errors}")
    tqdm.write(f"mean time per video: {(t1 - t0) / len(video_files):.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute descriptors for all videos in the dataset.")
    parser.add_argument("--model_path", type=Path, default=MODEL_PATH, help="Path to the trained Siamese model.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for descriptor computation.")
    parser.add_argument("--descriptor_save_dir", type=Path, default=DESCRIPTORS_DIR, help="Directory to save the computed descriptors.")
    parser.add_argument("--offset", type=int, default=0, help="Start index into the sorted video list.")
    parser.add_argument("--limit", type=int, default=None, help="Number of videos to process.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional file with one video ID per line to process.")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    id_filter = load_id_filter(args.ids_file)
    video_files = list_video_files(VIDEO_DIR, id_filter)
    video_files = video_files[args.offset: args.offset + args.limit if args.limit else None]
    print(f"Processing {len(video_files)} videos (offset={args.offset}, limit={args.limit})")

    compute_all_descriptors(video_files, model, device, batch_size=args.batch_size)