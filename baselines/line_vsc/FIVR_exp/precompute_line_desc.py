import argparse
import gc
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import time
import os
import tempfile
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "CompactSIamese"))

from runtime_config import FIVR_DIR, get_device
from data.precompute_fivr_fragments import list_video_files, load_id_filter

BASELINE_SRC = (
    Path(__file__).resolve().parent.parent
    / "meta-vsc-descriptor-runtime/submission_src"
)

sys.path.insert(0, str(BASELINE_SRC))

from src.model import create_model_in_runtime, create_copy_type_pred_model
from src.inference_impl import run_inference, VideoDataset, batch_forward
from src.vsc.storage import store_features

VIDEO_DIR = FIVR_DIR / "videos"
DESCRIPTORS_DIR = FIVR_DIR / "line_descriptors"
DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_EXTS = ("*.mp4", "*.mkv", "*.webm", "*.flv", "*.mov")
FPS = 2.0
DEVICE = get_device()

def compute_descriptors_batch(video_Dir, model, device, batch_size=128, preprocessor=None, tta=False, copytype_model=None):
    dataset = VideoDataset(str(video_Dir), fps=FPS, read_type="tensor", batch_size=batch_size, extensions=["mp4","mkv","webm","flv","mov"])
    loader = DataLoader(dataset, batch_size=None)
    descriptors = list(run_inference(
        model=model,
        dataloader=loader,
        transforms=preprocessor,
        tta=tta,
        copytype_model=copytype_model,
    ))
    if not descriptors:
        return None
    
    return descriptors
def compute_descriptors_line(video_path, model, device, batch_size=128, alpha=1, preprocessor=None, tta=False, copytype_model=None):
    """Computes descriptors for a video with the 3rd place solution of meta. Since it expects the video to be a directory we create a symlink"""
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path = Path(temp_dir)
        os.symlink(video_path, tmp_path / video_path.name)
        dataset = VideoDataset(str(tmp_path), fps=FPS, read_type="tensor", batch_size=batch_size, extensions=["mp4","mkv","webm","flv","mov"])

        loader = DataLoader(dataset, batch_size=None)
        descriptors = list(run_inference(
            model=model,
            dataloader=loader,
            transforms=preprocessor,
            tta=tta,
            copytype_model=copytype_model,
        ))
        if not descriptors:
            return None
        descriptor_dict = descriptors[0]
        norm = (1 / (descriptor_dict.feature @ descriptor_dict.feature.T).mean()) ** alpha # normalization like they did over all videoss
        descriptor  = descriptor_dict.feature * norm


        
        
    return {"descriptor":descriptor, "timestamps": descriptor_dict.timestamps}

def compute_all_descriptors_line(video_files, model, device, batch_size=128, descriptor_save_dir=DESCRIPTORS_DIR, alpha=1, preprocessor=None, tta=False, copytype_model=None):
    """Same as for siamese"""
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
            descriptor = compute_descriptors_line(video_file, model, device, batch_size, preprocessor=preprocessor, alpha=alpha, tta=tta, copytype_model=copytype_model)
            if descriptor is None:
                    empty += 1
                    continue
            np.save(save_path, {"descriptor": descriptor["descriptor"], "timestamps": descriptor["timestamps"]}, allow_pickle=True)

            saved += 1
            del descriptor
            gc.collect()
        except Exception as e:
             errors += 1
             tqdm.write(f"Error processing {video_file}: {e}")
        

    t1 = time.perf_counter()
    tqdm.write(f"Finished processing videos in {t1 - t0:.2f} seconds.")
    tqdm.write(f"Saved: {saved}, Skipped: {skipped}, Empty: {empty}, Errors: {errors}")
    tqdm.write(f"mean time per video: {(t1 - t0) / len(video_files):.2f} seconds.")

if __name__ == "__main__":
    device = DEVICE
    parser = argparse.ArgumentParser(description="Compute descriptors for all videos in the dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for descriptor computation.")
    parser.add_argument("--descriptor_save_dir", type=Path, default=DESCRIPTORS_DIR, help="Directory to save the computed descriptors.")
    parser.add_argument("--offset", type=int, default=0, help="Start index into the sorted video list.")
    parser.add_argument("--limit", type=int, default=None, help="Number of videos to process.")
    parser.add_argument("--video-dir", type=Path, default=None, help="Pre-built directory of videos for batch mode.")
    parser.add_argument("--output", type=Path, default=None, help="Output .npz path for batch mode.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Optional file with one video ID per line to filter processing.")
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation (requires copy_type_pred model).")
    args = parser.parse_args()

    os.chdir(BASELINE_SRC)  # ensure relative imports work correctly
    model, preprocessor = create_model_in_runtime(transforms_device=device)
    model = model.to(device).eval().half()

    copytype_model = None
    if args.tta:
        copytype_model = create_copy_type_pred_model(transforms_device=device)
        copytype_model = copytype_model.to(device).eval()

    os.chdir(Path(__file__).parent)  # switch back to original dir for output

    print(f"Using device: {device}, TTA: {args.tta}")

    if args.video_dir and args.output:
        print("Batch processing")
        features = compute_descriptors_batch(args.video_dir, model, device, batch_size=args.batch_size, preprocessor=preprocessor, tta=args.tta, copytype_model=copytype_model)
        print(f"computed {len(features)} features for {args.video_dir}")
        with open(args.output, "wb") as f:
            store_features(f, features)
    else:
        print("Processing individual videos")
        scan_dir = args.video_dir if args.video_dir else VIDEO_DIR
        id_filter = load_id_filter(args.ids_file)
        video_files = list_video_files(scan_dir, id_filter)
        video_files = video_files[args.offset: args.offset + args.limit if args.limit else None]
        print(f"Processing {len(video_files)} videos (offset={args.offset}, limit={args.limit})")

        compute_all_descriptors_line(video_files, model, device, batch_size=args.batch_size, preprocessor=preprocessor, alpha=1, descriptor_save_dir=args.descriptor_save_dir, tta=args.tta, copytype_model=copytype_model)
        