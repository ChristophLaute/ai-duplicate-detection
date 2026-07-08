import argparse
import gc
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.FramePrep import compute_fragments_streaming
from runtime_config import VCDB_DIR


VIDEO_DIR = VCDB_DIR / "core_dataset"
SAVE_DIR = VCDB_DIR / "fragments"
META_DATA_SAVE_PATH = VCDB_DIR / "fragment_metadata.csv"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("VIDEO_DIR:", VIDEO_DIR)
print("Exists:", VIDEO_DIR.exists())
print("Is dir:", VIDEO_DIR.is_dir())


def compute_fragments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--len-frames", type=int, default=15, help="Number of frames per fragment")
    parser.add_argument("--target-size", type=int, nargs=2, default=(32, 32), help="Target size for frame resizing (width height)")
    parser.add_argument("--metadata-path", type=Path, default=META_DATA_SAVE_PATH, help="Path to save fragment metadata CSV")
    args = parser.parse_args()
    video_files = []
    for ext in ("*.mp4", "*.flv", "*.avi", "*.mov", "*.mkv"):
        video_files.extend(
            p for p in VIDEO_DIR.rglob(ext)
            if p.is_file() and not p.name.startswith("._")
        )

    print("Video files found:", len(video_files))

    empty_frag_count = 0
    saved_video_count = 0
    skipped_existing_count = 0
    error_count = 0
    metadata_records = []
    return_metadata = True

    for video_file in tqdm(video_files):
        relative_parent = video_file.relative_to(VIDEO_DIR).parent
        local_video_dir = args.save_dir / relative_parent / video_file.stem

        # Skip only if actual fragments already exist
        if local_video_dir.exists() and any(local_video_dir.glob("frag_*.npy")):
            skipped_existing_count += 1
            continue

        try:
            fragments, metadata = compute_fragments_streaming(
                video_path=video_file,
                frame_skip=1,
                len_frames=args.len_frames,
                target_size=args.target_size,
                window_size=50,
                return_metadata=return_metadata,
            )
            

            if len(fragments) == 0:
                empty_frag_count += 1
                continue


            local_video_dir.mkdir(parents=True, exist_ok=True)

            for index, frag in enumerate(fragments):
                frag_path = local_video_dir / f"frag_{index}.npy"
                np.save(frag_path, frag)
                frag_metadata = metadata[index]
                metadata_records.append({
    "video_rel_path": str(video_file.relative_to(VIDEO_DIR)),
    "video_id": str(video_file.relative_to(VIDEO_DIR).with_suffix("")),
    "frag_id": frag_metadata["frag_id"],
    "frag_path": str(frag_path),
    "start_sec": frag_metadata["start_sec"],
    "end_sec": frag_metadata["end_sec"],
    "first_frame_idx": frag_metadata["first_frame_idx"],
    "last_frame_idx": frag_metadata["last_frame_idx"],
})

            saved_video_count += 1

            del fragments
            gc.collect()

        except Exception as e:
            error_count += 1
            print("Error:", video_file, e)
    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(args.metadata_path, index=False)
    
    print("Saved videos:", saved_video_count)
    print("Skipped existing:", skipped_existing_count)
    print("Empty fragments:", empty_frag_count)
    print("Errors:", error_count)


if __name__ == "__main__":
    compute_fragments()
