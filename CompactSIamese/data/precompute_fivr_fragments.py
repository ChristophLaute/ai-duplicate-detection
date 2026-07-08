import argparse
import gc
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))
from data.FramePrep import compute_fragments_streaming
from runtime_config import FIVR_DIR

VIDEO_DIR = FIVR_DIR / "videos"
SAVE_DIR = FIVR_DIR / "fragments"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR,
                        help="Directory containing flat FIVR video files")
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR,
                        help="Directory where fragment subfolders will be written")
    parser.add_argument("--ids-file", type=Path,
                        help="Optional text file with one video ID per line")
    parser.add_argument("--limit", type=int,
                        help="Optional cap on number of videos to process")
    parser.add_argument("--len-frames", type=int, default=15,
                        help="Number of frames per fragment")
    parser.add_argument("--target-size", type=int, nargs=2, default=(32, 32),
                        help="Target size for frame resizing (width height)")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    id_filter = load_id_filter(args.ids_file)
    video_files = list_video_files(args.video_dir, id_filter)
    if args.limit is not None:
        video_files = video_files[:args.limit]

    print(f"Video dir:       {args.video_dir}")
    print(f"Save dir:        {args.save_dir}")
    if args.ids_file is not None:
        print(f"ID filter:       {args.ids_file}")
    print(f"Videos found:    {len(video_files)}")

    saved = skipped = empty = errors = 0

    for video_file in tqdm(video_files, desc="Videos"):
        save_dir = args.save_dir / video_file.stem

        if save_dir.exists() and any(save_dir.glob("frag_*.npy")):
            skipped += 1
            continue

        try:
            fragments = compute_fragments_streaming(
                video_path=video_file,
                frame_skip=1,
                len_frames=args.len_frames,
                target_size=tuple(args.target_size),
                window_size=50,
            )

            if len(fragments) == 0:
                empty += 1
                continue

            save_dir.mkdir(parents=True, exist_ok=True)
            for i, frag in enumerate(fragments):
                np.save(save_dir / f"frag_{i}.npy", frag)

            saved += 1
            del fragments
            gc.collect()

        except Exception as e:
            errors += 1
            tqdm.write(f"Error {video_file.name}: {e}")

    print(f"Saved: {saved}  Skipped: {skipped}  Empty: {empty}  Errors: {errors}")
