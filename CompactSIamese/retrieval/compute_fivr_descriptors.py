import argparse
import gc
import shutil
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.pytorch_model import SiameseNetwork
from runtime_config import FIVR_DIR, get_device

FRAGMENTS_DIR = FIVR_DIR / "fragments"
DESCRIPTORS_DIR = FIVR_DIR / "descriptors"
DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = Path("best_siamese_model.pth")


def fragments_to_tensor(fragments: np.ndarray) -> torch.Tensor:
    """(N, T, H, W, C) uint8 → (N, T, C, H, W) float32 in [0,1]"""
    x = fragments.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 1, 4, 2, 3))
    return torch.tensor(x, dtype=torch.float32)


def encode_fragments(model, fragments: np.ndarray, device, batch_size: int) -> np.ndarray:
    x = fragments_to_tensor(fragments)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i + batch_size].to(device)
            out.append(model.encode(batch).cpu().numpy())
    return np.concatenate(out, axis=0)


def load_id_filter(ids_file: Path | None) -> set[str] | None:
    if ids_file is None:
        return None
    with ids_file.open() as f:
        return {line.strip() for line in f if line.strip()}


def list_video_dirs(fragments_dir: Path, id_filter: set[str] | None) -> list[Path]:
    video_dirs = sorted(
        d for d in fragments_dir.iterdir()
        if d.is_dir() and any(d.glob("frag_*.npy"))
    )
    if id_filter is not None:
        video_dirs = [d for d in video_dirs if d.name in id_filter]
    return video_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH),
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fragments-dir", type=Path, default=FRAGMENTS_DIR,
                        help="Directory containing fragment subfolders")
    parser.add_argument("--descriptors-dir", type=Path, default=DESCRIPTORS_DIR,
                        help="Directory where descriptor files will be written")
    parser.add_argument("--ids-file", type=Path,
                        help="Optional text file with one video ID per line")
    parser.add_argument("--limit", type=int,
                        help="Optional cap on number of fragment directories to process")
    parser.add_argument("--delete-fragments", action="store_true",
                        help="Delete each video's fragment folder after its descriptor is saved")
    parser.add_argument("--T", type=int, default=15, help="Number of frames per fragment")
    args = parser.parse_args()

    args.descriptors_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    model = SiameseNetwork(T=args.T)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    id_filter = load_id_filter(args.ids_file)
    video_dirs = list_video_dirs(args.fragments_dir, id_filter)
    if args.limit is not None:
        video_dirs = video_dirs[:args.limit]

    print(f"Fragments dir: {args.fragments_dir}")
    print(f"Descriptors:   {args.descriptors_dir}")
    if args.ids_file is not None:
        print(f"ID filter:     {args.ids_file}")
    print(f"Videos with fragments: {len(video_dirs)}")

    saved = skipped = empty = errors = 0

    for video_dir in tqdm(video_dirs, desc="Videos"):
        video_id = video_dir.name
        save_path = args.descriptors_dir / f"{video_id}.npy"

        if save_path.exists():
            skipped += 1
            continue

        try:
            frag_files = sorted(video_dir.glob("frag_*.npy"),
                                key=lambda p: int(p.stem.split("_")[1]))
            if not frag_files:
                empty += 1
                continue

            fragments = np.stack([np.load(f, allow_pickle=False) for f in frag_files])
            descriptors = encode_fragments(model, fragments, device, args.batch_size)

            np.save(save_path, {
                "descriptors": descriptors,
                "num_fragments": int(descriptors.shape[0]),
                "fragment_files": [str(f) for f in frag_files],
            }, allow_pickle=True)

            saved += 1
            del fragments, descriptors
            gc.collect()

            if args.delete_fragments:
                shutil.rmtree(video_dir)

        except Exception as e:
            errors += 1
            tqdm.write(f"Error {video_id}: {e}")

    print(f"Saved: {saved}  Skipped: {skipped}  Empty: {empty}  Errors: {errors}")
