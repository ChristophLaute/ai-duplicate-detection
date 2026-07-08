import gc
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.pytorch_model import SiameseNetwork
from runtime_config import VCDB_DIR, get_device

FRAGMENTS_DIR = VCDB_DIR / "fragments"
DESCRIPTORS_DIR = VCDB_DIR / "descriptors"
DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("best_siamese_model.pth")


def load_fragment_file(frag_path):
    return np.load(frag_path, allow_pickle=False)


def load_video_fragments(video_rel_path):
    video_dir = FRAGMENTS_DIR / video_rel_path
    frag_files = sorted(
        video_dir.glob("frag_*.npy"),
        key=lambda p: int(p.stem.split("_")[1])
    )

    if len(frag_files) == 0:
        return np.array([], dtype=np.uint8), []

    fragments = [load_fragment_file(f) for f in frag_files]
    fragments = np.stack(fragments, axis=0)
    return fragments, frag_files


def fragments_to_tensor(fragments):
    """
    Expected fragment shape per file: [T, H, W, C]
    After stacking: [N, T, H, W, C]
    Output tensor: [N, T, C, H, W]
    """
    x = fragments.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 1, 4, 2, 3))
    return torch.tensor(x, dtype=torch.float32)


def encode_frags_in_batches(model, fragments, device, batch_size=16, show_batch_bar=True):
    if len(fragments) == 0:
        return np.empty((0, 15, 64), dtype=np.float32)

    x = fragments_to_tensor(fragments)
    descriptors = []

    model.eval()
    with torch.no_grad():
        batch_indices = range(0, x.shape[0], batch_size)

        if show_batch_bar:
            batch_indices = tqdm(
                batch_indices,
                desc="  Batches",
                leave=False
            )

        for i in batch_indices:
            batch = x[i:i + batch_size].to(device)
            z = model.encode(batch)
            descriptors.append(z.cpu().numpy())

    return np.concatenate(descriptors, axis=0)


def find_fragment_video_dirs():
    """
    Only returns directories that directly contain frag_*.npy files.
    """
    dirs = {frag_file.parent for frag_file in FRAGMENTS_DIR.rglob("frag_*.npy")}
    return sorted(dirs)


def precompute_descriptors(batch_size=64, show_batch_bar=True):
    device = get_device()
    print("Using device:", device)

    model = SiameseNetwork()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    fragment_dirs = find_fragment_video_dirs()
    print("Total videos with fragments:", len(fragment_dirs))

    saved_count = 0
    skipped_count = 0
    error_count = 0
    empty_count = 0

    video_bar = tqdm(fragment_dirs, desc="Videos")

    for fragment_dir in video_bar:
        relative_path = fragment_dir.relative_to(FRAGMENTS_DIR)
        save_path = (DESCRIPTORS_DIR / relative_path).with_suffix(".npy")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            skipped_count += 1
            video_bar.set_postfix(
                saved=saved_count,
                skipped=skipped_count,
                empty=empty_count,
                errors=error_count
            )
            continue

        try:
            fragments, frag_paths = load_video_fragments(relative_path)

            if len(fragments) == 0:
                empty_count += 1
                video_bar.set_postfix(
                    saved=saved_count,
                    skipped=skipped_count,
                    empty=empty_count,
                    errors=error_count
                )
                continue

            descriptors = encode_frags_in_batches(
                model=model,
                fragments=fragments,
                device=device,
                batch_size=batch_size,
                show_batch_bar=show_batch_bar
            )

            payload = {
                "descriptors": descriptors,
                "num_fragments": int(descriptors.shape[0]),
                "fragment_files": [str(p) for p in frag_paths],
            }

            np.save(save_path, payload, allow_pickle=True)
            saved_count += 1

            del fragments, frag_paths, descriptors, payload
            gc.collect()

        except Exception as e:
            error_count += 1
            print(f"\nError processing {relative_path}: {e}")

        video_bar.set_postfix(
            saved=saved_count,
            skipped=skipped_count,
            empty=empty_count,
            errors=error_count
        )

    print(
        f"Saved: {saved_count}, "
        f"Skipped: {skipped_count}, "
        f"Empty: {empty_count}, "
        f"Errors: {error_count}"
    )


if __name__ == "__main__":
    precompute_descriptors(batch_size=64, show_batch_bar=True)
