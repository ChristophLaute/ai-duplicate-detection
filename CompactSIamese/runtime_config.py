import os
from functools import lru_cache
from pathlib import Path

import torch


def _existing_path(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_data_roots():
    user = os.getenv("USER", "")
    roots = []

    env_root = os.getenv("THESIS_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root))

    if user:
        roots.extend(
            [
                Path(f"/var/scratch/{user}/thesis/data"),
                Path(f"/scratch-shared/{user}/thesis/data"),
                Path(f"/home/{user}/thesis/data"),
            ]
        )

    roots.extend(
        [
            Path("/Volumes/LaCie/DuplicateDetectionThesis"),
            Path("/Volumes/LaCie"),
        ]
    )
    return roots


def resolve_vcdb_dir():
    env_vcdb_dir = os.getenv("VCDB_DIR")
    if env_vcdb_dir:
        return Path(env_vcdb_dir)

    candidates = []
    for root in _default_data_roots():
        candidates.extend(
            [
                root / "VCDB",
                root / "data" / "VCDB",
            ]
        )

    resolved = _existing_path(candidates)
    if resolved is not None:
        return resolved
    return candidates[0]


def resolve_fivr_dir():
    env = os.getenv("FIVR_DIR")
    if env:
        return Path(env)

    candidates = []
    for root in _default_data_roots():
        candidates.extend(
            [
                root / "FIVR-200K",
                root / "data" / "FIVR-200K",
            ]
        )

    resolved = _existing_path(candidates)
    if resolved is not None:
        return resolved
    return candidates[0]


VCDB_DIR = resolve_vcdb_dir()
FIVR_DIR = resolve_fivr_dir()
VCDB_CORE_DIR = VCDB_DIR / "core_dataset"
VIDEO_EXTS = {".mp4", ".flv", ".avi", ".mov", ".mkv", ".webm"}


@lru_cache(maxsize=1)
def build_video_index():
    video_index = {}
    for video_path in VCDB_CORE_DIR.rglob("*"):
        if (
            video_path.is_file()
            and video_path.suffix.lower() in VIDEO_EXTS
            and not video_path.name.startswith("._")
        ):
            rel_id = video_path.relative_to(VCDB_CORE_DIR).with_suffix("").as_posix()
            video_index[rel_id] = video_path
    return video_index


def video_path_to_id(video_path):
    return video_path.relative_to(VCDB_CORE_DIR).with_suffix("").as_posix()


def resolve_video_path(video_name):
    target_name = Path(video_name).name
    target_stem = Path(video_name).stem
    matches = []

    for rel_id, video_path in build_video_index().items():
        if video_path.name == target_name or video_path.stem == target_stem or rel_id == video_name:
            matches.append(video_path)

    if len(matches) == 1:
        return matches[0]
    return None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
