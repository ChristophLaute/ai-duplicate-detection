from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from tqdm import tqdm

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".flv", ".mov")


def load_annotation_ids(annotation_path, labels):
    """
    Load the annotation JSON and return two sets:
    - query_ids: IDs of all query videos
    - required_ids: IDs of all query videos and their related videos for the specified labels
    """
    annotations = json.loads(annotation_path.read_text())
    query_ids = set(annotations.keys())

    related_ids = set(query_ids)
    for rels in annotations.values():
        for label in labels:
            related_ids.update(rels.get(label, []))

    return query_ids, related_ids


def load_dataset_ids(ids_path):
    """Load the dataset IDs from a text file and return them as a set of strings."""
    return {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}


def collect_required_and_random_videos(
    video_dir,
    required_ids,
    candidate_random_ids,
    num_random):
    """
    Find all required videos exactly, and collect random extras opportunistically
    by taking the first eligible non-required videos encountered during the scan.
    This is much faster than trying to match an exact random-ID sample from metadata.
    """
    allowed_exts = {ext.lower() for ext in VIDEO_EXTS}
    required_relpaths = {}
    random_relpaths = {}
    remaining_required = set(required_ids)

    progress = tqdm(
        total=len(required_ids) + num_random,
        desc="Collecting FIVR subset videos",
        unit="video",
    )
    scanned = 0
    for path in video_dir.iterdir():
        scanned += 1
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() not in allowed_exts:
            continue

        video_id = path.stem
        relpath = str(Path("videos") / path.name)

        if video_id in remaining_required:
            required_relpaths[video_id] = relpath
            remaining_required.remove(video_id)
            progress.update(1)
        elif (
            len(random_relpaths) < num_random
            and video_id in candidate_random_ids
            and video_id not in random_relpaths
        ):
            random_relpaths[video_id] = relpath
            progress.update(1)

        progress.set_postfix(
            scanned=scanned,
            required_found=len(required_relpaths),
            required_missing=len(remaining_required),
            random_found=len(random_relpaths),
        )

        if not remaining_required and len(random_relpaths) >= num_random:
            break

    progress.close()
    return required_relpaths, random_relpaths


def write_lines(path, lines):
    """Write a list of lines to a text file, ensuring the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create an upload subset for FIVR")
    parser.add_argument(
        "--fivr-root",
        type=Path,
        default=Path("/Volumes/LaCie/FIVR-200K"),
        help="Local FIVR dataset root containing videos/",
    )
    parser.add_argument(
        "--annotation-json",
        type=Path,
        default=Path("datasets/FIVR-200K/dataset/annotation.json"),
        help="Official FIVR annotation JSON",
    )
    parser.add_argument(
        "--youtube-ids",
        type=Path,
        default=Path("datasets/FIVR-200K/dataset/youtube_ids.txt"),
        help="Official FIVR youtube_ids.txt",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["ND", "DS"],
        help="Relation labels to include in the deterministic core subset",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=1000,
        help="How many random extra videos to sample from the remaining pool",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the extra sample",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fivr_subset"),
        help="Where to write the subset files",
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Only write selected video IDs from metadata and skip local file scanning",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not args.annotation_json.exists():
        raise FileNotFoundError(f"annotation.json not found: {args.annotation_json}")
    if not args.youtube_ids.exists():
        raise FileNotFoundError(f"youtube_ids.txt not found: {args.youtube_ids}")

    labels = tuple(args.labels)
    query_ids, required_ids = load_annotation_ids(args.annotation_json, labels)
    all_dataset_ids = load_dataset_ids(args.youtube_ids)

    ids_path = output_dir / "fivr_subset_ids.txt"
    manifest_path = output_dir / "fivr_subset_rsync_files.txt"
    missing_path = output_dir / "fivr_subset_missing_local_ids.txt"
    summary_path = output_dir / "fivr_subset_summary.json"

    candidate_random_ids = sorted(all_dataset_ids - required_ids)
    rng = random.Random(args.seed)
    sampled_random_ids = sorted(rng.sample(candidate_random_ids, k=min(args.num_random, len(candidate_random_ids))))
    required_ids_sorted = sorted(required_ids)
    selected_ids = required_ids_sorted + sampled_random_ids

    if args.ids_only:
        write_lines(ids_path, selected_ids)
        summary_path.write_text(
            json.dumps(
                {
                    "labels_included": list(labels),
                    "query_count": len(query_ids),
                    "query_or_related_count": len(required_ids),
                    "random_extra_count_requested": args.num_random,
                    "random_extra_count_selected": len(sampled_random_ids),
                    "total_selected_ids": len(selected_ids),
                    "local_scan_performed": False,
                    "seed": args.seed,
                },
                indent=2,
            )
            + "\n"
        )

        print(f"Labels included:            {list(labels)}")
        print(f"Query videos:               {len(query_ids)}")
        print(f"Query + related videos:     {len(required_ids)}")
        print(f"Random extra videos:        {len(sampled_random_ids)}")
        print(f"Total selected IDs:         {len(selected_ids)}")
        print(f"IDs file:                   {ids_path}")
        print(f"Summary JSON:               {summary_path}")
        print("Skipped local video scan. No rsync manifest was written.")
        return

    video_dir = args.fivr_root / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"FIVR videos directory not found: {video_dir}")

    candidate_random_id_set = set(sampled_random_ids)
    print(
        f"Scanning {video_dir} for all {len(required_ids)} required videos "
        f"and up to {len(sampled_random_ids)} selected extra videos ..."
    )
    required_relpaths, random_relpaths = collect_required_and_random_videos(
        video_dir,
        required_ids=required_ids,
        candidate_random_ids=candidate_random_id_set,
        num_random=len(sampled_random_ids),
    )

    selected_relpaths = []
    missing_local_ids = []
    random_ids_sorted = sorted(random_relpaths.keys())

    for video_id in required_ids_sorted:
        relpath = required_relpaths.get(video_id)
        if relpath is None:
            missing_local_ids.append(video_id)
        else:
            selected_relpaths.append(relpath)

    for video_id in random_ids_sorted:
        selected_relpaths.append(random_relpaths[video_id])

    write_lines(ids_path, selected_ids)
    write_lines(manifest_path, sorted(selected_relpaths))
    write_lines(missing_path, missing_local_ids)
    summary_path.write_text(
        json.dumps(
            {
                "labels_included": list(labels),
                "query_count": len(query_ids),
                "query_or_related_count": len(required_ids),
                "random_extra_count_requested": args.num_random,
                "random_extra_count_selected": len(sampled_random_ids),
                "random_extra_count_found_locally": len(random_relpaths),
                "total_selected_ids": len(selected_ids),
                "selected_files_found_locally": len(selected_relpaths),
                "missing_local_ids": len(missing_local_ids),
                "local_scan_performed": True,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"FIVR root:                 {args.fivr_root}")
    print(f"Video dir:                  {video_dir}")
    print(f"Labels included:            {list(labels)}")
    print(f"Query videos:               {len(query_ids)}")
    print(f"Query + related videos:     {len(required_ids)}")
    print(f"Random extra videos chosen: {len(sampled_random_ids)}")
    print(f"Random extra videos found:  {len(random_relpaths)}")
    print(f"Total selected IDs:         {len(selected_ids)}")
    print(f"Files found locally:        {len(selected_relpaths)}")
    print(f"Missing local IDs:          {len(missing_local_ids)}")
    print(f"IDs file:                   {ids_path}")
    print(f"Rsync manifest:             {manifest_path}")
    print(f"Missing IDs file:           {missing_path}")
    print(f"Summary JSON:               {summary_path}")


if __name__ == "__main__":
    main()
