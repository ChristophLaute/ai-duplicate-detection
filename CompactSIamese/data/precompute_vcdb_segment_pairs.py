import argparse
import csv
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


from runtime_config import VCDB_DIR, VCDB_CORE_DIR, resolve_video_path, video_path_to_id

ANNOTATION_DIR = VCDB_DIR / "annotation"
VIDEO_DIR = VCDB_CORE_DIR
_DEFAULT_SAVE_DIR = VCDB_DIR / "positive_pairs"
_DEFAULT_FRAG_META = VCDB_DIR / "fragment_metadata.csv"


def get_segment_fragments(video_id, start_sec, end_sec, fragment_meta):
    """Retrieve fragments for a specific video segment based on start and end times."""
    rows = fragment_meta[
        (fragment_meta["video_id"] == video_id) &
        (fragment_meta["start_sec"] >= start_sec) &
        (fragment_meta["end_sec"] <= end_sec)
    ].sort_values("frag_id")

    return rows.reset_index(drop=True)


def time_to_seconds(t):
    """Convert a time string in the format 'HH:MM:SS' to total seconds."""
    h, m, s = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frag-meta", type=Path, default=_DEFAULT_FRAG_META,
                        help="Path to fragment metadata CSV")
    parser.add_argument("--save-dir", type=Path, default=_DEFAULT_SAVE_DIR,
                        help="Directory to save positive_pairs.csv")
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    fragment_metadata_df = pd.read_csv(args.frag_meta)

    records = []
    pair_id = 0

    annotation_files = list(ANNOTATION_DIR.glob("*.txt"))

    for ann_file in tqdm(annotation_files, desc="Annotation files"):
        with open(ann_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) < 6:
                    continue

                video1_name = row[0].strip()
                video2_name = row[1].strip()
                start1 = row[2].strip()
                end1 = row[3].strip()
                start2 = row[4].strip()
                end2 = row[5].strip()

                video1_path = resolve_video_path(video1_name)
                video2_path = resolve_video_path(video2_name)

                if video1_path is None or video2_path is None:
                    continue

                video1_id = video_path_to_id(video1_path)
                video2_id = video_path_to_id(video2_path)
                start1_sec = time_to_seconds(start1)
                end1_sec = time_to_seconds(end1)
                start2_sec = time_to_seconds(start2)
                end2_sec = time_to_seconds(end2)

                frags1 = get_segment_fragments(video1_id, start1_sec, end1_sec, fragment_metadata_df)
                frags2 = get_segment_fragments(video2_id, start2_sec, end2_sec, fragment_metadata_df)
                if len(frags1) == 0 or len(frags2) == 0:
                    continue
                min_len = min(len(frags1), len(frags2))
                if min_len == 0:
                    continue
                for i in range(min_len):
                    f1_path = f"{frags1.iloc[i]['frag_path']}"
                    f2_path = f"{frags2.iloc[i]['frag_path']}"
                    records.append({
                        "pair_id": pair_id,
                        "x1_path": f1_path,
                        "x2_path": f2_path,
                        "label": 1.0,
                        "video1": video1_id,
                        "video2": video2_id,
                        "pair_type": "positive",
                        "start1": start1,
                        "end1": end1,
                        "start2": start2,
                        "end2": end2,
                    })
                    pair_id += 1

    df = pd.DataFrame(records)
    csv_path = args.save_dir / "positive_pairs.csv"
    df.to_csv(csv_path, index=False)

    print("Saved CSV:", csv_path)
    print("Total positive fragment pairs:", len(df))
