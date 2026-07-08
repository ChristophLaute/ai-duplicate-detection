import csv
import sys
import random
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_whole_vid import WholeVidDataset
from runtime_config import VCDB_DIR, VCDB_CORE_DIR, build_video_index, resolve_video_path, video_path_to_id

ANNOTATION_DIR = VCDB_DIR / "annotation"
VIDEO_DIR = VCDB_CORE_DIR
OUTPUT_CSV = VCDB_DIR / "vcdb_whole_video_pairs.csv"

VIDEO_EXTS = {".mp4", ".flv", ".avi", ".mov", ".mkv"}


def create_video_index(video_dir):
    """Create a mapping from video IDs to their corresponding file paths in the specified directory."""
    return build_video_index()

def map_video_name_to_id(video_name, video_index):
    """Map a video name to its corresponding video ID using the provided index."""
    video_path = resolve_video_path(video_name)
    if video_path is None:
        return None
    return video_path_to_id(video_path)


def create_whole_video_test_pairs(annotation_dir, video_index, output_csv):
    """Create a CSV file containing positive and negative video pairs based on annotations."""
    random.seed(42)
    video_index = create_video_index(VIDEO_DIR)
    valid_videos = list(video_index.keys())
    print(f"Found {len(valid_videos)} valid videos in index.")
    pos_pairs = set()
    unresolved_rows = []
    neg_pairs = set()
    ann_files = list(annotation_dir.glob("*.txt"))
    print("Annotation files found:", len(ann_files))
    for ann_file in ann_files:
        with open(ann_file, "r",encoding='utf-8',errors = "ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    print(f"Skipping malformed row in {ann_file}: {row}")
                    continue
                video_1_name = row[0].strip()
                video_2_name = row[1].strip()
                video_1_id = map_video_name_to_id(video_1_name, video_index)
                video_2_id = map_video_name_to_id(video_2_name, video_index)
                if video_1_id is None or video_2_id is None:
                    print(f"Could not resolve video IDs for row in {ann_file}: {row}")
                    unresolved_rows.append(row)
                    continue
                if video_1_id == video_2_id:
                    print(f"Skipping self-pair in {ann_file}: {row}")
                    continue
                pair  = tuple(sorted((video_1_id, video_2_id)))
                pos_pairs.add(pair)
        print(f"unique positive pairs found in {ann_file}: {len(pos_pairs)}")
        print(f"unresolved rows in {ann_file}: {len(unresolved_rows)}")
        
    len_neg = len(pos_pairs)
    attempts = 0
    max_attempts = 100000
    while len(neg_pairs) < len_neg and attempts < max_attempts:
        vid1, vid2 = random.sample(valid_videos, 2)
        pair = tuple(sorted((vid1, vid2)))
        if pair not in pos_pairs:
            neg_pairs.add(pair)
        attempts += 1

    print(f"Generated {len(neg_pairs)}negative pairs")
    rows = []
    for pair in pos_pairs:
        rows.append({"video_1": pair[0], "video_2": pair[1], "label": 1})
    for pair in neg_pairs:
        rows.append({"video_1": pair[0], "video_2": pair[1], "label": 0})
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df, unresolved_rows

def simple_collate(batch):
    """Collate function for DataLoader that simply returns the batch as is."""
    return batch

def construct_dataloaders(batch_size=1):
    """Construct a DataLoader for the whole video dataset using the generated CSV file."""
    dataset = WholeVidDataset(OUTPUT_CSV, VCDB_DIR / "fragments")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=simple_collate)
    return dataloader


if __name__ == "__main__":    
    video_index = create_video_index(VIDEO_DIR)
    df, unresolved = create_whole_video_test_pairs(ANNOTATION_DIR, video_index, OUTPUT_CSV)
    



            

        
            
