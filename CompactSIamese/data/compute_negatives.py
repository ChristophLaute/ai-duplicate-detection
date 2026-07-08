import argparse
import sys
import csv
import cv2

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from precompute_vcdb_segment_pairs import get_segment_fragments, time_to_seconds


from runtime_config import VCDB_DIR, VCDB_CORE_DIR, build_video_index, resolve_video_path, video_path_to_id

ANNOTATION_DIR = VCDB_DIR / "annotation"
VIDEO_DIR = VCDB_CORE_DIR
_DEFAULT_SAVE_DIR = VCDB_DIR / "negative_pairs"
_DEFAULT_FRAG_META = VCDB_DIR / "fragment_metadata.csv"
_DEFAULT_POS_PAIRS = VCDB_DIR / "positive_pairs" / "positive_pairs.csv"

ALL_VIDEOS = list(build_video_index().values())
DURATION_CACHE = {}
BAD_VIDEOS = set()


def ann_parser(ann_file):
    """Parses a the string from the annotations into the individual parts"""
    video1_name = ann_file[0].strip()
    video2_name = ann_file[1].strip()

    start1 = ann_file[2].strip()
    end1 = ann_file[3].strip()          
    start2 = ann_file[4].strip()
    end2 = ann_file[5].strip()
    return video1_name, video2_name, start1, end1, start2, end2

def get_video_duration(video_path):
    """Returns the duration of the video in seconds"""
    cache_key = str(video_path)
    if cache_key in DURATION_CACHE:
        return DURATION_CACHE[cache_key]
    if cache_key in BAD_VIDEOS:
        return 0.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        BAD_VIDEOS.add(cache_key)
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps is None or fps <= 1e-6:
        fps = 25.0

    duration = float(frame_count) / float(fps)
    if duration <= 0:
        BAD_VIDEOS.add(cache_key)
        return 0.0

    DURATION_CACHE[cache_key] = duration
    return duration

def seconds_to_time(t):
    """Converts seconds to HH:MM:SS.sss format"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def choose_random_video_segment(all_videos, old_video, duration):
    """chooses a random sengment from the video that is unequal to the old video and has same duration"""
    while True:
        random_video = np.random.choice(all_videos)
        if random_video.name != old_video and str(random_video) not in BAD_VIDEOS:
            break
   
    video_name = random_video.name

    len_video = get_video_duration(random_video)  
    if len_video <= 0:
        BAD_VIDEOS.add(str(random_video))
        return None
    start_sec = np.random.uniform(0, max(0, len_video - duration))
    end_sec = start_sec + duration
    start_time = seconds_to_time(start_sec)
    end_time = seconds_to_time(end_sec)

    return video_name, start_time, end_time

def get_annotation_files(annotation_dir):
    """returns a list of all annotation files while excluding weird mac ._ files"""
    return [p for p in annotation_dir.glob("*.txt") if not p.name.startswith("._")]

def get_easy_negatives(annotation_dir):
    random_ann = np.random.choice(get_annotation_files(annotation_dir))
    all_videos = ALL_VIDEOS
    with open(random_ann, "r", encoding="latin-1") as f:
        rows = list(csv.reader(f))
        chosen_row = rows[np.random.randint(len(rows))]
        video1_name, _, start1, end1, _, _ = ann_parser(chosen_row)
        duration = time_to_seconds(end1) - time_to_seconds(start1)
        random_segment = choose_random_video_segment(all_videos, video1_name, duration)
        if random_segment is None:
            return None
        video2_name, start2, end2 = random_segment
        return(video1_name, video2_name, start1, end1, start2, end2)
    

def overlaps(start_a, end_a, start_b, end_b):
    """helper function to check overlap"""
    return max(start_a, start_b) < min(end_a, end_b)

def get_hard_negatives(annotation_dir):
    """Choosed hard negatives by randomly choosing a different segment from same video. It checks if there is no overlap with original segment to have it cleanly."""
    random_ann = np.random.choice(get_annotation_files(annotation_dir))
    with open(random_ann, "r", encoding="latin-1") as f:
        rows = list(csv.reader(f))
        chosen_row = rows[np.random.randint(len(rows))]
        video1_name, video2_name, start1, end1, start2, end2 = ann_parser(chosen_row)

    video2_path = resolve_video_path(video2_name)
    if video2_path is None:
        return None

    len_video = get_video_duration(video2_path)
    original_start = time_to_seconds(start2)
    original_end = time_to_seconds(end2)
    duration = original_end - original_start

    if duration <= 0 or len_video <= duration:
        return None

    for _ in range(100):
        new_start = np.random.uniform(0, max(0, len_video - duration))
        new_end = new_start + duration
        if not overlaps(original_start, original_end, new_start, new_end):
            return (video1_name,video2_name,start1,end1,seconds_to_time(new_start),seconds_to_time(new_end))

    return None
    



def construct_negatives(annotation_dir, hard_negatives=0.5, pos_pairs_csv=None):
    """Construct negative pairs so the final mix matches the requested hard ratio."""
    if pos_pairs_csv is None:
        pos_pairs_csv = _DEFAULT_POS_PAIRS
    len_pos_df = len(pd.read_csv(pos_pairs_csv))
    target_hard = int(round(len_pos_df * hard_negatives))
    target_hard = min(max(target_hard, 0), len_pos_df)
    target_easy = len_pos_df - target_hard

    hard_negative_list = []
    easy_negative_list = []
    progress = tqdm(total=len_pos_df, desc="Sampling negatives", unit="pair")

    while len(hard_negative_list) < target_hard or len(easy_negative_list) < target_easy:
        need_hard = len(hard_negative_list) < target_hard
        need_easy = len(easy_negative_list) < target_easy

        if need_hard and need_easy:
            choose_hard = np.random.rand() < hard_negatives
        else:
            choose_hard = need_hard

        if choose_hard:
            hard_segment_pair = get_hard_negatives(annotation_dir)
            if hard_segment_pair is not None:
                hard_negative_list.append(hard_segment_pair)
                progress.update(1)
        else:
            easy_segment_pair = get_easy_negatives(annotation_dir)
            if easy_segment_pair is not None:
                easy_negative_list.append(easy_segment_pair)
                progress.update(1)

        progress.set_postfix(
            hard=len(hard_negative_list),
            easy=len(easy_negative_list),
            bad=len(BAD_VIDEOS),
        )

    progress.close()

    negatives = hard_negative_list + easy_negative_list
    np.random.shuffle(negatives)
    return negatives, hard_negative_list, easy_negative_list





def construction(hard_negative_ratio=0.5, save_dir=None, fragment_meta_df=None, pos_pairs_csv=None):
    """main loop for construction of negative pairs. It first samples the requested number of hard and easy negatives and then extracts the corresponding fragments and saves them as np arrays with the corresponding labels and metadata."""
    if save_dir is None:
        save_dir = _DEFAULT_SAVE_DIR
    if fragment_meta_df is None:
        fragment_meta_df = pd.read_csv(_DEFAULT_FRAG_META)
    save_dir.mkdir(parents=True, exist_ok=True)
    records = []
    pair_id = 0
    negatives, hard_negatives, easy_negatives = construct_negatives(
        ANNOTATION_DIR,
        hard_negatives=hard_negative_ratio,
        pos_pairs_csv=pos_pairs_csv,
    )
    ### main looop 

    for row in tqdm(negatives, desc="Negative pairs"):
        if row is None or len(row) < 6:
            continue

        video1_name, video2_name, start1, end1, start2, end2 = [x.strip() for x in row]

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
        frags1 = get_segment_fragments(video1_id, start1_sec, end1_sec, fragment_meta_df)
        frags2 = get_segment_fragments(video2_id, start2_sec, end2_sec, fragment_meta_df)
        if len(frags1) == 0 or len(frags2) == 0:
            continue
        min_len = min(len(frags1), len(frags2))
        if min_len == 0:
            continue
        if row in hard_negatives:
            pair_type = "hard_negative"
        else:
            pair_type = "easy_negative"
        for i in range(min_len):
            f1_path = f"{frags1.iloc[i]['frag_path']}"
            f2_path = f"{frags2.iloc[i]['frag_path']}"
            records.append({
                "pair_id": pair_id,
                "x1_path": f1_path,
                "x2_path": f2_path,
                "label": 0.0,
                "video1": video1_id,
                "video2": video2_id,
                "pair_type": pair_type,
                "start1": start1,
                "end1": end1,
                "start2": start2,
                "end2": end2,
            })
            pair_id += 1

    df = pd.DataFrame(records)
    csv_path = save_dir / "negative_pairs.csv"
    df.to_csv(csv_path, index=False)

    print("Saved CSV:", csv_path)
    print("Total negative fragment pairs:", len(df))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=Path, default=_DEFAULT_SAVE_DIR,
                        help="Directory to save negative_pairs.csv")
    parser.add_argument("--frag-meta", type=Path, default=_DEFAULT_FRAG_META,
                        help="Path to fragment metadata CSV")
    parser.add_argument("--pos-pairs-csv", type=Path, default=_DEFAULT_POS_PAIRS,
                        help="Path to positive_pairs.csv (used to determine negative count)")
    args = parser.parse_args()

    fragment_meta_df = pd.read_csv(args.frag_meta)
    construction(
        save_dir=args.save_dir,
        fragment_meta_df=fragment_meta_df,
        pos_pairs_csv=args.pos_pairs_csv,
    )

if __name__ == "__main__":
    main()
