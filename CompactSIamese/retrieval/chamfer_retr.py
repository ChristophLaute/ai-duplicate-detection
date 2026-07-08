import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime_config import VCDB_DIR

DESCR_DIR = VCDB_DIR / "descriptors"


def load_descriptor_file(descriptor_path):
    """Load a descriptor file and return the 'descriptors' key as a numpy array."""
    obj = np.load(descriptor_path, allow_pickle=True).item()
    if "descriptors" not in obj:
        raise KeyError(f"'descriptors' key not found in {descriptor_path}")
    return obj["descriptors"]


def relative_video_id(video_path):
    """Return the relative video ID for a given descriptor file path."""
    return video_path.relative_to(DESCR_DIR).with_suffix("").as_posix()


def construct_fragment_db():
    """Construct a database of all fragments from descriptor files in DESCR_DIR.
    Each entry in the database is a dictionary containing:
    - 'video_id': the relative video ID
    - 'frag_id': the fragment index
    - 'descriptor': the fragment descriptor as a numpy array
    - 'path': the path to the descriptor file
    """
    db = []
    if not DESCR_DIR.exists():
        raise FileNotFoundError(f"Descriptor directory not found: {DESCR_DIR}")

    all_descriptor_files = sorted(DESCR_DIR.rglob("*.npy"))

    for path in all_descriptor_files:
        desc = load_descriptor_file(path)   # shape: (num_fragments, ?, feat_dim)
        video_id = relative_video_id(path)

        for frag_id, frag_desc in enumerate(desc):
            db.append({
                "video_id": video_id,
                "frag_id": frag_id,
                "descriptor": frag_desc,
                "path": path,
            })

    return db

def construct_video_db():
    """Construct a database of all videos from descriptor files in DESCR_DIR.
    Each entry in the database is a dictionary containing:
    - 'video_id': the relative video ID
    - 'descriptor': the video descriptor as a numpy array (flattened fragments)
    - 'path': the path to the descriptor file
    """
    db = {}
    for path in sorted(DESCR_DIR.rglob("*.npy")):
        desc = load_descriptor_file(path)   # shape: (num_fragments, ?, feat_dim)
        video_id = relative_video_id(path)
        db[video_id] = desc.reshape(-1, desc.shape[-1])  # flatten to (num_frags * ?, feat_dim)
    return db


def chamfer_sim(desc1, desc2):
    """
    Symmetric Chamfer score between two sets of frame embeddings using
    Euclidean distance"""
    diff = desc1[:, None, :] - desc2[None, :, :]          # (n1, n2, d)
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-8)  # (n1, n2)

    min_dist_1 = np.min(dist_matrix, axis=1).mean()
    min_dist_2 = np.min(dist_matrix, axis=0).mean()

    return -float((min_dist_1 + min_dist_2) / 2.0) #negative so higher == bette

def get_video_scores_chamfer(query_desc_path, video_db):
    """
    Compute Chamfer similarity scores between a query video's descriptors and all videos in the database.
    Returns a dictionary mapping video IDs to their Chamfer similarity scores.
    """
    query_desc = load_descriptor_file(query_desc_path)  # (num_frags, ?, feat_dim)
    query_frames = query_desc.reshape(-1, query_desc.shape[-1])  # (num_frags * ?, feat_dim)

    video_scores = {}
    for video_id, cand_desc in video_db.items():
        video_scores [video_id] = chamfer_sim(query_frames, cand_desc)  # cand_desc is already flattened to (num_frags * ?, feat_dim)

    return video_scores

def get_top_k_videos_for_query_fragment(query_frag, frag_db, k=5):
    """
    For one query fragment:
    1. compare against all database fragments
    2. keep only the best matching fragment per video
    3. rank videos by that best score
    4. return top-k videos
    """
    best_per_video = {}

    for frag in frag_db:
        video_id = frag["video_id"]
        sim = chamfer_sim(query_frag, frag["descriptor"])

        if video_id not in best_per_video or sim > best_per_video[video_id]["score"]:
            best_per_video[video_id] = {
                "score": sim,
                "frag_id": frag["frag_id"],
                "path": frag["path"],
            }

    ranked_videos = sorted(
        best_per_video.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    top_k_videos = ranked_videos[:k]
    return top_k_videos, best_per_video


def get_video_scores_for_query_video(
    query_desc_path,
    frag_db,
    k=5,
    score_weight=0.7,
    coverage_weight=0.3,
):
    """
    For all query fragments:
    - get top-k videos per query fragment
    - count how often each video appears in those top-k lists
    - keep each video's best score across all query fragments
    - compute final score = score_weight * max_score + coverage_weight * coverage
    """
    query_desc = load_descriptor_file(query_desc_path)

    video_max_scores = {}
    hit_counts = {}

    for query_frag in query_desc:
        top_k_videos, best_per_video = get_top_k_videos_for_query_fragment(
            query_frag, frag_db, k=k
        )

        for video_id, info in best_per_video.items():
            score = info["score"]
            current_best = video_max_scores.get(video_id, -np.inf)
            if score > current_best:
                video_max_scores[video_id] = score

    return video_max_scores


def retrieve_similar_videos_chamfer(query_desc_path, video_db, top_n=5):
    """Retrieve the top-n most similar videos to a query video based on Chamfer distance."""
    video_scores = get_video_scores_chamfer(
        query_desc_path, video_db)

    sorted_videos = sorted(
        video_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_videos[:top_n]


if __name__ == "__main__":
    video_db = construct_video_db()

    query_files = sorted(DESCR_DIR.rglob("*.npy"))
    if not query_files:
        raise FileNotFoundError(f"No .npy descriptor files found in {DESCR_DIR}")

    query_desc_path = query_files[0]
    similar_videos = get_video_scores_chamfer(query_desc_path, video_db)
    similar_videos = sorted(similar_videos.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"Query video: {query_desc_path.stem}")
    print("Top retrieved videos:")
    for rank, (video_id, stats) in enumerate(similar_videos, start=1):
        print(
            f"{rank}. {video_id} | "
            f"score={stats:.4f}"
        )
