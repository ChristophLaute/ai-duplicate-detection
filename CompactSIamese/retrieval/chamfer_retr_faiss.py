import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import argparse

from retrieval.chamfer_retr import chamfer_sim
from retrieval.pool_embeddings import video_level_retrieval, pool_all_fragments,construct_faiss_index, SAVE_DIR


sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime_config import VCDB_DIR

DESCR_DIR = VCDB_DIR / "descriptors"
INDEX_DIR = VCDB_DIR / "frame_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = INDEX_DIR / "faiss_frame_index.bin"
METADATA_PATH = INDEX_DIR / "frame_metadata.pkl"
DESCR_CACHE_PATH = INDEX_DIR / "descriptor_cache.pkl"


def load_descriptor_file(descriptor_path):
    """"Load a descriptor file and return the 'descriptors' key as a numpy array."""
    obj = np.load(descriptor_path, allow_pickle=True).item()
    if "descriptors" not in obj:
        raise KeyError(f"'descriptors' key not found in {descriptor_path}")
    return obj["descriptors"]  # shape: (num_fragments, T, feat_dim)




def build_frame_index():
    """
    Builds flat L2 index on all frame vectors for fast retrieval of candidate videos.
    """
    all_descriptor_files = sorted(DESCR_DIR.rglob("*.npy"))
    if not all_descriptor_files:
        raise FileNotFoundError(f"No descriptor files in {DESCR_DIR}")

    vectors = []
    metadata = []       # metadata[i] = video_id of vectors[i]
    descriptor_cache = {}  

    print(f"Loading descriptors from {len(all_descriptor_files)} videos...")
    for path in all_descriptor_files:
        video_id = path.relative_to(DESCR_DIR).with_suffix("").as_posix()
        desc = load_descriptor_file(path)  # (num_frags, T, feat_dim)
        descriptor_cache[video_id] = desc

        for frag in desc:
            for frame_vec in frag:
                vectors.append(frame_vec.astype("float32"))
                metadata.append(video_id)

    vector_db = np.stack(vectors).astype("float32")
    n, d = vector_db.shape
    print(f"Total frame vectors: {n:,}  |  dim: {d}")

    
    index = faiss.IndexFlatL2(d)  
    
    index.add(vector_db)
    

    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    with open(DESCR_CACHE_PATH, "wb") as f:
        pickle.dump(descriptor_cache, f)

    print(f"Index built: {index.ntotal:,} vectors, saved to {INDEX_DIR}")


def load_frame_index():
    """Loads the FAISS index, metadata, and descriptor cache from disk."""
    for p in (INDEX_PATH, METADATA_PATH, DESCR_CACHE_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run build_frame_index() first.")

    index = faiss.read_index(str(INDEX_PATH))
    

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    with open(DESCR_CACHE_PATH, "rb") as f:
        descriptor_cache = pickle.load(f)

    return index, metadata, descriptor_cache




def get_candidates(query_desc, index, metadata, top_k_frames = 50):
    """
    Get candidate videos for one query video by retrieving top-K similar frames per query frame.
    """
    
    seen = {}

    for frag in query_desc:          # (T, d)
        frames = frag.astype("float32")
        _, indices = index.search(frames, top_k_frames)  # (T, top_k_frames)

        for row in indices:
            for idx in row:
                if 0 <= idx < len(metadata):
                    vid = metadata[idx]
                    if vid not in seen:
                        seen[vid] = None
    seen_vids = list(seen.keys())

    return seen_vids

def retrieve_similar_videos_chamfer_faiss(
    query_desc_path,
    descriptor_cache,
    top_n = 1000,
    top_k_videos = 50,
    top_k_per_fragment = None,
    max_candidates = 50,
    score_weight = 0.7,
    coverage_weight = 0.3,
    index = None,
    metadata = None,
    video_id_array = None,
    descr_dir = None,
    computed_descriptor = None,
):
    """
    Two-stage retrieval for one query video. Not used in final thesis. 

    Args:
        top_n:           How many results to return.
        top_k_frames:    FAISS nearest frames per query frame.
        max_candidates:  Cap on Stage-2 candidate videos.
    Returns:
        dict {video_id: final_score} for all candidates.
    """
    descr_dir = descr_dir or DESCR_DIR
    ### This part is the old retrieval on the frame level but I will introduce on fragment level like in pool_embeddings
    if computed_descriptor is  None:
        query_desc = load_descriptor_file(query_desc_path)  # (num_frags, T, d)
    else:
        query_desc = computed_descriptor
    query_video_id = query_desc_path.relative_to(descr_dir).with_suffix("").as_posix()


    if not (SAVE_DIR/"faiss_index.bin").exists():
        pool_all_fragments()
        construct_faiss_index()
    faiss_ranked = video_level_retrieval(query_video_id, top_k_per_fragment=top_k_per_fragment if top_k_per_fragment is not None else index.ntotal, index=index, metadata=metadata, video_id_array=video_id_array)
    candidates = faiss_ranked[:top_k_videos]
    rest = faiss_ranked[top_k_videos:]
    
    video_scores = {}

    for video in candidates:
        video_id = video["video_id"]
        if video_id not in descriptor_cache:
            continue
        cand_desc = descriptor_cache[video_id]  # (num_frags, T, d)

        query_frames = query_desc.reshape(-1, query_desc.shape[-1])  # (num_query_frags * T, d)
        cand_frames = cand_desc.reshape(-1, cand_desc.shape[-1])     # (
        max_score = chamfer_sim(query_frames, cand_frames)
        final_score = max_score  

        video_scores[video_id] = final_score
    reranked = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
    reranked_flat = [vid for vid, _ in reranked]
    total_ranking = reranked_flat + [entry["video_id"] for entry in rest ]

    return total_ranking




if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the FAISS index")
    parser.add_argument("--query", type=str, help="Path to a query descriptor .npy file")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--top-k-frames", type=int, default=50)
    parser.add_argument("--max-candidates", type=int, default=10000)
    args = parser.parse_args()

    if args.build:
        print("Building frame-level FAISS IVF+PQ index...")
        build_frame_index()

    if args.query:
        query_desc_path = Path(args.query)
        print(f"\nLoading index...")
        index, metadata, descriptor_cache = load_frame_index()

        print(f"Query: {query_desc_path.stem}")
        scores = retrieve_similar_videos_chamfer_faiss(
            query_desc_path=query_desc_path,
            index=index,
            metadata=metadata,
            descriptor_cache=descriptor_cache,
            top_n=args.top_n,
            top_k_frames=args.top_k_frames,
            max_candidates=args.max_candidates,
        )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {args.top_n} results:")
        for rank, (video_id, score) in enumerate(ranked[:args.top_n], start=1):
            print(f"  {rank:3d}. {video_id}  score={score:.4f}")
