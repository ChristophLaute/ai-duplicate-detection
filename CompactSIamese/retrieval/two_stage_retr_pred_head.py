import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import argparse
import torch
import time

from retrieval.chamfer_retr import chamfer_sim
from retrieval.pool_embeddings import load_index_and_metadata, video_level_retrieval, pool_all_fragments,construct_faiss_index, SAVE_DIR



sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime_config import VCDB_DIR

DESCR_DIR = VCDB_DIR / "descriptors"
INDEX_DIR = VCDB_DIR / "frame_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = INDEX_DIR / "faiss_frame_index.bin"
METADATA_PATH = INDEX_DIR / "frame_metadata.pkl"
DESCR_CACHE_PATH = INDEX_DIR / "descriptor_cache.pkl"

def load_descriptor_file(descriptor_path):
    obj = np.load(descriptor_path, allow_pickle=True).item()
    if "descriptors" not in obj:
        raise KeyError(f"'descriptors' key not found in {descriptor_path}")
    return obj["descriptors"]  


def whole_video_logit_from_descriptors(frags1, frags2, model, device, aggregation="chamfer"):
    """computes whole video logit by comparing all fragment pairs and aggregating scores"""
    f1 = torch.tensor(np.stack(frags1), dtype=torch.float32).to(device)  # (n1, 15, 64)
    f2 = torch.tensor(np.stack(frags2), dtype=torch.float32).to(device)  # (n2, 15, 64)

    f1_comb = f1.unsqueeze(1).expand(-1,len(frags2),-1,-1)  # (n1, n2, 15, 64)
    f2_comb = f2.unsqueeze(0).expand(len(frags1),-1,-1,-1)  # (n1, n2, 15, 64)
    dists = torch.sqrt(torch.sum((f1_comb - f2_comb) ** 2, dim=-1) + 1e-8)  # (n1, n2, 15) euclidean distances in accordance with training
    dists = dists.reshape(-1, dists.shape[-1])
    with torch.no_grad():
        logits = model.prediction_head(dists).squeeze(-1)  # (n1*n2,)
    if aggregation == "chamfer":
        scores = logits.reshape(len(frags1), len(frags2))
        score_1to2 = scores.max(dim=1).values.mean()
        score_2to1 = scores.max(dim=0).values.mean()
        return ((score_1to2 + score_2to1) / 2).item()
    return logits.max().item()  # fallback: original max aggregation


def build_frame_index():
    """
    Builds flat L2 index on all frame vectors for fast retrieval of candidate videos.
    """
    all_descriptor_files = sorted(DESCR_DIR.rglob("*.npy"))
    if not all_descriptor_files:
        raise FileNotFoundError(f"No descriptor files in {DESCR_DIR}")

    vectors = []
    metadata = []       # metadata[i] = video_id of vectors[i]
    descriptor_cache = {}  # video_id -> descriptor array for Stage 2

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

    for frag in query_desc:         
        frames = frag.astype("float32")
        _, indices = index.search(frames, top_k_frames)  

        for row in indices:
            for idx in row:
                if 0 <= idx < len(metadata):
                    vid = metadata[idx]
                    if vid not in seen:
                        seen[vid] = None
    seen_vids = list(seen.keys())

    return seen_vids

def retrieve_similar_videos_pred_head(
    query_desc_path,
    descriptor_cache,
    model,
    device,
    top_k_videos = 50,
    top_k_per_fragment = None,
    index = None,
    metadata = None,
    video_id_array = None,
    descr_dir = None,
    aggregation = "chamfer",
    computed_descriptor = None,
):
    """Retrieves videos with the prediction head as a sevcond stage after FAISS retrieval"""
    descr_dir = descr_dir or DESCR_DIR
    if computed_descriptor is  None:
        query_desc = load_descriptor_file(query_desc_path)  # (num_frags, T, d)
    else:
        query_desc = computed_descriptor
    query_video_id = query_desc_path.relative_to(descr_dir).with_suffix("").as_posix()

    if not (SAVE_DIR/"faiss_index.bin").exists():
        pool_all_fragments()
        construct_faiss_index()
    if index is None or metadata is None:
        index, metadata, video_id_array = load_index_and_metadata()

    t = time.perf_counter()
    faiss_ranked = video_level_retrieval(query_video_id, top_k_per_fragment=top_k_per_fragment if top_k_per_fragment is not None else index.ntotal, index=index, metadata=metadata, video_id_array=video_id_array)
    print(f"FAISS stage: {time.perf_counter()-t:.2f}s")
    candidates = faiss_ranked[:top_k_videos]
    rest = faiss_ranked[top_k_videos:]
    video_scores = {}
    t = time.perf_counter()
    for i,video in enumerate(candidates):
        vid = video["video_id"]
        if vid in descriptor_cache:
            video_scores[vid] = whole_video_logit_from_descriptors(query_desc, descriptor_cache[vid], model, device, aggregation=aggregation)
    print(f"Prediction head reranking stage: {time.perf_counter()-t:.2f}s")
    reranked = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)
    reranked_flat = [vid for vid, _ in reranked]

    total_ranking = reranked_flat + [entry["video_id"] for entry in rest ]

    return total_ranking

def retrieve_similar_videos_pred_only(
    query_desc_path,
    descriptor_cache,
    model,
    device,
):
    """Retrieves videos with the prediction head only, without FAISS retrieval"""
    query_desc = load_descriptor_file(query_desc_path)  # (num_frags, T, d)
    query_video_id = query_desc_path.relative_to(DESCR_DIR).with_suffix("").as_posix()

    video_scores = {}
    for video_id, cand_desc in descriptor_cache.items():
        if video_id == query_video_id:
            continue
        video_scores[video_id] = whole_video_logit_from_descriptors(query_desc, cand_desc, model, device)

    return video_scores


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
        scores = retrieve_similar_videos_pred_head(
            query_desc_path=query_desc_path,
            index=index,
            metadata=metadata,
            descriptor_cache=descriptor_cache,
            top_n=args.top_n,
            top_k_frames=args.top_k_frames,
            max_candidates=args.max_candidates,
            model=model,
            device=device,
            computed_descriptor=None,
        )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {args.top_n} results:")
        for rank, (video_id, score) in enumerate(ranked[:args.top_n], start=1):
            print(f"  {rank:3d}. {video_id}  score={score:.4f}")
