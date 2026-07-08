import pickle
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime_config import VCDB_DIR

DESCR_DIR = VCDB_DIR / "descriptors"
SAVE_DIR = VCDB_DIR / "pooled_embeddings"

SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_descriptor_file(descriptor_path):
    """Load a descriptor from a .npy file and return the 'descriptors' key."""
    obj = np.load(descriptor_path, allow_pickle=True).item()
    if "descriptors" not in obj:
        raise KeyError(f"'descriptors' key not found in {descriptor_path}")
    return obj["descriptors"]


def pool_fragment(frag):
    """Pool a fragment's frame-level descriptors into a single vector by concatenating mean and max pooling."""
    mean_pool = np.mean(frag, axis=0)
    max_pool = np.max(frag, axis=0)
    pooled = np.concatenate([mean_pool, max_pool]).astype("float32")
    return pooled


def _encode_rel_id(relative_id):
    """Encode a relative video ID to be filesystem-safe by replacing '/' with '___'."""
    return relative_id.replace("/", "___")


def _decode_rel_id(encoded_rel_id):
    """Decode a filesystem-safe encoded video ID back to its original form by replacing '___' with '/'."""
    return encoded_rel_id.replace("___", "/")


def clear_old_pooled_embeddings():
    """Delete all existing pooled embedding files in the SAVE_DIR."""
    for path in SAVE_DIR.glob("*.npy"):
        path.unlink()

    index_path = SAVE_DIR / "faiss_index.bin"
    metadata_path = SAVE_DIR / "metadata.pkl"

    if index_path.exists():
        index_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()


def pool_all_fragments(clear_existing = False, descr_dir = None, save_dir = None):
    """
    Read all descriptor files from DESCR_DIR recursively, pool every fragment,
    and save each pooled fragment as:
        SAVE_DIR / "{encoded_relative_video_id}__frag{i}.npy"
    """
    if clear_existing:
        clear_old_pooled_embeddings()
    descr_dir = descr_dir or DESCR_DIR
    save_dir = save_dir or SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    total_descriptor_files = 0
    total_fragments = 0
    failed_files = []

    descriptor_files = sorted(descr_dir.rglob("*.npy"))
    print(f"Found descriptor files: {len(descriptor_files)}")

    for descriptor_file in descriptor_files:
        relative_video_id = descriptor_file.relative_to(descr_dir).with_suffix("").as_posix()
        encoded_video_id = _encode_rel_id(relative_video_id)
        video_stem = Path(relative_video_id).name
        frag_counter = 0

        try:
            descriptors = load_descriptor_file(descriptor_file)

            if descriptors.ndim == 2:
                pooled_embedding = pool_fragment(descriptors)
                pooled_path = save_dir / f"{encoded_video_id}__frag{frag_counter}.npy"
                np.save(pooled_path, pooled_embedding)
                frag_counter += 1

            elif descriptors.ndim == 3:
                for frag in descriptors:
                    pooled_embedding = pool_fragment(frag)
                    pooled_path = save_dir / f"{encoded_video_id}__frag{frag_counter}.npy"
                    np.save(pooled_path, pooled_embedding)
                    frag_counter += 1

            else:
                raise ValueError(
                    f"Unexpected descriptor shape {descriptors.shape} in {descriptor_file}"
                )

        except Exception as e:
            failed_files.append(
                {
                    "video_stem": video_stem,
                    "video_rel_id": relative_video_id,
                    "path": str(descriptor_file),
                    "error": str(e),
                }
            )

        print(f"Processed {video_stem}: saved {frag_counter} pooled fragments")
        total_descriptor_files += 1
        total_fragments += frag_counter

    print(f"Processed descriptor files: {total_descriptor_files}")
    print(f"Saved pooled fragments: {total_fragments}")
    print(f"Failed descriptor files: {len(failed_files)}")

    return failed_files


def construct_faiss_index(save_dir = None):
    """Construct a FAISS index from all pooled fragment embeddings in SAVE_DIR and save the index and metadata."""
    vectors = []
    metadata = []

    save_dir = save_dir or SAVE_DIR

    for path in sorted(save_dir.glob("*.npy")):
        stem = path.stem
        if "__frag" not in stem:
            continue

        encoded_video_id = stem.rsplit("__frag", 1)[0]
        video_rel_id = _decode_rel_id(encoded_video_id)
        video_stem = Path(video_rel_id).name

        vec = np.load(path).astype("float32")
        vectors.append(vec)
        metadata.append(
            {
                "video_stem": video_stem,
                "video_rel_id": video_rel_id,
                "path": str(path),
            }
        )

    if not vectors:
        raise ValueError(f"No pooled embeddings found in {save_dir}")

    vector_db = np.stack(vectors).astype("float32")

    index = faiss.IndexFlatL2(vector_db.shape[1])
    index.add(vector_db)

    faiss.write_index(index, str(save_dir / "faiss_index.bin"))
    with open(save_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    unique_stems = len({m["video_stem"] for m in metadata})
    unique_rel_ids = len({m["video_rel_id"] for m in metadata})
    print(f"Built FAISS index with {len(metadata)} pooled fragment vectors")
    print(f"Unique video stems in FAISS metadata: {unique_stems}")
    print(f"Unique relative video IDs in FAISS metadata: {unique_rel_ids}")


def load_index_and_metadata(save_dir = None ):
    """Load the FAISS index and metadata from SAVE_DIR."""
    save_dir = save_dir or SAVE_DIR
    index_path = save_dir / "faiss_index.bin"
    metadata_path = save_dir / "metadata.pkl"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    video_id_array = np.array([m["video_rel_id"] for m in metadata])

    return index, metadata, video_id_array


def retrieve_similar_fragments(query_vec, top_k = 5):
    """Retrieve the top-K most similar pooled fragment embeddings for a given query vector using the FAISS index."""
    index, metadata, _ = load_index_and_metadata()

    query_vec = query_vec.astype("float32").reshape(1, -1)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(metadata):
            results.append((metadata[idx], -float(score)))  # negate: lower L2 dist = higher score

    return results


def _query_paths_for_video_stem(video_stem, metadata):
    """Return the paths of all pooled fragment embeddings corresponding to a given video stem."""
    return sorted(
        m["path"]
        for m in metadata
        if m["video_stem"] == video_stem
    )


def retrieve_fragments_for_video(video_id, top_k = 5):
    """Retrieve the top-K most similar pooled fragment embeddings for all fragments of a given video ID."""
    _index, metadata, _ = load_index_and_metadata()
    query_paths = _query_paths_for_video_stem(video_id, metadata)

    all_results = []
    for query_path in query_paths:
        query_vec = np.load(query_path)
        results = retrieve_similar_fragments(query_vec, top_k=top_k)
        all_results.append((str(query_path), results))

    return all_results


def video_level_retrieval(query_video_id, top_k_per_fragment = 50, index=None, metadata=None, video_id_array=None, computed_descriptor=None):
    """Retrieve candidate videos for a given query video by aggregating fragment-level retrieval results."""
    if index is None or metadata is None:
        index, metadata, video_id_array = load_index_and_metadata()
    if video_id_array is None:
        video_id_array = np.array([m["video_rel_id"] for m in metadata])
    if computed_descriptor is  None:
        query_stem = Path(query_video_id).name
        query_paths = _query_paths_for_video_stem(query_stem, metadata)
        query_vecs = query_vecs = [np.load(p).astype("float32").reshape(1, -1) for p in query_paths]
    else:
        if computed_descriptor.ndim == 2:
            query_vecs = [pool_fragment(computed_descriptor).reshape(1, -1)]

        else:  # (num_frags, T, d)
            query_vecs = [pool_fragment(frag).reshape(1, -1) for frag in computed_descriptor]
            

    candidate_best = {}  # video_rel_id: max score across all fragments

    for query_vec in query_vecs:
        scores_arr, indices_arr = index.search(query_vec, top_k_per_fragment)

        idxs = indices_arr[0]
        scs = -scores_arr[0]  # negate L2 dist: higher is better

        valid = idxs >= 0
        idxs = idxs[valid]
        scs = scs[valid]

        hit_vids = video_id_array[idxs]

        not_query = hit_vids != query_video_id
        hit_vids = hit_vids[not_query]
        scs = scs[not_query]

        # per-video max score for this fragment
        unique_vids, inv_idx = np.unique(hit_vids, return_inverse=True)
        frag_max = np.full(len(unique_vids), -np.inf)
        np.maximum.at(frag_max, inv_idx, scs)

        for vid, score in zip(unique_vids, frag_max):
            if vid not in candidate_best or candidate_best[vid] < score:
                candidate_best[vid] = score

    ranked = [
        {"video_id": vid, "combined_score": score, "max_score": score, "mean_score": score, "num_hits": 1}
        for vid, score in candidate_best.items()
    ]
    ranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return ranked


if __name__ == "__main__":
    pool_all_fragments(clear_existing=True)
    construct_faiss_index()
