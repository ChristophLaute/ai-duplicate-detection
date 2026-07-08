import argparse
import json
import sys
import dataclasses
from pathlib import Path

import numpy as np
import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "CompactSIamese"))

BASELINE_SRC = Path(__file__).resolve().parent.parent / "meta-vsc-descriptor-runtime/submission_src"
sys.path.insert(0, str(BASELINE_SRC))

from src.vsc.index import VideoFeature
from src.vsc.storage import store_features
from src.postproc import sliding_pca_with_ref
from src.score_normalization import score_normalize_with_ref, transform_features, normalize
from src.vsc.candidates import CandidateGeneration, MaxScoreAggregation

FIVR_DIR = Path("/var/scratch/zzb857/thesis/data/FIVR-200K")
DESCRIPTORS_DIR = FIVR_DIR / "line_descriptors"
OUTPUT_DIR = FIVR_DIR / "line_descriptors_postprocessed" / "50k"
IDS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "CompactSIamese/fivr_subset_50k"


def load_descriptors(ids, descriptor_dir, no_consistency_weight=False):
    features = []
    cs = {}
    for vid in ids:
        stem = Path(vid).stem
        desc_path = descriptor_dir / f"{stem}.npy"
        if not desc_path.exists():
            continue
        desc_dic = np.load(desc_path, allow_pickle=True).item()
        desc = desc_dic["descriptor"]
        if not no_consistency_weight:
            c  = (desc @ desc.T).mean()
        else:
                c = 1.0
        desc_dic["descriptor"] = desc / c
        cs[stem] = c

        vf = VideoFeature(
            video_id=stem,
            timestamps=desc_dic["timestamps"],
            feature=desc_dic["descriptor"].astype(np.float32),
        )
        
        features.append(vf)
    return features,cs


def main():
    parser = argparse.ArgumentParser(description="Preprocess FIVR descriptors for line VSC evaluation")
    parser.add_argument("--no-consistency-weight", action="store_true", help="Disable adding consistency weight to features")
    parser.add_argument("--no-temporal-concat", action="store_true", help="Disable temporal concatenation of features")
    parser.add_argument("--no-score-norm", action="store_true", help="Disable score normalization")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to save processed descriptors and transform parameters")
    args = parser.parse_args()

    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr_5k = json.load(f)
    with open(IDS_DIR / "fivr_reference_ids.txt") as f:
        reference_ids = [k.strip() for k in f if k.strip()]
    with open(IDS_DIR / "fivr_noise_ids.txt") as f:
        noise_ids = [k.strip() for k in f if k.strip()]

    val_query_ids_path = IDS_DIR / "fivr_val_query_ids.txt"
    has_val_queries = val_query_ids_path.exists()
    if has_val_queries:
        with open(val_query_ids_path) as f:
            val_query_ids = [l.strip() for l in f if l.strip()]
    else:
        print("WARNING: fivr_val_query_ids.txt not found — val_queries.npz will not be created")

    query_ids = fivr_5k["query"]
    queries, cs_queries = load_descriptors(query_ids, DESCRIPTORS_DIR, no_consistency_weight=args.no_consistency_weight)
    references, cs_references = load_descriptors(reference_ids, DESCRIPTORS_DIR, no_consistency_weight=args.no_consistency_weight)
    noise, cs_noise = load_descriptors(noise_ids, DESCRIPTORS_DIR, no_consistency_weight=args.no_consistency_weight)

    # PCA fit on references+noise, applied to queries, val_queries, and references
    if not args.no_temporal_concat:
        video_features, pca_model = sliding_pca_with_ref(queries, references, noise, stride=3, n_components=512)
        queries_pca = video_features["query"]
        references_pca = video_features["ref"]
        noise_pca = video_features["noise"]
    else:
        queries_pca = queries
        references_pca = references
        noise_pca = noise

    if not args.no_score_norm:
        queries_out, references_out = score_normalize_with_ref(queries_pca, references_pca, score_norm_refs=noise_pca, beta=1.0)
    else:
        queries_out = [dataclasses.replace(vf, feature=normalize(vf.feature)) for vf in queries_pca]
        references_out = [dataclasses.replace(vf, feature=normalize(vf.feature)) for vf in references_pca]
    for vf in queries_out:
        vf.feature *= cs_queries[vf.video_id]
    for vf in references_out:
        vf.feature *= cs_references[vf.video_id]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "queries.npz", "wb") as f:
        store_features(f, queries_out)
    with open(output_dir / "references.npz", "wb") as f:
        store_features(f, references_out)
    print(f"Saved {len(queries_out)} test queries and {len(references_out)} references")

    # Save transform parameters needed for online query processing
    # 1. Fitted PCA matrix (faiss VectorTransform)
    if not args.no_temporal_concat:
        faiss.write_VectorTransform(pca_model, str(output_dir / "pca_matrix.faiss"))

    # 2. Low-variance dimension index removed during score normalisation
    sn_features = np.concatenate([ref.feature for ref in noise_pca], axis=0)
    low_var_dim = int(sn_features.var(axis=0).argmin())
    np.save(output_dir / "low_var_dim.npy", np.array([low_var_dim]))

    # 3. FAISS index built from noise refs (after dim removal + L2 norm) for score norm lookup
    if not args.no_score_norm:
        noise_for_index = [
            dataclasses.replace(vf, feature=np.delete(vf.feature, low_var_dim, axis=1))
            for vf in noise_pca
        ]
        noise_for_index = transform_features(noise_for_index, normalize)
        score_norm_index = CandidateGeneration(noise_for_index, MaxScoreAggregation()).index.index
        faiss.write_index(score_norm_index, str(output_dir / "score_norm_index.faiss"))

    print(f"Saved transform parameters to {output_dir}")

    # Val queries — same PCA/normalization applied independently
    if has_val_queries:
        val_queries, cs_val = load_descriptors(val_query_ids, DESCRIPTORS_DIR, no_consistency_weight=args.no_consistency_weight)
        if not args.no_temporal_concat:
            val_features, _ = sliding_pca_with_ref(val_queries, references, noise, stride=3, n_components=512)
            val_queries_pca = val_features["query"]
        else:
            val_queries_pca = val_queries
        if not args.no_score_norm:
            val_queries_out, _ = score_normalize_with_ref(val_queries_pca, references_pca, score_norm_refs=noise_pca, beta=1.0)
        else:
            val_queries_out = [dataclasses.replace(vf, feature=normalize(vf.feature)) for vf in val_queries_pca]
        for vf in val_queries_out:
            vf.feature *= cs_val[vf.video_id]
        with open(output_dir / "val_queries.npz", "wb") as f:
            store_features(f, val_queries_out)
        print(f"Saved {len(val_queries_out)} val queries")


if __name__ == "__main__":
    main()
