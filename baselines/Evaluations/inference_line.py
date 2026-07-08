import sys
import os
import json
import time
import torch
import faiss
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import normalize


sys.path.insert(0, str(Path(__file__).resolve().parent / "meta-vsc-descriptor-runtime/submission_src"))
from src.model import create_model_in_runtime, create_copy_type_pred_model
from src.postproc import sliding_pca
from src.vsc.index import VideoFeature
from src.vsc.candidates import CandidateGeneration, MaxScoreAggregation, ScoreAggregation
from src.vsc.storage import load_features

sys.path.insert(0, str(Path(__file__).resolve().parent / "FIVR_exp"))
from precompute_line_desc import compute_descriptors_line


EXTENSIONS = ["mp4", "webm", "mkv", "flv", "mov", "avi"]

THRESHOLDS = {
    "all": 0.8061,
    "nd":  0.8112,
}

class AvgScoreAggregation(ScoreAggregation):
    """aggregate scores by averaging the scores of all matches for a given reference video."""
    def aggregate(self, match):
        return np.mean([m.score for m in match.matches])

def compute_query_descriptor_online(video_path, model, device, copytype_model, preprocessor, pca_mat, low_var_dim, score_norm_index, beta=1.0, no_consistency_weight=False, no_temporal_concat=False, no_score_norm=False, not_tta=False):
    """Compute the video descriptor for a query video using the LINE model, with optional post-processing steps."""
    result = compute_descriptors_line(video_path, model, device, tta=not not_tta, copytype_model=copytype_model, preprocessor=preprocessor)
    if result is None:
        return None
    desc = result["descriptor"]
    timestamps = result["timestamps"]
    if no_consistency_weight:   
        c = 1.0
    else:
        c = float((desc @ desc.T).mean())
    if c == 0:
        return None
    vf = VideoFeature(video_id=video_path.stem, feature=(desc / c).astype(np.float32), timestamps=timestamps)
    if not no_temporal_concat:
        processed = sliding_pca([vf], pca_mat, stride=3)
        if not processed:
            return None
        vf = processed[0]
    if not no_score_norm:
        feat = np.delete(vf.feature, low_var_dim, axis=1)
        feat = normalize(feat)
        sim, _ = score_norm_index.search(feat.astype(np.float32), 1)
        feat = np.concatenate([feat, -beta * sim[:, :1]], axis=1)
    else:
        feat = normalize(vf.feature)
    feat = feat * c
    return VideoFeature(video_id=vf.video_id, timestamps=vf.timestamps, feature=feat)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc_dir",    type=str, required=True, help="Folder with references.npz, pca_matrix.faiss, etc.")
    parser.add_argument("--video_dir",   type=str, default=None,  help="Folder with query video files")
    parser.add_argument("--queries",     type=str, default=None,  help="Text file with query IDs")
    parser.add_argument("--query_desc",  type=str, default=None,  help="Precomputed query descriptors (.npz)")
    parser.add_argument("--output",      type=str, default="predictions.json")
    parser.add_argument("--mode",        type=str, default="all", choices=["all", "nd"])
    parser.add_argument("--aggregation_type", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--no_consistency_weight", action="store_true", default=False)
    parser.add_argument("--no_temporal_concat",    action="store_true", default=False)
    parser.add_argument("--no_score_norm",         action="store_true", default=False)
    parser.add_argument("--no_tta",                action="store_true", default=False)

    args = parser.parse_args()
    threshold = THRESHOLDS[args.mode]

    DESC_DIR = Path(args.desc_dir)
    VIDEO_DIR = Path(args.video_dir) if args.video_dir else None



    with open(DESC_DIR / "references.npz", "rb") as f:
        refs = load_features(f)

    # offline setup
    if args.video_dir:
        BASELINE_SRC = Path(__file__).resolve().parent.parent / "line_vsc/meta-vsc-descriptor-runtime/submission_src"
        os.chdir(BASELINE_SRC)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, preprocessor = create_model_in_runtime(transforms_device=device)
        model = model.to(device).eval().half()

        copytype_model = create_copy_type_pred_model(transforms_device=device)
        copytype_model = copytype_model.to(device).eval()
        os.chdir(Path(__file__).parent)
    if not args.no_temporal_concat:
        pca_mat = faiss.read_VectorTransform(str(DESC_DIR / "pca_matrix.faiss"))
    else:
        pca_mat = None
    low_var_dim = int(np.load(DESC_DIR / "low_var_dim.npy")[0])
    if not args.no_score_norm:
        score_norm_index = faiss.read_index(str(DESC_DIR / "score_norm_index.faiss"))
    else:
        score_norm_index = None

    n_ref = sum(len(r) for r in refs)
    ti1 = time.perf_counter()
    if args.aggregation_type == "max":
        cg = CandidateGeneration(refs, MaxScoreAggregation())
    else:
     
        cg = CandidateGeneration(refs, AvgScoreAggregation())
    ti2 = time.perf_counter()
    print(f"Index built in {ti2 - ti1:.2f}s")

    all_refs = {vf.video_id for vf in refs}
    db = all_refs
    if args.queries:
        with open(args.queries) as f:
            query_ids = [line.strip() for line in f if line.strip()]
        
    else:
        query_ids = list(db)
    

    online_times = []
    similarities = {}
    precomputed = {}
    if args.query_desc and Path(args.query_desc).exists():
        with open(args.query_desc, "rb") as f:
            precomputed = {vf.video_id: vf for vf in load_features(f)}
    for stem in tqdm(query_ids, desc="Evaluating queries"):
        t0 = time.perf_counter()
        if stem in precomputed:
            
            q_desc = precomputed[stem]
        elif VIDEO_DIR:
            video_path = next((VIDEO_DIR / f"{stem}.{ext}" for ext in EXTENSIONS if (VIDEO_DIR / f"{stem}.{ext}").exists()), None)
            if video_path is None:
                continue
            
            q_desc = compute_query_descriptor_online(video_path, model, device, copytype_model, preprocessor, pca_mat, low_var_dim, score_norm_index, beta=1.0, no_consistency_weight=args.no_consistency_weight, no_temporal_concat=args.no_temporal_concat, no_score_norm=args.no_score_norm, not_tta=args.no_tta)
            if q_desc is None:
                continue
        else:
            continue
        candidates = cg.query([q_desc], n_ref)
        t1 = time.perf_counter()
        similarities[q_desc.video_id] = {cand.ref_id: cand.score for cand in candidates}
        online_times.append(t1 - t0)

    print(f"Online: {sum(online_times):.2f}s total, {np.mean(online_times):.2f}s per query")

    predictions = {}
    for qid,refscore in similarities.items():
        predictions[qid] = {rid: float(score) for rid, score in refscore.items() if score >= threshold}
        
    with open(args.output, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
