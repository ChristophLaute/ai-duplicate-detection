import sys
import json
import copy
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.model import SuppressionWeightGenerationStage, EasyDistractorEliminationStage
from datasets.generators import ToTensorNormalize
from arg import arg_func
import argparse




# ── Paths ─────────────────────────────────────────────────────────────────────
FIVR_DIR        = Path("/var/scratch/zzb857/thesis/data/FIVR-200K")
FRAME_FEATS_DIR = FIVR_DIR / "vvs_features"           
OUTPUT_DIR      = FIVR_DIR / "vvs_video_descriptors"  
CHECKPOINT      = Path("jobs/table_benchmark_dim_3840/model/m00080000.pth")  
IDS_DIR         = Path(__file__).resolve().parent.parent.parent / "CompactSIamese/fivr_subset_50k"

def main():
    args = argparse.Namespace(
    suppression="vvs",
    vvs_vinit="v_t",
    vvs_h_connect=True,
    vvs_sigmoid_T=512.0,
    vvs_sigmoid_T_tsm=1.0,
    vvs_sigmoid_T_ddm=1.0,
    vvs_tsm=True,
    vvs_tgm=True,
    vvs_ddm=True,
    refinement=True,
    base_normalize=True,
    pca_whitening=True,
    pca_reduction=None,
    gpu=0,
    feature_extract="resnet50_l4imac",
    feature_backbone="resnet50",
    feature_type="l4imac",
    retrieve_triplet=False,
    iterations=-1,
    cycles=1,
    imple_flag="eval",
    weight_triplet=1.0,
    weight_saliency=1.0,
    weight_frame=1.0,
    distractor_sampling_ratio="20_50",
    mag_opt=5,
    mag_thresh=40,
    neg_len=32,
    thresh_s=0.5,
    baseline=False,
    use_oracle=False,
    fg_gt_label=None,
)   
    loaded = torch.load(CHECKPOINT)
    tsm_tgm_model = SuppressionWeightGenerationStage(args)
    ddm_model = EasyDistractorEliminationStage(args)
    tsm_tgm_model.load_state_dict(loaded["tsm_tgm_model"], strict=False)
    tsm_tgm_model = tsm_tgm_model.cuda().eval()
    ddm_model.load_state_dict(loaded["ddm_model"], strict=False)
    ddm_model = ddm_model.cuda().eval()
    with open(IDS_DIR / "fivr5k_vid.json") as f:
        fivr_5k = json.load(f)
    query_ids = fivr_5k["query"]
    with open(IDS_DIR / "fivr_reference_ids.txt") as f:
        reference_ids = [k.strip() for k in f if k.strip()]
    all_ids = query_ids + reference_ids
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for vid in tqdm(all_ids):
        stem = Path(vid).stem
        pt_path = FRAME_FEATS_DIR / f"{stem}.pt"
        if not pt_path.exists():
            continue
        frame_feats = torch.load(pt_path).cuda().unsqueeze(0)
        with torch.no_grad():
            in_data = {"anchor": frame_feats}
            if args.vvs_ddm:
                ddm_out = ddm_model(in_data["anchor"])
                if (torch.sigmoid(ddm_out["confidence"]) > 0.5).float().sum() != 0:
                    mask = (torch.sigmoid(ddm_out["confidence"]) > 0.5)
                    in_data["anchor"] = in_data["anchor"][:,mask]
                
            while in_data["anchor"].shape[1] < 4:
                in_data["anchor"] = torch.cat([in_data["anchor"], in_data["anchor"]], dim=1)

            feats_out = tsm_tgm_model(in_data, is_anc_processed = False)
            torch.save(feats_out["features"].cpu(), OUTPUT_DIR / f"{stem}.pt")

if __name__ == "__main__":
    main()