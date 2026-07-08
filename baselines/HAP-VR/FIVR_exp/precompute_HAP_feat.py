import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from CompactSIamese.runtime_config import FIVR_DIR, get_device
import utils as hap_utils
from model.feature_extractor import FeatureExtractor

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm


VIDEO_DIR = FIVR_DIR / "videos"
#FRAGMENTS_DIR = FIVR_DIR / "fragments "
DESCRIPTORS_DIR = FIVR_DIR / "HAP_Features"
DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(__file__).parent.parent / "ckpt" / "model.pth"
VIDEO_EXTS = ("*.mp4", "*.mkv", "*.webm", "*.flv", "*.mov")


if __name__ == "__main__":
    


    device = get_device()
    print(f"Using device: {device}")
    feat_ext = FeatureExtractor["RESNET"].get_model(512)