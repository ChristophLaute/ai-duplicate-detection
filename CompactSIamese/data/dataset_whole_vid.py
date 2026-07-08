import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

class WholeVidDataset(Dataset):
    """Dataset class for whole video fragments from VCDB. Expects a CSV with columns: video_1, video_2, label"""
    def __init__(self,pairs_csv,fragment_dir):
        self.df = pd.read_csv(pairs_csv).reset_index(drop=True)
        self.fragment_dir = Path(fragment_dir)


    def __len__(self):
        return len(self.df)
    def __load_video_fragments(self,video_id):
        video_dir = self.fragment_dir / video_id
        frag_paths = sorted(video_dir.glob("frag_*.npy"))
        fragments = []
        for path in frag_paths:
            frag = np.load(path,allow_pickle=False) # THWC
            frag = torch.tensor(frag, dtype=torch.float32).permute(0,3,1,2) / 255.0  # TCHW
            fragments.append(frag)
        return fragments
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video1 = row["video_1"]
        video2 = row["video_2"]
        label = torch.tensor(row["label"], dtype=torch.float32)
        frags1 = self.__load_video_fragments(video1)
        frags2 = self.__load_video_fragments(video2)
        return frags1, frags2, label, video1, video2 # returns fragments and video_paths in order to track
