import numpy as np
import torch
from torch.utils.data import Dataset




class VCDBSegmentPairDataset(Dataset):
    """Dataset class for segment pairs from VCDB returns both pairs and their labels and pair type"""
    def __init__(self, pairs_df):
        self.pairs = pairs_df.reset_index(drop=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]

        x1 = np.load(row.x1_path, allow_pickle=False)
        x2 = np.load(row.x2_path, allow_pickle=False)
        y = float(row.label)
        pair_type = row.get("pair_type")

        x1 = torch.from_numpy(x1).float().permute(0, 3, 1, 2) / 255.0 # normalizes
        x2 = torch.from_numpy(x2).float().permute(0, 3, 1, 2) / 255.0
        y = torch.tensor(y, dtype=torch.float32)

        return x1, x2, y, pair_type