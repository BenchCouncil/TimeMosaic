import os
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset_BLAST(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 scale=True, nan_mode='replace', return_mask=False, sample_ratio=None, seed=42):
        assert nan_mode in ('replace', 'mask'), "nan_mode must be 'replace' or 'mask'"
        self.nan_mode = nan_mode
        self.return_mask = bool(return_mask)

        # size: [seq_len, pred_len]
        if size is None:
            self.seq_len, self.pred_len = 96, 96
        else:
            assert len(size) == 2, "size should be [seq_len, pred_len]"
            self.seq_len, self.pred_len = size
        self.win = self.seq_len + self.pred_len

        assert flag in ['train', 'val'], "flag must be 'train' or 'val'"
        split_dir = 'train' if flag == 'train' else 'valid'

        shp = np.load(os.path.join(root_path, split_dir, 'shape.npy'))
        self.data = np.memmap(
            os.path.join(root_path, split_dir, 'data.dat'),
            dtype=np.float32, mode='r', shape=tuple(shp)
        )

        self.N, self.L = self.data.shape
        if self.L < self.win:
            raise ValueError(f"seq_len+pred_len={self.win} > {self.L}")

        # self.W = self.L - self.win + 1
        self.W = self.L // self.win
        self.total = self.N * self.W
        self.scale = scale
        
        # smaple
        if sample_ratio is None:
            sample_ratio = 1 if flag == 'train' else 1.0
        k = max(1, int(round(self.total * float(sample_ratio))))
        rng = np.random.RandomState(seed)
        self.idxs = rng.choice(self.total, size=k, replace=False)
        
        self.rng = np.random.RandomState(seed)
        self.min_tail = 1000
        self.min_density = 0.5


    def __len__(self):
        return len(self.idxs)


    def _index_to_row_start(self, index: int):
        row_id = index // self.W
        k = index % self.W
        s_begin = k * self.win
        return row_id, s_begin

    def _random_valid_begin(self, row: np.ndarray):
        finite = np.isfinite(row)
        if not finite.any():
            return None
        last = int(np.flatnonzero(finite)[-1])
        tlen = last + 1
        if tlen < max(self.win, self.min_tail):
            return None
        density = float(finite[:tlen].mean())
        if density < self.min_density:
            return None
        return int(self.rng.randint(0, tlen - self.win + 1))

    def __getitem__(self, index):
        index = int(self.idxs[index])
        row_id, s_begin = self._index_to_row_start(index)
        s_end = s_begin + self.seq_len
        p_end = s_end + self.pred_len

        row = self.data[row_id]  # shape: (L,)
        
        # add sample
        while True:
            rbegin = self._random_valid_begin(row)
            if rbegin is not None:
                s_begin = rbegin
                break
            row_id = int(self.rng.randint(0, self.N))
            row = self.data[row_id]
        # add sample

        s_end = s_begin + self.seq_len
        p_end = s_end + self.pred_len
                
        seq_x = row[s_begin:s_end].astype(np.float32, copy=False)
        seq_y = row[s_end:p_end].astype(np.float32, copy=False)

        if self.nan_mode == 'mask':
            mask_x = (~np.isnan(seq_x)).astype(np.int32)
            mask_y = (~np.isnan(seq_y)).astype(np.int32)

        seq_x = np.nan_to_num(seq_x, nan=0.0, posinf=0.0, neginf=0.0)
        seq_y = np.nan_to_num(seq_y, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.scale:
            mean = float(np.nanmean(row))
            std  = float(np.nanstd(row))
            if not np.isfinite(mean):
                mean = 0.0
            if not np.isfinite(std) or std < 1e-8:
                std = 1e-8

        if self.scale:
            seq_x = (seq_x - mean) / std
            seq_y = (seq_y - mean) / std

        seq_x = torch.from_numpy(seq_x.reshape(self.seq_len, 1))
        seq_y = torch.from_numpy(seq_y.reshape(self.pred_len, 1))

        if self.nan_mode == 'mask' and self.return_mask:
            mask_x = torch.from_numpy(mask_x.reshape(self.seq_len, 1))
            mask_y = torch.from_numpy(mask_y.reshape(self.pred_len, 1))
            return seq_x, seq_y, mask_x, mask_y

        return seq_x, seq_y
