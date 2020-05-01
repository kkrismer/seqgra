"""MIT - CSAIL - Gifford Lab - seqgra

PyTorch DataSet class

@author: Konstantin Krismer
"""
from typing import List, Optional

import torch
import numpy as np


class DNAMultiClassDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y=None,
                 labels: Optional[List[str]] = None):
        self.x = x
        self.y = y
        self.labels: Optional[List[str]] = labels

        self.x = np.array(self.x).astype(np.float32)

        if self.y is not None:
            if not isinstance(self.y, np.ndarray):
                self.y = np.array(self.y)

            if self.y.dtype == np.bool:
                self.y = np.argmax(self.y.astype(np.int64), axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]


class DNAMultiLabelDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y=None,
                 labels: Optional[List[str]] = None):
        self.x = x
        self.y = y
        self.labels: Optional[List[str]] = labels

        self.x = np.array(self.x).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]


class ProteinMultiClassDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y=None,
                 labels: Optional[List[str]] = None):
        self.x = x
        self.y = y
        self.labels: Optional[List[str]] = labels

        self.x = np.array(self.x).astype(np.float32)

        if self.y is not None:
            if not isinstance(self.y, np.ndarray):
                self.y = np.array(self.y)

            if self.y.dtype == np.bool:
                self.y = np.argmax(self.y.astype(np.int64), axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]


class ProteinMultiLabelDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y=None,
                 labels: Optional[List[str]] = None):
        self.x = x
        self.y = y
        self.labels: Optional[List[str]] = labels

        self.x = np.array(self.x).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]
