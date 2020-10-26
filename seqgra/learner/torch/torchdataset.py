"""MIT - CSAIL - Gifford Lab - seqgra

PyTorch DataSet class

@author: Konstantin Krismer
"""
from typing import List

import torch
import numpy as np

from seqgra.learner import Learner


class MultiClassDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

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


class MultiLabelDataSet(torch.utils.data.Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

        self.x = np.array(self.x).astype(np.float32)

        if self.y is not None:
            if not isinstance(self.y, np.ndarray):
                self.y = np.array(self.y)

            if self.y.dtype == np.bool:
                self.y = self.y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]


class IterableMultiClassDataSet(torch.utils.data.IterableDataset):
    def __init__(self, file_name: str, learner: Learner,
                 contains_y: bool = True):
        self.file_name: str = file_name
        self.learner: Learner = learner
        self.contains_y: bool = contains_y

    def __iter__(self):
        with open(self.file_name, "r") as f:
            # skip header
            next(f)
            for line in f:
                cells: List[str] = line.split("\t")

                if len(cells) == 2 or (len(cells) == 1 and not self.contains_y):
                    # one hot encode input and labels
                    x = self.learner.encode_x([cells[0].strip()])[0]
                    if not isinstance(x, np.ndarray):
                        x = np.array(x)
                    x = x.astype(np.float32)

                    if self.contains_y:
                        y = self.learner.encode_y([cells[1].strip()])[0]
                        if not isinstance(y, np.ndarray):
                            y = np.array(y)
                        y = np.argmax(y.astype(np.int64))
                        yield x, y
                    else:
                        yield x
                else:
                    raise Exception("invalid example: " + line)


class IterableMultiLabelDataSet(torch.utils.data.IterableDataset):
    def __init__(self, file_name: str, learner: Learner,
                 contains_y: bool = True):
        self.file_name: str = file_name
        self.learner: Learner = learner
        self.contains_y: bool = contains_y

    def __iter__(self):
        with open(self.file_name, "r") as f:
            # skip header
            next(f)
            for line in f:
                cells: List[str] = line.split("\t")

                if len(cells) == 2 or (len(cells) == 1 and not self.contains_y):
                    # one hot encode input and labels
                    x = self.learner.encode_x([cells[0].strip()])[0]
                    if not isinstance(x, np.ndarray):
                        x = np.array(x)
                    x = x.astype(np.float32)

                    if self.contains_y:
                        y = self.learner.encode_y([cells[1].strip()])[0]
                        if not isinstance(y, np.ndarray):
                            y = np.array(y)
                        y = y.astype(np.int64)
                        yield x, y
                    else:
                        yield x
                else:
                    raise Exception("invalid example: " + line)
