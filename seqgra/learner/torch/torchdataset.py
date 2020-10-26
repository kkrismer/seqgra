"""MIT - CSAIL - Gifford Lab - seqgra

PyTorch DataSet class

@author: Konstantin Krismer
"""
from collections import deque
import random
from typing import Any, Deque, List, Tuple

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
    def __init__(self, file_name: str, learner: Learner, shuffle: bool = True,
                 contains_y: bool = True, cache_size: int = 10000):
        self.file_name: str = file_name
        self.learner: Learner = learner
        self.shuffle: bool = shuffle
        self.contains_y: bool = contains_y
        self.cache_size: int = cache_size
        self.cache: Deque[Tuple[Any, Any]] = deque()

    def __iter__(self):
        with open(self.file_name, "r") as f:
            # skip header
            next(f)

            x, y = self._get_next_example(f)
            while x is not None:
                if self.contains_y:
                    yield x, y
                else:
                    yield x

                x, y = self._get_next_example(f)

    def _get_next_example(self, file_handle) -> Tuple[Any, Any]:
        if not self.cache:
            # read next chunk in memory
            x_vec: List[str] = list()
            y_vec: List[str] = list()

            line: str = file_handle.readline()
            i = 1
            while line and i <= self.cache_size:
                cells: List[str] = line.split("\t")

                if len(cells) == 2 or (len(cells) == 1 and not self.contains_y):
                    x_vec.append(cells[0].strip())

                    if self.contains_y:
                        y_vec.append(cells[1].strip())
                else:
                    raise Exception("invalid example: " + line)

                line = file_handle.readline()
                i += 1

            if x_vec:
                # shuffle
                if self.shuffle:
                    if self.contains_y:
                        temp = list(zip(x_vec, y_vec))
                        random.shuffle(temp)
                        x_vec, y_vec = zip(*temp)
                    else:
                        random.shuffle(x_vec)

                # process chunk in memory
                encoded_x_vec = self.learner.encode_x(x_vec)
                if not isinstance(encoded_x_vec, np.ndarray):
                    encoded_x_vec = np.array(encoded_x_vec)
                encoded_x_vec = encoded_x_vec.astype(np.float32)

                if self.contains_y:
                    encoded_y_vec = self.learner.encode_y(y_vec)
                    if not isinstance(encoded_y_vec, np.ndarray):
                        encoded_y_vec = np.array(encoded_y_vec)
                    encoded_y_vec = np.argmax(encoded_y_vec.astype(np.int64),
                                              axis=1)

                # fill cache
                if self.contains_y:
                    for i in range(encoded_x_vec.shape[0]):
                        self.cache.append((encoded_x_vec[i, ...],
                                           encoded_y_vec[i]))
                else:
                    for i in range(encoded_x_vec.shape[0]):
                        self.cache.append((encoded_x_vec[i, ...], None))

        if self.cache:
            return self.cache.popleft()
        else:
            return (None, None)


class IterableMultiLabelDataSet(torch.utils.data.IterableDataset):
    def __init__(self, file_name: str, learner: Learner, shuffle: bool = True,
                 contains_y: bool = True, cache_size: int = 10000):
        self.file_name: str = file_name
        self.learner: Learner = learner
        self.shuffle: bool = shuffle
        self.contains_y: bool = contains_y
        self.cache_size: int = cache_size
        self.cache: Deque[Tuple[Any, Any]] = deque()

    def __iter__(self):
        with open(self.file_name, "r") as f:
            # skip header
            next(f)

            x, y = self._get_next_example(f)
            while x is not None:
                if self.contains_y:
                    yield x, y
                else:
                    yield x

                x, y = self._get_next_example(f)

    def _get_next_example(self, file_handle) -> Tuple[Any, Any]:
        if not self.cache:
            # read next chunk in memory
            x_vec: List[str] = list()
            y_vec: List[str] = list()

            line: str = file_handle.readline()
            i = 1
            while line and i <= self.cache_size:
                cells: List[str] = line.split("\t")

                if len(cells) == 2 or (len(cells) == 1 and not self.contains_y):
                    x_vec.append(cells[0].strip())

                    if self.contains_y:
                        y_vec.append(cells[1].strip())
                else:
                    raise Exception("invalid example: " + line)

                line = file_handle.readline()
                i += 1

            if x_vec:
                # shuffle
                if self.shuffle:
                    if self.contains_y:
                        temp = list(zip(x_vec, y_vec))
                        random.shuffle(temp)
                        x_vec, y_vec = zip(*temp)
                    else:
                        random.shuffle(x_vec)

                # process chunk in memory
                encoded_x_vec = self.learner.encode_x(x_vec)
                if not isinstance(encoded_x_vec, np.ndarray):
                    encoded_x_vec = np.array(encoded_x_vec)
                encoded_x_vec = encoded_x_vec.astype(np.float32)

                if self.contains_y:
                    encoded_y_vec = self.learner.encode_y(y_vec)
                    if not isinstance(encoded_y_vec, np.ndarray):
                        encoded_y_vec = np.array(encoded_y_vec)
                    encoded_y_vec = encoded_y_vec.astype(np.int64)

                # fill cache
                if self.contains_y:
                    for i in range(encoded_x_vec.shape[0]):
                        self.cache.append((encoded_x_vec[i, ...],
                                           encoded_y_vec[i]))
                else:
                    for i in range(encoded_x_vec.shape[0]):
                        self.cache.append((encoded_x_vec[i, ...], None))

        if self.cache:
            return self.cache.popleft()
        else:
            return (None, None)
