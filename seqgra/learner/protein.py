"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for amino acid sequence learners

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from seqgra.learner.learner import MultiClassClassificationLearner
from seqgra.learner.learner import MultiLabelClassificationLearner
from seqgra.parser.modelparser import ModelParser
from seqgra.learner.proteinhelper import ProteinHelper


class ProteinMultiClassClassificationLearner(MultiClassClassificationLearner):
    def __init__(self, parser: ModelParser, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(parser, data_dir, output_dir)

    def encode_x(self, x: List[str]):
        return np.stack([ProteinHelper.convert_dense_to_one_hot_encoding(seq)
                         for seq in x])

    def decode_x(self, x):
        return np.stack([ProteinHelper.convert_one_hot_to_dense_encoding(seq)
                         for seq in x])

    def encode_y(self, y: List[str]):
        if self.labels is None:
            raise Exception("unknown labels, call parse_data or "
                            "load_model first")
        labels = np.array(self.labels)
        return np.vstack([ex == labels for ex in y])

    def decode_y(self, y):
        if self.labels is None:
            raise Exception("unknown labels, call parse_data or "
                            "load_model first")
        labels = np.array(self.labels)

        decoded_y = np.vstack([labels[ex] for ex in y])
        decoded_y = list(itertools.chain(*decoded_y))
        return decoded_y

    def parse_data(self, file_name: str) -> None:
        df = pd.read_csv(file_name, sep="\t")
        x: List[str] = df["x"].tolist()
        y: List[str] = df["y"].tolist()

        ProteinHelper.check_sequence(x)
        return (x, y)


class ProteinMultiLabelClassificationLearner(MultiLabelClassificationLearner):
    def __init__(self, parser: ModelParser, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(parser, data_dir, output_dir)

    def encode_x(self, x: List[str]):
        return np.stack([ProteinHelper.convert_dense_to_one_hot_encoding(seq)
                         for seq in x])

    def decode_x(self, x):
        return np.stack([ProteinHelper.convert_one_hot_to_dense_encoding(seq)
                         for seq in x])

    def encode_y(self, y: List[str]):
        if self.labels is None:
            raise Exception("unknown labels, call parse_data or "
                            "load_model first")

        y = [ex.split("|") for ex in y]
        mlb = MultiLabelBinarizer(classes=self.labels)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = mlb.fit_transform(y).astype(bool)
        return y

    def decode_y(self, y):
        if self.labels is None:
            raise Exception("unknown labels, call parse_data or "
                            "load_model first")
        labels = np.array(self.labels)

        decoded_y = [labels[ex] for ex in y]
        decoded_y = ["|".join(ex) for ex in decoded_y]
        return decoded_y

    def parse_data(self, file_name: str) -> None:
        df = pd.read_csv(file_name, sep="\t")
        x: List[str] = df["x"].tolist()
        y: List[str] = df["y"].replace(np.nan, "", regex=True).tolist()

        ProteinHelper.check_sequence(x)
        return (x, y)