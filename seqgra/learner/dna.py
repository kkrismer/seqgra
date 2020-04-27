"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for learners

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List, Tuple
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from seqgra.learner import MultiClassClassificationLearner
from seqgra.learner import MultiLabelClassificationLearner
from seqgra.learner import DNAHelper
from seqgra.model import ModelDefinition


class DNAMultiClassClassificationLearner(MultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

    def encode_x(self, x: List[str]):
        return np.stack([DNAHelper.convert_dense_to_one_hot_encoding(seq)
                         for seq in x])

    def decode_x(self, x):
        return np.stack([DNAHelper.convert_one_hot_to_dense_encoding(seq)
                         for seq in x])

    def encode_y(self, y: List[str]):
        if self.definition.labels is None:
            raise Exception("unknown labels, call parse_examples_data or "
                            "load_model first")
        labels = np.array(self.definition.labels)
        return np.vstack([ex == labels for ex in y])

    def decode_y(self, y):
        if self.definition.labels is None:
            raise Exception("unknown labels, call parse_examples_data or "
                            "load_model first")
        labels = np.array(self.definition.labels)

        decoded_y = np.vstack([labels[ex] for ex in y])
        decoded_y = list(itertools.chain(*decoded_y))
        return decoded_y

    def parse_examples_data(self, file_name: str) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(file_name, sep="\t")
        x: List[str] = df["x"].tolist()
        y: List[str] = df["y"].tolist()

        DNAHelper.check_sequence(x)
        return (x, y)


class DNAMultiLabelClassificationLearner(MultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str) -> None:
        super().__init__(model_definition, data_dir, output_dir)

    def encode_x(self, x: List[str]):
        return np.stack([DNAHelper.convert_dense_to_one_hot_encoding(seq)
                         for seq in x])

    def decode_x(self, x):
        return np.stack([DNAHelper.convert_one_hot_to_dense_encoding(seq)
                         for seq in x])

    def encode_y(self, y: List[str]):
        if self.definition.labels is None:
            raise Exception("unknown labels, call parse_examples_data or "
                            "load_model first")

        y = [ex.split("|") for ex in y]
        mlb = MultiLabelBinarizer(classes=self.definition.labels)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = mlb.fit_transform(y).astype(bool)
        return y

    def decode_y(self, y):
        if self.definition.labels is None:
            raise Exception("unknown labels, call parse_examples_data or "
                            "load_model first")
        labels = np.array(self.definition.labels)

        decoded_y = [labels[ex] for ex in y]
        decoded_y = ["|".join(ex) for ex in decoded_y]
        return decoded_y

    def parse_examples_data(self,
                            file_name: str) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(file_name, sep="\t")
        x: List[str] = df["x"].tolist()
        y: List[str] = df["y"].replace(np.nan, "", regex=True).tolist()

        DNAHelper.check_sequence(x)
        return (x, y)
