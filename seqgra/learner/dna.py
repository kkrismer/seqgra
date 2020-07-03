"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for learners

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from seqgra import ExampleSet
from seqgra.learner import MultiClassClassificationLearner
from seqgra.learner import MultiLabelClassificationLearner
from seqgra.learner import DNAHelper
from seqgra.model import ModelDefinition


class DNAMultiClassClassificationLearner(MultiClassClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True,
                 gpu_id: int = 0) -> None:
        super().__init__(model_definition, data_dir, output_dir,
                         validate_data, gpu_id)

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

        if isinstance(y, list):
            y = np.asarray(y)
        elif not isinstance(y, np.ndarray):
            raise Exception("y is neither list nor np.ndarry")

        if y.dtype == np.float32 or y.dtype == np.float64 or \
                y.dtype == np.float_:
            # binarize y
            true_idx = np.argmax(y, axis=1)
            y = np.zeros(y.shape)
            y[np.arange(len(y)), true_idx] = 1
            y = y.astype(bool)
        elif y.dtype == np.int8 or y.dtype == np.int16 or \
                y.dtype == np.int32 or y.dtype == np.int64 or \
                y.dtype == np.uint8 or y.dtype == np.uint16 or \
                y.dtype == np.uint32 or y.dtype == np.uint64 or \
                y.dtype == np.intp or y.dtype == np.uintp or \
                y.dtype == np.int_:
            y = y.astype(bool)
        elif y.dtype != np.bool_:
            raise Exception("y has invalid data type; valid data types "
                            "include bool, int, float")

        decoded_y = np.vstack([labels[ex] for ex in y])
        decoded_y = list(itertools.chain(*decoded_y))
        return decoded_y

    def parse_examples_data(self, file_name: str) -> ExampleSet:
        df = pd.read_csv(file_name, sep="\t", dtype={"x": "string",
                                                     "y": "string"})
        df = df.fillna("")
        x: List[str] = df["x"].tolist()
        y: List[str] = df["y"].tolist()

        if self.validate_data:
            DNAHelper.check_sequence(x)
            self.check_labels(y)
        return ExampleSet(x, y)


class DNAMultiLabelClassificationLearner(MultiLabelClassificationLearner):
    def __init__(self, model_definition: ModelDefinition, data_dir: str,
                 output_dir: str, validate_data: bool = True,
                 gpu_id: int = 0) -> None:
        super().__init__(model_definition, data_dir, output_dir,
                         validate_data, gpu_id)

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

        if isinstance(y, list):
            y = np.asarray(y)
        elif not isinstance(y, np.ndarray):
            raise Exception("y is neither list nor np.ndarry")

        if y.dtype == np.float32 or y.dtype == np.float64 or \
                y.dtype == np.float_:
            # binarize y
            y = np.greater(y, 0.5).astype(bool)
        elif y.dtype == np.int8 or y.dtype == np.int16 or \
                y.dtype == np.int32 or y.dtype == np.int64 or \
                y.dtype == np.uint8 or y.dtype == np.uint16 or \
                y.dtype == np.uint32 or y.dtype == np.uint64 or \
                y.dtype == np.intp or y.dtype == np.uintp or \
                y.dtype == np.int_:
            y = y.astype(bool)
        elif y.dtype != np.bool_:
            raise Exception("y has invalid data type; valid data types "
                            "include bool, int, float")

        decoded_y = [labels[ex] for ex in y]
        decoded_y = ["|".join(ex) for ex in decoded_y]
        return decoded_y

    def parse_examples_data(self, file_name: str) -> ExampleSet:
        df = pd.read_csv(file_name, sep="\t", dtype={"x": "string",
                                                     "y": "string"})
        df = df.fillna("")
        x: List[str] = df["x"].tolist()
        y: List[str] = df["y"].replace(np.nan, "", regex=True).tolist()

        if self.validate_data:
            DNAHelper.check_sequence(x)
            self.check_labels(y)
        return ExampleSet(x, y)
