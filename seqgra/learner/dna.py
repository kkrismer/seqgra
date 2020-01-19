"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for learners

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List, Set
import re
import itertools
import warnings

import tensorflow as tf
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from seqgra.learner.learner import MultiClassClassificationLearner
from seqgra.learner.learner import MultiLabelClassificationLearner
from seqgra.parser.modelparser import ModelParser


class DNAMultiClassClassificationLearner(MultiClassClassificationLearner):
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

    def __convert_dense_to_one_hot_encoding(self, seq: str):
        seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
        seq = np.array(list(seq), dtype = int)

        one_hot_encoded_seq = np.zeros((len(seq), 4))
        one_hot_encoded_seq[np.arange(len(seq)), seq] = 1
        return one_hot_encoded_seq

    def __convert_one_hot_to_dense_encoding(self, seq: str):
        densely_encoded_seq = ["N"] * seq.shape[0]
        for i in range(seq.shape[0]):
            if all(seq[i, :] == [1, 0, 0, 0]):
                densely_encoded_seq[i] = "A"
            elif all(seq[i, :] == [0, 1, 0, 0]):
                densely_encoded_seq[i] = "C"
            elif all(seq[i, :] == [0, 0, 1, 0]):
                densely_encoded_seq[i] = "G"
            elif all(seq[i, :] == [0, 0, 0, 1]):
                densely_encoded_seq[i] = "T"
        return "".join(densely_encoded_seq)

    def encode_x(self, x: List[str]):
        return np.stack([self.__convert_dense_to_one_hot_encoding(seq)
                         for seq in x])

    def decode_x(self, x):
        return np.stack([self.__convert_one_hot_to_dense_encoding(seq)
                         for seq in x])
    
    def __check_sequence(self, seqs: List[str]) -> None:
        for seq in seqs:
            if not re.match("^[ACGT]*$", seq):
                logging.warn("example with invalid sequence:" + seq)
        
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

        self.__check_sequence(x)
        return (x, y)


class DNAMultiLabelClassificationLearner(MultiLabelClassificationLearner):
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)

    def __convert_dense_to_one_hot_encoding(self, seq: str):
        seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
        seq = np.array(list(seq), dtype = int)

        one_hot_encoded_seq = np.zeros((len(seq), 4))
        one_hot_encoded_seq[np.arange(len(seq)), seq] = 1
        return one_hot_encoded_seq

    def __convert_one_hot_to_dense_encoding(self, seq: str):
        densely_encoded_seq = ["N"] * seq.shape[0]
        for i in range(seq.shape[0]):
            if all(seq[i, :] == [1, 0, 0, 0]):
                densely_encoded_seq[i] = "A"
            elif all(seq[i, :] == [0, 1, 0, 0]):
                densely_encoded_seq[i] = "C"
            elif all(seq[i, :] == [0, 0, 1, 0]):
                densely_encoded_seq[i] = "G"
            elif all(seq[i, :] == [0, 0, 0, 1]):
                densely_encoded_seq[i] = "T"
        return "".join(densely_encoded_seq)

    def encode_x(self, x: List[str]):
        return np.stack([self.__convert_dense_to_one_hot_encoding(seq) 
                         for seq in x])

    def decode_x(self, x):
        return np.stack([self.__convert_one_hot_to_dense_encoding(seq)
                         for seq in x])
    
    def __check_sequence(self, seqs: List[str]) -> None:
        for seq in seqs:
            if not re.match("^[ACGT]*$", seq):
                logging.warn("example with invalid sequence:" + seq)
        
    def encode_y(self, y: List[str]):
        if self.labels is None:
            raise Exception("unknown labels, call parse_data or "
                            "load_model first")

        y = [ex.split("|") for ex in y]
        mlb = MultiLabelBinarizer(classes = self.labels)
        
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

        self.__check_sequence(x)
        return (x, y)
