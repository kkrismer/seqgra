"""MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for learners

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List, Set
import re

import tensorflow as tf
import numpy as np
import logging
import pandas as pd

from seqgra.learner.learner import MultiClassClassificationLearner
from seqgra.parser.modelparser import ModelParser

# class TensorFlowEstimatorMultiClassClassificationLearner(MultiClassClassificationLearner):
#     def __init__(self, output_dir: str) -> None:
#         super().__init__(output_dir)

#     @staticmethod
#     def __bytes_feature(value):
#         if isinstance(value, type(tf.constant(0))):
#             value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#         return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

#     @staticmethod
#     def __serialize_example(sequence_feature, label):
#         """
#         Creates a tf.Example message ready to be written to a file.
#         """
#         # Create a dictionary mapping the feature name to the tf.Example-compatible
#         # data type.
#         feature = {
#             "sequence": TensorFlowMultiClassClassificationLearner.__bytes_feature(sequence_feature.tostring()),
#             "label": TensorFlowMultiClassClassificationLearner.__bytes_feature(label.tostring())
#         }
#         example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#         return example_proto.SerializeToString()

#     def __parse_data_file(self, data_file_path: str, set_name: str):
#         tfrecord_file_name: str = self.output_dir + "/" + set_name + ".tfrecord"
#         with open(data_file_path, "r") as reader, tf.io.TFRecordWriter(tfrecord_file_name) as writer:
#             # skip header
#             next(reader)

#             for line in reader:
#                 cells = line.split("\t")
#                 seq = cells[0].strip()
#                 is_condition1 = cells[1].strip() == "c1"
#                 is_condition2 = not is_condition1
#                 label = np.array([is_condition1, is_condition2], dtype = bool)
#                 if re.match("^[ACGT]*$", seq):
#                     seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
#                     seq = np.array(list(seq), dtype = int)
#                     example = TensorFlowMultiClassClassificationLearner.__serialize_example(seq, label)
#                     writer.write(example)
#                 else:
#                     logging.warn("skipped invalid example:" + seq + " (label: " + str(label) + ")")
#         return tfrecord_file_name

#     def parse_data(self, training_set_file: str, validation_set_file: str) -> None:
#         tfrecord_file_name = self.__parse_data_file(training_set_file, "training")
#         tfrecord_file_name = self.__parse_data_file(validation_set_file, "validation")


class DNAMultiClassClassificationLearner(MultiClassClassificationLearner):
    def __init__(self, parser: ModelParser, output_dir: str) -> None:
        super().__init__(parser, output_dir)
        self.labels: List[str] = None
        self.x_train: List[str] = None
        self.y_train: List[str] = None
        self.x_val: List[str] = None
        self.y_val: List[str] = None

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

    def _encode_x(self, x: List[str]):
        return np.stack([self.__convert_dense_to_one_hot_encoding(seq) for seq in x])

    def _decode_x(self, x):
        return np.stack([self.__convert_one_hot_to_dense_encoding(seq) for seq in x])
    
    def __check_sequence(self, seqs: List[str]) -> None:
        for seq in seqs:
            if not re.match("^[ACGT]*$", seq):
                logging.warn("example with invalid sequence:" + seq)
        
    def _encode_y(self, y: List[str]):
        if self.labels is None:
            raise Exception("unknown labels, call parse_data or load_model first")
        labels = np.array(self.labels)
        return np.vstack([np.array([label] * len(labels)) == labels for label in y])
        
    def _decode_y(self, y):
        pass

    def __discover_labels(self, training_set_y: List[str], validation_set_y: List[str]) -> List[str]:
        label_set: Set[str] = set(training_set_y)
        label_set.update(validation_set_y)
        labels: List[str] = list(label_set)
        labels.sort()
        return labels

    def parse_data(self, training_set_file: str, validation_set_file: str) -> None:
        training_set_df = pd.read_csv(training_set_file, sep="\t")
        x_train_plain: List[str] = training_set_df["x"].tolist()
        y_train_plain: List[str] = training_set_df["y"].tolist()

        validation_set_df = pd.read_csv(validation_set_file, sep="\t")
        x_val_plain: List[str] = validation_set_df["x"].tolist()
        y_val_plain: List[str] = validation_set_df["y"].tolist()

        self.labels = self.__discover_labels(y_train_plain, y_val_plain)

        self.__check_sequence(x_train_plain)
        self.x_train = self._encode_x(x_train_plain)
        self.y_train = self._encode_y(y_train_plain)

        self.__check_sequence(x_val_plain)
        self.x_val = self._encode_x(x_val_plain)
        self.y_val = self._encode_y(y_val_plain)

    def parse_test_data(self, test_set_file: str):
        if self.labels is None:
            raise Exception("unknown labels, call parse_data or load_model first")
        test_set_df = pd.read_csv(test_set_file, sep="\t")
        x_test_plain: List[str] = test_set_df["x"].tolist()
        y_test_plain: List[str] = test_set_df["y"].tolist()
        
        self.__check_sequence(x_test_plain)
        # TODO instead of using properties, return x and y, maybe tuple?
        self.x_test = self._encode_x(x_test_plain)
        self.y_test = self._encode_y(y_test_plain)

