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
        return np.stack([self.__convert_dense_to_one_hot_encoding(seq) for seq in x])

    def decode_x(self, x):
        return np.stack([self.__convert_one_hot_to_dense_encoding(seq) for seq in x])
    
    def __check_sequence(self, seqs: List[str]) -> None:
        for seq in seqs:
            if not re.match("^[ACGT]*$", seq):
                logging.warn("example with invalid sequence:" + seq)
        
    def encode_y(self, y: List[str]):
        if self.labels is None:
            raise Exception("unknown labels, call parse_data or load_model first")
        labels = np.array(self.labels)
        return np.vstack([np.array([label] * len(labels)) == labels for label in y])
        
    def decode_y(self, y):
        pass

    def parse_data(self, file_name: str) -> None:
        df = pd.read_csv(file_name, sep="\t")
        x: List[str] = df["x"].tolist()
        y: List[str] = df["y"].tolist()

        self.__check_sequence(x)
        return (x, y)
