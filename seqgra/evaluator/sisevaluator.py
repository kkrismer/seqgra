"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import random

import numpy as np
import pandas as pd

from seqgra.learner.learner import Learner
from seqgra.evaluator.evaluator import Evaluator
from seqgra.sis import sis_collection, make_empty_boolean_mask_broadcast_over_axis, produce_masked_inputs


class SISEvaluator(Evaluator):
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        super().__init__(learner, data_dir, output_dir)

    def evaluate(self, for_set="training") -> None:
        pass

    def select_random_n_examples(self, for_label, for_set, n):
        input_df, annotation_df = self.__select_examples(for_label, for_set)
        
        if n > len(input_df.index):
            logging.warn("n is larger than number of examples in set")
            n = len(input_df.index)
        
        idx: List[int] = list(range(len(input_df.index)))
        random.shuffle(idx)
        idx = idx[:n]

        input_df = input_df.iloc[idx]
        annotation_df = annotation_df.iloc[idx]

        return (input_df["x"].tolist(), annotation_df["annotation"].tolist())

    def select_first_n_examples(self, for_label, for_set, n):
        input_df, annotation_df = self.__select_examples(for_label, for_set)
        
        if n > len(input_df.index):
            logging.warn("n is larger than number of examples in set")
            n = len(input_df.index)
        
        input_df = input_df.iloc[range(n)]
        annotation_df = annotation_df.iloc[range(n)]

        return (input_df["x"].tolist(), annotation_df["annotation"].tolist())

    def find_sis(self, for_label, label_index, for_set, n=10,
                 select_randomly=False,
                 threshold=0.9):
        if select_randomly:
            examples = self.select_random_n_examples(for_label, for_set, n)
        else:
            examples = self.select_first_n_examples(for_label, for_set, n)

        decoded_examples = examples[0]
        annotations = examples[1]
        encoded_examples = self.learner.encode_x(decoded_examples)

        def sis_predict(x):
            return np.array(self.learner.predict(x, encode=False))[:, label_index]

        input_shape = encoded_examples[0].shape
        fully_masked_input = np.ones(input_shape) * 0.25
        initial_mask = make_empty_boolean_mask_broadcast_over_axis(
            input_shape, 1)

        for i in range(len(encoded_examples)):
            encoded_example = encoded_examples[i]
            print(decoded_examples[i])
            print(annotations[i])
            collection = sis_collection(sis_predict, threshold, encoded_example,
                                        fully_masked_input,
                                        initial_mask=initial_mask)

            if len(collection) > 0:
                sis_masked_inputs = produce_masked_inputs(encoded_example,
                                                          fully_masked_input,
                                                          [sr.mask for sr in collection])
                print(self.learner.decode_x(sis_masked_inputs))
            else:
                print("(no SIS)")

    def __get_valid_file(self, data_file: str) -> str:
        data_file = data_file.replace("\\", "/").replace("//", "/").strip()
        if os.path.isfile(data_file):
            return data_file
        else:
            raise Exception("file does not exist: " + data_file)

    def __select_examples(self, for_label, for_set):
        if for_set == "training":
            input_file = self.__get_valid_file(self.data_dir + "/training.txt")
            annotation_file = self.__get_valid_file(
                self.data_dir + "/training-annotation.txt")
        elif for_set == "validation":
            input_file = self.__get_valid_file(
                self.data_dir + "/validation.txt")
            annotation_file = self.__get_valid_file(
                self.data_dir + "/validation-annotation.txt")
        elif for_set == "test":
            input_file = self.__get_valid_file(self.data_dir + "/test.txt")
            annotation_file = self.__get_valid_file(
                self.data_dir + "/test-annotation.txt")
        else:
            raise Exception("unsupported set: " + for_set)

        input_df = pd.read_csv(input_file, sep="\t")
        input_df = input_df[input_df.y == for_label]

        annotation_df = pd.read_csv(annotation_file, sep="\t")
        annotation_df = annotation_df[annotation_df.y == for_label]

        return (input_df, annotation_df)