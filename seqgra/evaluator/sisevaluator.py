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

    def evaluate_model(self, set_name: str = "training") -> None:
        pass
    
    def save_results(self, results, name: str) -> None:
        tmp = results.copy()
        for i in range(len(tmp)):
            result = tmp[i]
            tmp[i] = (result[0], result[1], ";".join(result[2]))

        df = pd.DataFrame(tmp, columns=["input", "annotation", "sis"]) 

        df.to_csv(self.output_dir + name + ".txt", sep = "\t", index=False)
    
    def load_results(self, name: str):
        df = pd.read_csv(self.output_dir + name + ".txt", sep = "\t")

        results = [tuple(x) for x in df.values]
        for i in range(len(results)):
            result = results[i]
            results[i] = (result[0], result[1], result[2].split(";"))
        return results

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

        return [(decoded_examples[i],
                 annotations[i],
                 self.__produce_masked_inputs(encoded_examples[i],
                                              sis_predict,
                                              threshold,
                                              fully_masked_input,
                                              initial_mask))
                for i in range(len(encoded_examples))]

    def calculate_precision(self, x, grammar_letter="G", background_letter="_", masked_letter="N") -> List[float]:
        precision_values: List[float] = [self.__calculate_precision(sis,
                                                                    grammar_letter=grammar_letter,
                                                                    background_letter=background_letter,
                                                                    masked_letter=masked_letter) for sis in x]
        return precision_values

    def calculate_recall(self, x, grammar_letter="G", background_letter="_", masked_letter="N") -> List[float]:
        recall_values: List[float] = [self.__calculate_recall(sis,
                                                              grammar_letter=grammar_letter,
                                                              background_letter=background_letter,
                                                              masked_letter=masked_letter) for sis in x]
        return recall_values

    def __calculate_precision(self, x, grammar_letter="G", background_letter="_", masked_letter="N") -> float:
        annotation: str = x[1]
        sis: str = self.__collapse_sis(x[2], masked_letter=masked_letter)

        if sis == "":
            return 1.0
        else:
            num_selected: int = 0
            num_selected_relevant: int = 0
            for i, c in enumerate(sis):
                if c != masked_letter:
                    num_selected += 1
                    if annotation[i] == grammar_letter:
                        num_selected_relevant += 1
            return num_selected_relevant / num_selected

    def __calculate_recall(self, x, grammar_letter="G", background_letter="_", masked_letter="N") -> float:
        annotation: str = x[1]
        sis: str = self.__collapse_sis(x[2], masked_letter=masked_letter)

        if sis == "":
            return 0.0
        else:
            num_relevent: int = 0
            num_relevant_selected: int = 0
            for i, c in enumerate(annotation):
                if c == grammar_letter:
                    num_relevent += 1
                    if sis[i] != masked_letter:
                        num_relevant_selected += 1

            return num_relevant_selected / num_relevent

    def __collapse_sis(self, sis: List[str], masked_letter: str = "N") -> str:
        if len(sis) == 0:
            return ""
        elif len(sis) == 1:
            return sis[0]
        else:
            collapsed_sis: str = sis[0]
            sis.pop(0)
            for i, c in enumerate(collapsed_sis):
                if c == masked_letter:
                    for s in sis:
                        if s[i] != masked_letter:
                            collapsed_sis = collapsed_sis[:i] + \
                                s[i] + collapsed_sis[(i + 1):]
            return collapsed_sis

    def __produce_masked_inputs(self, x, sis_predict, threshold, fully_masked_input, initial_mask) -> List[str]:
        collection = sis_collection(sis_predict, threshold, x,
                                    fully_masked_input,
                                    initial_mask=initial_mask)

        if len(collection) > 0:
            sis_masked_inputs = produce_masked_inputs(x,
                                                      fully_masked_input,
                                                      [sr.mask for sr in collection])
            return self.learner.decode_x(sis_masked_inputs).tolist()
        else:
            return list()

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
