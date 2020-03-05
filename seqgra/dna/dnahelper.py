"""MIT - CSAIL - Gifford Lab - seqgra

Helper class for functions operating on DNA

@author: Konstantin Krismer
"""
import re
import logging
from typing import List

import numpy as np


class DNAHelper:
    @staticmethod
    def convert_dense_to_one_hot_encoding(seq: str):
        seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
        seq = np.array(list(seq), dtype = int)

        one_hot_encoded_seq = np.zeros((len(seq), 4))
        one_hot_encoded_seq[np.arange(len(seq)), seq] = 1
        return one_hot_encoded_seq

    @staticmethod
    def convert_one_hot_to_dense_encoding(seq: str):
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

    @staticmethod
    def check_sequence(seqs: List[str]) -> bool:
        is_valid: bool = True
        for seq in seqs:
            if not re.match("^[ACGT]*$", seq):
                logging.warn("example with invalid sequence:" + seq)
                is_valid = False
        return is_valid
