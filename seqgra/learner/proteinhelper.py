"""MIT - CSAIL - Gifford Lab - seqgra

Helper class for functions operating on amino acid sequences

@author: Konstantin Krismer
"""
import re
import logging
from typing import List

import numpy as np


class ProteinHelper:
    @staticmethod
    def convert_dense_to_one_hot_encoding(seq: str):
        aa_to_num = dict({"A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
                          "E": 5, "Q": 6, "G": 7, "H": 8, "I": 9,
                          "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
                          "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19})
        seq = list(seq)
        seq = np.array([aa_to_num[aa] for aa in seq], dtype=int)

        one_hot_encoded_seq = np.zeros((len(seq), len(aa_to_num)))
        one_hot_encoded_seq[np.arange(len(seq)), seq] = 1
        return one_hot_encoded_seq

    @staticmethod
    def convert_one_hot_to_dense_encoding(seq: str):
        densely_encoded_seq = ["X"] * seq.shape[0]
        for i in range(seq.shape[0]):
            if all(seq[i, :] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "A"
            elif all(seq[i, :] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "R"
            elif all(seq[i, :] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "N"
            elif all(seq[i, :] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "D"
            elif all(seq[i, :] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "C"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "E"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "Q"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "G"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "H"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "I"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   1, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "L"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 1, 0, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "K"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "M"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "F"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "P"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0]):
                densely_encoded_seq[i] = "S"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 1, 0, 0, 0]):
                densely_encoded_seq[i] = "T"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 1, 0, 0]):
                densely_encoded_seq[i] = "W"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 1, 0]):
                densely_encoded_seq[i] = "Y"
            elif all(seq[i, :] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1]):
                densely_encoded_seq[i] = "V"
        return "".join(densely_encoded_seq)

    @staticmethod
    def check_sequence(seqs: List[str]) -> bool:
        is_valid: bool = True
        for seq in seqs:
            if not re.match("^[ARNDCEQGHILKMFPSTWYV]*$", seq):
                logging.warn("example with invalid amino acid sequence "
                             "(only uppercase single letter codes of 20 "
                             "canonical amino acids allowed): " + seq)
                is_valid = False
        return is_valid
