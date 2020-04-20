"""
MIT - CSAIL - Gifford Lab - seqgra

Background class definition, markup language agnostic

@author: Konstantin Krismer
"""
from typing import List

from seqgra.model.data.alphabetdistribution import AlphabetDistribution

class Background:
    def __init__(self, min_length: int, max_length: int,
                 alphabet_distributions: List[AlphabetDistribution]) -> None:
        self.min_length: int = int(min_length)
        self.max_length: int = int(max_length)
        self.alphabet_distributions: List[AlphabetDistribution] = \
            alphabet_distributions

    def __str__(self):
        str_rep = ["Background:\n",
        "\tMinimum length: ", str(self.min_length), "\n",
        "\tMaximum length: ", str(self.max_length), "\n",
        "\tAlphabet distributions:\n"]
        alphabets_string: List[str] = [str(alphabet_distribution)
                                       for alphabet_distribution
                                       in self.alphabet_distributions]
        alphabets_str_rep = ''.join(alphabets_string)
        str_rep += ["\t\t" + s + "\n" for s in alphabets_str_rep.splitlines()]
        return "".join(str_rep)
