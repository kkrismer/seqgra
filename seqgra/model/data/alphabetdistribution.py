"""
MIT - CSAIL - Gifford Lab - seqgra

AlphabetDistribution class definition, markup language agnostic

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from seqgra.model.data import Condition


class AlphabetDistribution:
    def __init__(self, letters: List[Tuple[str, float]],
                 condition: Condition = None, set_name: str = None) -> None:
        self.letters: List[Tuple[str, float]] = letters
        self.__l = [letter[0] for letter in self.letters]
        self.__p = [letter[1] for letter in self.letters]
        self.__p = [prop / sum(self.__p) for prop in self.__p]
        self.condition: Condition = condition
        self.set_name: str = set_name
        self.condition_independent: bool = condition is None
        self.set_independent: bool = set_name is None

    def __str__(self):
        config = ["Alphabet distribution:\n"]
        if self.condition_independent:
            config += ["\tcondition: all\n"]
        else:
            config += ["\tcondition: ",
                       self.condition.condition_id, " [cid]\n"]
        if self.set_independent:
            config += ["\tset: all\n"]
        else:
            config += ["\tset: ", self.set_name, " [setname]\n"]
        config += ["\tletters:\n"]
        letters_string: List[str] = [("\t\t" + letter[0] + ": " +
                                      str(round(letter[1], 3)) + "\n") for letter in self.letters]
        config += letters_string
        return "".join(config)

    def generate_letters(self, n: int) -> str:
        return "".join(np.random.choice(self.__l, n, p=self.__p))
