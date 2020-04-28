"""
MIT - CSAIL - Gifford Lab - seqgra

AlphabetDistribution class definition, markup language agnostic

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from seqgra import ProbabilisticToken
from seqgra.model.data import Condition


class AlphabetDistribution:
    def __init__(self, letters: List[ProbabilisticToken],
                 condition: Optional[Condition] = None,
                 set_name: Optional[str] = None) -> None:
        self.letters: List[ProbabilisticToken] = letters
        self._letters = [letter.token for letter in self.letters]
        self._probabilities = [letter.probability for letter in self.letters]
        self._probabilities = [p / sum(self._probabilities)
                               for p in self._probabilities]
        self.condition: Optional[Condition] = condition
        self.set_name: Optional[str] = set_name
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
        letters_string: List[str] = [("\t\t" + letter.token + ": " +
                                      str(round(letter.probability, 3)) + "\n")
                                     for letter in self.letters]
        config += letters_string
        return "".join(config)

    def generate_letters(self, n: int) -> str:
        return "".join(np.random.choice(self._letters, n,
                                        p=self._probabilities))
