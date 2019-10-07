"""MIT - CSAIL - Gifford Lab - seqgra

AlphabetDistribution class definition, markup language agnostic

@author: Konstantin Krismer
"""

from __future__ import annotations

from typing import List, Tuple
import random

import numpy as np

from seqgra.model.condition import Condition

class AlphabetDistribution:
    def __init__(self, letters: List[Tuple[str, float]], condition: Condition = None) -> None:
        self.letters: List[Tuple[str, float]] = letters
        self.l = [letter[0] for letter in self.letters]
        self.p = [letter[1] for letter in self.letters]
        self.p = [prop / sum(self.p)  for prop in self.p]
        self.condition: Condition = condition
        self.condition_independent: bool = condition is None

    def __str__(self):
        config = ["Alphabet distribution:\n"]
        if self.condition_independent:
            config += ["\tcondition: all\n"]
        else:
            config += ["\tcondition: ", self.condition.id, "[cid]\n"]
        config += ["\tletters:\n"]
        letters_string: List[str] = [("\t\t" + letter[0] + ": " + str(round(letter[1], 3)) + "\n") for letter in self.letters]
        config += letters_string
        return "".join(config)
    
    def generate_letters(self, n: int) -> str:
        return "".join(np.random.choice(self.l, n, p=self.p))