"""TODO

@author: Konstantin Krismer
"""

from typing import List

from seqgra.model.alphabetdistribution import AlphabetDistribution

class Background:
    def __init__(self, min_length: int, max_length: int,
                 alphabet_distributions: List[AlphabetDistribution]) -> None:
        self.min_length: int = int(min_length)
        self.max_length: int = int(max_length)
        self.alphabet_distributions: List[AlphabetDistribution] = alphabet_distributions
