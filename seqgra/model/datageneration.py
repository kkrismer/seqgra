"""TODO

@author: Konstantin Krismer
"""

from typing import List

from alphabetdistribution import AlphabetDistribution

class DataGeneration:
    def __init__(self, min_length: int, max_length: int,
                 alphabet_distributions: List[AlphabetDistribution]) -> None:
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.alphabet_distributions = alphabet_distributions
