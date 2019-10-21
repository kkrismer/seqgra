"""
MIT - CSAIL - Gifford Lab - seqgra

Background generator

@author: Konstantin Krismer
"""
from __future__ import annotations

import numpy as np

from seqgra.model.data.alphabetdistribution import AlphabetDistribution
from seqgra.model.data.background import Background
from seqgra.model.data.condition import Condition

class BackgroundGenerator:

    @staticmethod
    def generate_background(background: Background, condition: Condition) -> str:
        bg_length: int = BackgroundGenerator.__determine_length(background)
        alphabet_distribution: AlphabetDistribution = BackgroundGenerator.__select_alphabet_distribution(background, condition)
        return alphabet_distribution.generate_letters(bg_length)

    @staticmethod
    def __determine_length(background: Background) -> int:
        if background.min_length == background.max_length:
            return background.min_length
        else:
            return np.random.randint(background.min_length, high=background.max_length + 1)

    @staticmethod
    def __select_alphabet_distribution(background: Background, condition: Condition) -> AlphabetDistribution:
        if condition is None:
            # pick random alphabet distribution
            random_idx: int = np.random.randint(0, high=len(background.alphabet_distributions))
            return background.alphabet_distributions[random_idx]
        else:
            for alphabet_distribution in background.alphabet_distributions:
                if alphabet_distribution.condition_independent or alphabet_distribution.condition.id == condition.id:
                    return alphabet_distribution
            raise Exception("no alphabet distribution found for condition " + condition.id + " [cid]")
