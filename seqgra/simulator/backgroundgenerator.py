"""
MIT - CSAIL - Gifford Lab - seqgra

Background generator

@author: Konstantin Krismer
"""
from __future__ import annotations

import numpy as np

from seqgra.model.data import AlphabetDistribution
from seqgra.model.data import Background
from seqgra.model.data import Condition

class BackgroundGenerator:

    @staticmethod
    def generate_background(background: Background, condition: Condition, 
                            set_name: str) -> str:
        bg_length: int = BackgroundGenerator.__determine_length(background)
        alphabet_distribution: AlphabetDistribution = \
            BackgroundGenerator.__select_alphabet_distribution(
                background, 
                condition, 
                set_name)
        return alphabet_distribution.generate_letters(bg_length)

    @staticmethod
    def __determine_length(background: Background) -> int:
        if background.min_length == background.max_length:
            return background.min_length
        else:
            return np.random.randint(background.min_length, 
                                     high=background.max_length + 1)

    @staticmethod
    def __select_alphabet_distribution(background: Background, 
                                       condition: Condition,
                                       set_name: str) -> AlphabetDistribution:
        if condition is None:
            # return set specific alphabet distribution, if exists
            for alphabet_distribution in background.alphabet_distributions:
                if alphabet_distribution.condition_independent and \
                   alphabet_distribution.set_name == set_name:
                    return alphabet_distribution
                    
            # or return global alphabet distribution, if exists
            for alphabet_distribution in background.alphabet_distributions:
                if alphabet_distribution.condition_independent and \
                   alphabet_distribution.set_independent:
                    return alphabet_distribution

            # otherwise pick random alphabet distribution
            random_idx: int = np.random.randint(
                0,
                high=len(background.alphabet_distributions))
            return background.alphabet_distributions[random_idx]
        else:
            global_alphabet_distribution: AlphabetDistribution = None
            set_alphabet_distribution: AlphabetDistribution = None
            condition_alphabet_distribution: AlphabetDistribution = None
            for alphabet_distribution in background.alphabet_distributions:
                if alphabet_distribution.condition_independent and \
                   alphabet_distribution.set_independent:
                    global_alphabet_distribution = alphabet_distribution
                elif alphabet_distribution.condition_independent and \
                     alphabet_distribution.set_name == set_name:
                    set_alphabet_distribution = alphabet_distribution
                elif alphabet_distribution.set_independent and \
                     alphabet_distribution.condition.id == condition.id:
                    condition_alphabet_distribution = alphabet_distribution
                elif alphabet_distribution.set_name == set_name and \
                     alphabet_distribution.condition.id == condition.id:
                    return alphabet_distribution
            
            if set_alphabet_distribution is not None:
                return set_alphabet_distribution
            elif condition_alphabet_distribution is not None:
                return condition_alphabet_distribution
            elif global_alphabet_distribution is not None:
                return global_alphabet_distribution
            
            raise Exception("no alphabet distribution found for set " +
                            set_name + " and condition " + condition.id +
                            " [cid]")
