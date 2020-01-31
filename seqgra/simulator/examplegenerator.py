"""
MIT - CSAIL - Gifford Lab - seqgra

Example generator

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List, Tuple, Dict
import random

import numpy as np

from seqgra.model.data.rule import Rule
from seqgra.model.data.condition import Condition
from seqgra.model.data.background import Background
from seqgra.model.data.example import Example
from seqgra.model.data.spacingconstraint import SpacingConstraint
from seqgra.simulator.backgroundgenerator import BackgroundGenerator

class ExampleGenerator:    
    @staticmethod
    def generate_example(conditions: List[Condition], set_name: str,
                         background: Background,
                         background_character: str = "_") -> Example:
        if conditions is None:
            background: str = \
                BackgroundGenerator.generate_background(background, None, 
                                                        set_name)
            annotation: str = "".join([background_character] * len(background))
            example: Example = Example(background, annotation)
        else:
            # randomly shuffle the order of the conditions, 
            # which determines in what order the condition rules are applied
            random.shuffle(conditions)
            # pick the background distribution of the first 
            # condition (after random shuffle)
            background: str = \
                BackgroundGenerator.generate_background(background, 
                                                        conditions[0], 
                                                        set_name)
            annotation: str = "".join([background_character] * len(background))
            example: Example = Example(background, annotation)

            for condition in conditions:
                if condition is not None:
                    for rule in condition.grammar:
                        example = ExampleGenerator.apply_rule(rule, example)
        return example

    @staticmethod
    def apply_rule(rule: Rule, example: Example) -> Example:
        if random.uniform(0, 1) <= rule.probability:
            elements: Dict[str, str] = dict()
            for sequence_element in rule.sequence_elements:
                elements[sequence_element.id] = sequence_element.generate()
            
            if rule.spacing_constraints is not None and \
               len(rule.spacing_constraints) > 0:
                # process all sequence elements with spacing constraints
                for spacing_constraint in rule.spacing_constraints:
                    example = \
                        ExampleGenerator.add_spatially_constrained_elements(
                            example,
                            spacing_constraint,
                            elements[spacing_constraint.sequence_element1.id],
                            elements[spacing_constraint.sequence_element2.id],
                            rule.position)
                    if spacing_constraint.sequence_element1.id in elements:
                        del elements[spacing_constraint.sequence_element1.id]
                    if spacing_constraint.sequence_element2.id in elements:
                        del elements[spacing_constraint.sequence_element2.id]
                
            # process remaining sequence elements (w/o spacing constraints)
            for element in elements.values(): 
                position: int = ExampleGenerator.get_position(
                    rule.position,
                    len(example.sequence),
                    len(element))
                example = ExampleGenerator.add_element(example, element, 
                                                       position)

        return example
    
    @staticmethod
    def get_position(rule_position: str, sequence_length,
                     element_length) -> int:
        if rule_position == "random":
            return np.random.randint(0, 
                                     high=sequence_length - element_length + 1)
        elif rule_position == "start":
            return 0
        elif rule_position == "end":
            return sequence_length - element_length
        elif rule_position == "center":
            return int(sequence_length / 2 - element_length / 2)
        else:
            return int(rule_position) - 1
        
    @staticmethod
    def get_distance(example: Example, spacing_constraint: SpacingConstraint,
                     element1: str, element2: str, rule_position: str) -> int:
        max_length: int = len(example.sequence)
        if rule_position != "random" and \
           rule_position != "start" and \
           rule_position != "end" and \
           rule_position != "center":
            position = int(rule_position)
            max_length -= position
        
        max_distance = max_length - len(element1) - len(element2)
        return np.random.randint(
            spacing_constraint.min_distance,
            high=min(spacing_constraint.max_distance, max_distance) + 1)

    @staticmethod
    def add_spatially_constrained_elements(
        example: Example, 
        spacing_constraint: SpacingConstraint,
        element1: str,
        element2: str,
        rule_position: str) -> Example:
        distance: int = ExampleGenerator.get_distance(example, 
                                                      spacing_constraint, 
                                                      element1, element2, 
                                                      rule_position)

        if spacing_constraint.direction == "random":
            if random.uniform(0, 1) <= 0.5:
                element1, element2 = element2, element1

        position1: int = ExampleGenerator.get_position(
            rule_position,
            len(example.sequence),
            len(element1) + distance + len(element2))
        example = ExampleGenerator.add_element(example, element1, position1)
        
        position2: int = position1 + len(element1) + distance
        example = ExampleGenerator.add_element(example, element2, position2)
        return example

    @staticmethod
    def add_element(example: Example, element: str, position: int,
                    grammar_character: str = "G") -> Example:
        example.sequence = example.sequence[:position] + element + \
            example.sequence[position + len(element):]
        example.annotation = example.annotation[:position] + \
            (grammar_character * len(element)) + \
            example.annotation[position + len(element):]
        return example
