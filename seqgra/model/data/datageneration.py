"""
MIT - CSAIL - Gifford Lab - seqgra

DataGeneration and Set class definitions, markup language agnostic

@author: Konstantin Krismer
"""
from __future__ import annotations

from typing import List, Optional

from seqgra.model.data import Condition
from seqgra.model.data import PostprocessingOperation


class DataGenerationExample:
    def __init__(self, samples: int,
                 conditions: Optional[List[Condition]] = None) -> None:
        self.samples: int = samples
        self.conditions: Optional[List[Condition]] = conditions

    def __str__(self):
        str_rep: List[str] = ["Example:\n",
                              "\tNumber of samples drawn: ", str(
                                  self.samples), "\n",
                              "\tInstance of the following conditions:\n"]
        if self.conditions is not None and len(self.conditions) > 0:
            str_rep += ["\t\t" + "condition " + condition.condition_id + " [cid]\n"
                        for condition in self.conditions]
        else:
            str_rep += ["\t\tnone"]
        return "".join(str_rep)


class DataGenerationSet:
    def __init__(self, name: str, examples: List[DataGenerationExample]) -> None:
        self.name: str = name
        self.examples: List[DataGenerationExample] = examples

    def __str__(self):
        str_rep = ["Set:\n",
                   "\tName: ", self.name, "\n",
                   "\tExamples:\n"]
        examples_string: List[str] = [str(example)
                                      for example in self.examples]
        examples_str_rep = "".join(examples_string)
        str_rep += ["\t\t" + s + "\n" for s in examples_str_rep.splitlines()]
        return "".join(str_rep)


class DataGeneration:
    def __init__(self, sets: List[DataGenerationSet],
                 postprocessing_operations: Optional[List[PostprocessingOperation]] = None) -> None:
        self.sets: List[DataGenerationSet] = sets
        self.postprocessing_operations: Optional[List[PostprocessingOperation]
                                                 ] = postprocessing_operations

    def __str__(self):
        str_rep = ["Data generation:\n"]
        sets_string: List[str] = [str(example_set)
                                  for example_set in self.sets]
        sets_str_rep = ''.join(sets_string)
        str_rep += ["\t\t" + s + "\n" for s in sets_str_rep.splitlines()]
        str_rep += ["\tPost-processing operations:\n"]
        if not self.postprocessing_operations:
            str_rep += ["\t\tnone\n"]
        else:
            operations_string: List[str] = [str(operation)
                                            for operation in self.postprocessing_operations]
            operations_str_rep = ''.join(operations_string)
            str_rep += ["\t\t" + s +
                        "\n" for s in operations_str_rep.splitlines()]
        return "".join(str_rep)
