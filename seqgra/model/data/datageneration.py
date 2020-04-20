"""
MIT - CSAIL - Gifford Lab - seqgra

DataGeneration and Set class definitions, markup language agnostic

@author: Konstantin Krismer
"""
from typing import List

from seqgra.model.data.condition import Condition
from seqgra.model.data.operation import PostprocessingOperation


class Example:
    def __init__(self, samples: int, conditions: List[Condition]) -> None:
        self.samples: int = samples
        self.conditions: List[Condition] = conditions

    def __str__(self):
        str_rep: List[str] = ["Example:\n",
                              "\tNumber of samples drawn: ", str(
                                  self.samples), "\n",
                              "\tInstance of the following conditions:\n"]
        str_rep += ["\t\t" + "condition " + condition.id + " [cid]\n"
                    for condition in self.conditions]
        return "".join(str_rep)


class ExampleSet:
    def __init__(self, name: str, examples: List[Example]) -> None:
        self.name: str = name
        self.examples: List[Example] = examples

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
    def __init__(self, seed: int, sets: List[ExampleSet],
                 postprocessing_operations: List[PostprocessingOperation] = None) -> None:
        self.seed: int = int(seed)
        self.sets: List[ExampleSet] = sets
        self.postprocessing_operations: List[PostprocessingOperation] = postprocessing_operations

    def __str__(self):
        str_rep = ["Data generation:\n",
                   "\tSets:\n"]
        sets_string: List[str] = [str(example_set)
                                  for example_set in self.sets]
        sets_str_rep = ''.join(sets_string)
        str_rep += ["\t\t" + s + "\n" for s in sets_str_rep.splitlines()]
        str_rep += ["\tPost-processing operations:\n"]
        if self.postprocessing_operations is None or \
                len(self.postprocessing_operations) == 0:
            str_rep += ["\t\tnone\n"]
        else:
            operations_string: List[str] = [str(operation)
                                            for operation in self.postprocessing_operations]
            operations_str_rep = ''.join(operations_string)
            str_rep += ["\t\t" + s +
                        "\n" for s in operations_str_rep.splitlines()]
        return "".join(str_rep)
