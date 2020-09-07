"""
MIT - CSAIL - Gifford Lab - seqgra

@author: Konstantin Krismer
"""
from dataclasses import dataclass
from typing import List, NamedTuple


@dataclass
class Example:
    x: str
    y: str


@dataclass
class AnnotatedExample:
    x: str
    y: str
    annotation: str


class ExampleSet(NamedTuple):
    x: List[str]
    y: List[str]


class AnnotationSet(NamedTuple):
    annotations: List[str]
    y: List[str]


class AnnotatedExampleSet(NamedTuple):
    x: List[str]
    y: List[str]
    annotations: List[str]


class ProbabilisticToken(NamedTuple):
    token: str
    probability: float


class Metrics(NamedTuple):
    loss: float
    accuracy: float


class ModelSize(NamedTuple):
    num_trainable_params: int
    num_non_trainable_params: int
