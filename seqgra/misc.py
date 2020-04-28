"""
MIT - CSAIL - Gifford Lab - seqgra

Class with miscellaneous helper functions as static methods

@author: Konstantin Krismer
"""
from dataclasses import dataclass
import os
import shutil
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

class MiscHelper:
    @staticmethod
    def prepare_path(path: str, allow_exists: bool = True) -> str:
        path = path.replace("\\", "/").replace("//", "/").strip()
        if not path.endswith("/"):
            path += "/"

        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception("directory cannot be created "
                                "(file with same name exists)")
            elif len(os.listdir(path)) > 0:
                num_files: int = len([name
                                      for name in os.listdir(path)
                                      if os.path.isfile(path + name)])
                if num_files > 0:
                    if not allow_exists:
                        raise Exception("directory cannot be created "
                                        "(non-empty folder with same "
                                        "name exists)")
                else:
                    shutil.rmtree(path)
                    os.makedirs(path)
        else:
            os.makedirs(path)

        return path
