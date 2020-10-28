"""Abstract base class for all comparators
"""
from abc import ABC, abstractmethod
from itertools import compress
import logging
import os
from typing import List, Optional

import seqgra.constants as c
from seqgra import MiscHelper


class Comparator(ABC):
    @abstractmethod
    def __init__(self, comparator_id: str, comparator_name: str,
                 analysis_name: str, output_dir: str,
                 model_labels: Optional[List[str]] = None,
                 silent: bool = False) -> None:
        self.logger = logging.getLogger(__name__)
        if silent:
            self.logger.setLevel(os.environ.get("LOGLEVEL", "WARNING"))
        self.comparator_id: str = comparator_id
        self.comparator_name: str = comparator_name
        self.data_dir: str = MiscHelper.prepare_path(
            output_dir + "/input/", allow_non_empty=True)
        self.model_dir: str = MiscHelper.prepare_path(
            output_dir + "/models/", allow_non_empty=True)
        self.evaluation_dir: str = MiscHelper.prepare_path(
            output_dir + "/evaluation/", allow_non_empty=True)
        self.output_dir: str = MiscHelper.prepare_path(
            output_dir + "/model-comparisons/" + analysis_name,
            allow_exists=True)
        self.model_labels: Optional[List[str]] = model_labels

    @abstractmethod
    def compare_models(self, grammar_ids: Optional[List[str]] = None,
                       model_ids: Optional[List[str]] = None,
                       set_names: List[str] = None) -> None:
        pass

    def get_labels(self, grammar_id: str, model_id: str,
                   set_name: str) -> int:
        predict_file_name: str = self.evaluation_dir + \
            grammar_id + "/" + \
            model_id + "/" + c.EvaluatorID.PREDICT + \
            "/" + set_name + "-y-hat.txt"
        if os.path.isfile(predict_file_name):
            with open(predict_file_name, "r") as f:
                column_line: str = f.readline()
            columns: List[str] = column_line.split("\t")
            idx = [column.startswith("y_hat_") for column in columns]
            columns = list(compress(columns, idx))

            return [column.replace("y_hat_", "").strip() for column in columns]
        else:
            self.logger.warning("file does not exist: %s", predict_file_name)
            return None
