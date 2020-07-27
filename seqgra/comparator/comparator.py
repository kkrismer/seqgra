"""Abstract base class for all comparators
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from seqgra import MiscHelper


class Comparator(ABC):
    @abstractmethod
    def __init__(self, comparator_id: str, comparator_name: str,
                 analysis_name: str, output_dir: str, 
                 model_labels: Optional[List[str]] = None) -> None:
        self.logger = logging.getLogger(__name__)
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
