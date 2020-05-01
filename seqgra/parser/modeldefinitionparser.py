"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file parser 
(using Strategy design pattern)

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod
from typing import List, Dict

from seqgra.model import ModelDefinition
from seqgra.model.model import Architecture


class ModelDefinitionParser(ABC):
    @abstractmethod
    def get_model_id(self) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def get_task(self) -> str:
        pass

    @abstractmethod
    def get_sequence_space(self) -> str:
        pass

    @abstractmethod
    def get_library(self) -> str:
        pass

    @abstractmethod
    def get_implementation(self) -> str:
        pass

    @abstractmethod
    def get_input_encoding(self) -> str:
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        pass

    @abstractmethod
    def get_seed(self) -> int:
        pass

    @abstractmethod
    def get_architecture(self) -> Architecture:
        pass

    @abstractmethod
    def get_loss_hyperparameters(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_optimizer_hyperparameters(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_training_process_hyperparameters(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_model_definition(self) -> ModelDefinition:
        pass
