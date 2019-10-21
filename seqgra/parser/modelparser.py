"""
MIT - CSAIL - Gifford Lab - seqgra

Abstract base class for configuration file parser (using Strategy design pattern)

@author: Konstantin Krismer
"""
from abc import ABC, abstractmethod
from typing import List, Dict

from seqgra.model.model.architecture import Architecture
from seqgra.model.model.metric import Metric

class ModelParser(ABC):
    @abstractmethod
    def get_id(self) -> str:
        pass
    
    @abstractmethod
    def get_label(self) -> str:
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        pass
    
    @abstractmethod
    def get_library(self) -> str:
        pass
    
    @abstractmethod
    def get_learner_type(self) -> str:
        pass
    
    @abstractmethod
    def get_learner_implementation(self) -> str:
        pass

    @abstractmethod
    def get_metrics(self) -> List[Metric]:
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
