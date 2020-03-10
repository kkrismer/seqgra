"""
MIT - CSAIL - Gifford Lab - seqgra

- class for all saliency evaluators
- models must be in pytorch

@author: Jennifer Hammelma
"""
from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from plotnine import ggplot, geom_tile, aes, ggsave, scale_x_discrete, element_blank, theme, scale_fill_manual

from seqgra.learner.learner import Learner
from seqgra.evaluator.evaluator import Evaluator
from seqgra.evaluator.explainer import backprop as bp
from seqgra.evaluator.explainer import deeplift as df
from seqgra.evaluator.explainer import gradcam as gc
from seqgra.evaluator.explainer import patterns as pt
from seqgra.evaluator.explainer import ebp

class GradientBasedEvaluator(Evaluator):
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        super().__init__(learner,data_dir,output_dir)

    def evaluate_model(self, set_name: str = "test") -> None:
        '''
        TODO make calls to calculate saliency
        of form of SISEvaluator evaluate_model
        '''

    def calculate_saliency(self,data,label):
        result = self.explainer.explain(data,label)
        return self.__explainer_transform(data,result)

    def __explainer_transform(self,data,result):
        pass

class GradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        self.explainer = bp.VanillaGradExplainer(model)
        super().__init__(learner,data_dir,output_dir)
        
    def __explainer_transform(self,data,result):
        return result.cuda().numpy()
    
class GradientxInputEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        self.explainer = bp.GradxInputExplainer(model)
        super().__init__(learner,data_dir,output_dir)
        
    def __explainer_transform(self,data,result):
        return result.cuda().numpy()

class SaliencyEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        self.explainer = bp.SaliencyExplainer(model)
        super().__init__(learner,data_dir,output_dir)
        
    def __explainer_transform(self,data,result):
        return result.cuda().numpy()

class IntegratedGradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        self.explainer = bp.IntegrateGradExplainer(model)
        super().__init__(learner,data_dir,output_dir)
        
    def __explainer_transform(self,data,result):
        return result.cuda().numpy()
    
class NonlinearIntegratedGradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        #TODO NonlinearIntegratedGradExplainer
        # requires other data and how to handle reference (default is None)
        self.explainer = bp.NonlinearIntegrateGradExplainer(model)
        super().__init__(learner,data_dir,output_dir)
        
    def __explainer_transform(self,data,result):
        return result.cuda().numpy()

class GradCamGradientEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        self.explainer = gc.GradCAMExplainer(model)
        super().__init__(learner,data_dir,output_dir)
        
    def __explainer_transform(self,data,result):
        return functional.interpolate(result.view(1,1,-1),size=data.shape[2],mode='linear').cpu().numpy()

class DeepLiftEvaluator(GradientBasedEvaluator):
    #TODO where to set reference?
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        self.explainer = df.DeepLIFTRescaleExplainer(model,'shuffled')
        super().__init__(learner,data_dir,output_dir)
        
    def __explainer_transform(self,data,result):
        return result.cuda().numpy()

class ExcitationBackpropEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        self.explainer = ebp.ExcitationBackpropExplainer(model)
        super().__init__(learner,data_dir,output_dir)
        
    def __explainer_transform(self,data,result):
        return result.cuda().numpy()
    
class ContrastiveExcitationBackpropEvaluator(GradientBasedEvaluator):
    def __init__(self, learner: Learner, data_dir: str, output_dir: str) -> None:
        self.explainer = ebp.ContrastiveExcitationBackpropExplainer(model)
        super().__init__(learner,data_dir,output_dir)
        
    def __explainer_transform(self,data,result):
        return result.cuda().numpy()
    

