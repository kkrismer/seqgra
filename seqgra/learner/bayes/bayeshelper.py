"""MIT - CSAIL - Gifford Lab - seqgra

TensorFlow Keras learner helper class

@author: Konstantin Krismer
"""
import logging
import math
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import pkg_resources

import seqgra.constants as c
from seqgra import MiscHelper
from seqgra.learner import Learner
from seqgra.model import DataDefinition
from seqgra.model.data import Condition
from seqgra.model.data import SequenceElement
from seqgra.model.model import Architecture
from seqgra.parser import DataDefinitionParser
from seqgra.parser import XMLDataDefinitionParser


class BayesOptimalHelper:
    @staticmethod
    def train_model(learner: Learner) -> None:
        # save number of model parameters
        with open(learner.output_dir +
                  "num-model-parameters.txt", "w") as model_param_file:
            model_param_file.write("number of trainable parameters\t0\n")
            model_param_file.write("number of non-trainable parameters\t0\n")
            model_param_file.write("number of all parameters\t0\n")

    @staticmethod
    def ppm_to_pwm(ppm, alphabet_size) -> Any:
        ppm = np.where(np.isclose(ppm, 0.0), 0.0001, ppm)
        return np.log2(ppm * alphabet_size)

    @staticmethod
    def se_to_pwm(sequence_element: SequenceElement) -> Any:
        # only MatrixBasedSequenceElement supported
        alphabet_size: int = len(sequence_element.positions[0])
        width: int = sequence_element.get_max_length()
        ppm = np.zeros((width, alphabet_size))

        if alphabet_size == 4:
            for i, position in enumerate(sequence_element.positions):
                total_probability = sum([letter.probability
                                         for letter in position])
                for letter in position:
                    index: int = -1
                    if letter.token.upper() == "A":
                        index = 0
                    elif letter.token.upper() == "C":
                        index = 1
                    elif letter.token.upper() == "G":
                        index = 2
                    elif letter.token.upper() == "T":
                        index = 3
                    else:
                        raise Exception("unsupported letter: " +
                                        letter.token.upper())

                    ppm[i, index] = letter.probability / total_probability

        elif alphabet_size == 20:
            pass
        else:
            raise Exception("unsupported alphabet")

        return BayesOptimalHelper.ppm_to_pwm(ppm, alphabet_size)

    @ staticmethod
    def create_model(learner: Learner) -> None:
        arch: Architecture = learner.definition.architecture
        if arch.external_model_format != "data-definition":
            raise Exception("invalid external model format: " +
                            arch.external_model_format)
        path: str = arch.external_model_path
        if os.path.isfile(path):
            with open(path, "r") as data_definition_file:
                data_config: str = data_definition_file.read()
            data_def_parser: DataDefinitionParser = XMLDataDefinitionParser(
                data_config)

            data_definition: DataDefinition = \
                data_def_parser.get_data_definition()
            learner.model = (data_definition.conditions,
                             BayesOptimalHelper.create_se_pwm_dict(
                                 data_definition.sequence_elements))
        else:
            raise Exception("data definition file does not exist: " + path)

    @staticmethod
    def print_model_summary(learner: Learner):
        if learner.model:
            for condition in learner.model[0]:
                print(condition)
            for sequence_element in learner.model[1]:
                print(sequence_element)
        else:
            print("uninitialized model")

    @staticmethod
    def set_seed(learner: Learner) -> None:
        random.seed(learner.definition.seed)
        np.random.seed(learner.definition.seed)

    @staticmethod
    def write_session_info(learner: Learner) -> None:
        with open(learner.output_dir + "session-info.txt", "w") as session_file:
            session_file.write(
                "seqgra package version: " +
                pkg_resources.require("seqgra")[0].version + "\n")
            session_file.write("NumPy version: " + np.version.version + "\n")
            session_file.write("Python version: " + sys.version + "\n")

    @staticmethod
    def score_example(example, pwm) -> Any:
        scores = np.zeros((example.shape[0] - pwm.shape[0] + 1))

        for i in range(example.shape[1]):
            scores += np.correlate(example[:, i], pwm[:, i], mode="valid")

        return scores

    @staticmethod
    def create_se_pwm_dict(sequence_elements: List[SequenceElement]) -> Dict[str, Any]:
        pwm_dict: Dict[str, Any] = dict()
        for se in sequence_elements:
            pwm_dict[se.sid] = BayesOptimalHelper.se_to_pwm(se)

        return pwm_dict

    @staticmethod
    def get_pwms_for_label(label: str, conditions: List[Condition],
                           pwm_dict: Dict[str, Any]) -> Dict[str, Any]:
        condition: Condition = Condition.get_by_id(conditions, label)
        pwms: Dict[str, Any] = dict()
        if condition:
            for rule in condition.grammar:
                for se in rule.sequence_elements:
                    pwms[se.sid] = pwm_dict[se.sid]

        return pwms

    @staticmethod
    def get_pwm_max_score(pwm) -> float:
        return np.sum(np.max(pwm, axis=1))

    @staticmethod
    def get_pwm_min_score(pwm) -> float:
        return np.sum(np.min(pwm, axis=1))

    @staticmethod
    def normalize_pwm_score(score: float, pwm) -> float:
        min_score: float = BayesOptimalHelper.get_pwm_min_score(pwm)
        max_score: float = BayesOptimalHelper.get_pwm_max_score(pwm)

        if math.isclose(min_score, max_score):
            return 0.5
        else:
            if min_score < 0:
                # squish negative range
                min_score = (-1.0) * (abs(min_score) ** (1.0 / 4.0))
            if score < 0:
                score = (-1.0) * (abs(score) ** (1.0 / 4.0))
            return (score - min_score) / (max_score - min_score)

    @staticmethod
    def predict(learner: Learner, x: Any, encode: bool = True,
                silent: bool = False):
        """ This is the forward calculation from x to y
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        if encode:
            x = learner.encode_x(x)

        conditions: List[Condition] = learner.model[0]
        y_hat = np.zeros((x.shape[0], len(learner.definition.labels)))

        for example_index in range(x.shape[0]):
            for i, label in enumerate(learner.definition.labels):
                # get rules for label
                # for now: just PWM
                pwms: Dict[str, Any] = BayesOptimalHelper.get_pwms_for_label(
                    label, conditions, learner.model[1])
                raw_scores: Dict[str, Any] = dict()
                for sid, pwm in pwms.items():
                    raw_scores[sid] = BayesOptimalHelper.score_example(
                        x[example_index, :, :], pwm)
                    condition: Condition = Condition.get_by_id(
                        conditions, label)

                for rule in condition.grammar:
                    for se in rule.sequence_elements:
                        pwm = pwms[se.sid]
                        if rule.position == "random":
                            raw_score = max(raw_scores[se.sid])
                        else:
                            mask = np.zeros((raw_scores[se.sid].shape))
                            if rule.position == "start":
                                start_position: int = 0
                                end_position: int = pwm.shape[0] + 1
                            elif rule.position == "end":
                                start_position: int = max(
                                    mask.shape[0] - pwm.shape[0] - 1, 0)
                                end_position: int = mask.shape[0]
                            elif rule.position == "center":
                                start_position: int = max(
                                    int(mask.shape[0] / 2 - pwm.shape[0] / 2 - 1), 0)
                                end_position: int = min(
                                    start_position + pwm.shape[0] + 2, mask.shape[0])
                            else:
                                start_position: int = max(
                                    int(rule.position) - pwm.shape[0] / 2 - 1, 0)
                                end_position: int = min(
                                    int(rule.position) + pwm.shape[0] + 2, mask.shape[0])

                            start_position = int(start_position)
                            end_position = int(end_position)
                            mask[start_position:end_position] = 1.0
                            raw_score = max(raw_scores[se.sid] * mask)

                        y_hat[example_index, i] += \
                            BayesOptimalHelper.normalize_pwm_score(raw_score,
                                                                   pwm)

                    if rule.spacing_constraints:
                        temp_combined_score: float = 0.0
                        y_hat[example_index, i] = 0.0
                        for spacing_contraint in rule.spacing_constraints:
                            raw_scores_se1 = raw_scores[spacing_contraint.sequence_element1.sid]
                            raw_scores_se2 = raw_scores[spacing_contraint.sequence_element2.sid]
                            pwm_se1 = pwms[spacing_contraint.sequence_element1.sid]
                            pwm_se2 = pwms[spacing_contraint.sequence_element2.sid]
                            temp_combined_score = 0.0

                            for j in range(raw_scores_se1.shape[0]):
                                first_pos: int = min(j + spacing_contraint.min_distance,
                                                     raw_scores_se2.shape[0])
                                last_pos: int = min(j + spacing_contraint.max_distance,
                                                    raw_scores_se2.shape[0])

                                if first_pos < last_pos:
                                    temp_combined_score = \
                                        BayesOptimalHelper.normalize_pwm_score(
                                            raw_scores_se1[j], pwm_se1) + \
                                        BayesOptimalHelper.normalize_pwm_score(
                                            max(raw_scores_se2[first_pos:last_pos]), pwm_se2)
                                    if temp_combined_score > y_hat[example_index, i]:
                                        y_hat[example_index,
                                              i] = temp_combined_score

                            if spacing_contraint.order == "random":
                                for j in range(raw_scores_se2.shape[0]):
                                    first_pos: int = min(j + spacing_contraint.min_distance,
                                                         raw_scores_se1.shape[0])
                                    last_pos: int = min(j + spacing_contraint.max_distance,
                                                        raw_scores_se1.shape[0])

                                    if first_pos < last_pos:
                                        temp_combined_score = \
                                            BayesOptimalHelper.normalize_pwm_score(
                                                raw_scores_se2[j], pwm_se2) + \
                                            BayesOptimalHelper.normalize_pwm_score(
                                                max(raw_scores_se1[first_pos:last_pos]), pwm_se1)
                                        if temp_combined_score > y_hat[example_index, i]:
                                            y_hat[example_index,
                                                  i] = temp_combined_score

            if not silent and x.shape[0] > 5000:
                MiscHelper.print_progress_bar(example_index, x.shape[0] - 1)

        if learner.definition.task == c.TaskType.MULTI_CLASS_CLASSIFICATION:
            # shift
            zero_col = np.zeros((y_hat.shape[0], 1))
            y_hat -= np.hstack((y_hat, zero_col)).min(axis=1)[:, None]
            # scale to [0, 1]
            y_hat /= y_hat.sum(axis=1)[:, None]

        return y_hat

    @staticmethod
    def evaluate_model(learner: Learner, x: List[str], y: List[str]):
        y_hat = BayesOptimalHelper.predict(learner, x)
        y_hat = learner.decode_y(y_hat)
        accuracy = [y_i == y_hat_i for y_i, y_hat_i in zip(y, y_hat)]
        accuracy = np.asarray(accuracy).astype(int)
        accuracy = np.sum(accuracy) / len(accuracy)

        return {"loss": float("nan"), "accuracy": accuracy}
