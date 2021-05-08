"""
MIT - CSAIL - Gifford Lab - seqgra

Grammar heatmap

@author: Konstantin Krismer
"""
import logging
import os
import shutil
import subprocess
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pkg_resources

import seqgra.constants as c


class GrammarPositionHeatmap:
    @staticmethod
    def detect_task(y: List[str]) -> str:
        for example_y in y:
            if "|" in example_y:
                return c.TaskType.MULTI_LABEL_CLASSIFICATION

        return c.TaskType.MULTI_CLASS_CLASSIFICATION

    @staticmethod
    def detect_max_length(annotations: List[str]) -> int:
        max_length: int = 0
        for annotation in annotations:
            if len(annotation) > max_length:
                max_length = len(annotation)

        return max_length

    @staticmethod
    def create(output_dir: str, set_name: str, task: Optional[str] = None,
               max_length: Optional[int] = None) -> None:
        logger = logging.getLogger(__name__)

        file_name: str = output_dir + "/" + set_name + "-grammar-heatmap"
        annotation_file_name: str = output_dir + "/" + set_name + "-annotation.txt"

        if os.path.isfile(annotation_file_name) and \
                not os.path.isfile(file_name + ".pdf"):
            df = pd.read_csv(annotation_file_name, sep="\t",
                             dtype={"annotation": "string", "y": "string"})
            df = df.fillna("")
            counts: Dict[str, List[int]] = dict()
            annotations: List[str] = df["annotation"].tolist()

            if task is None:
                task = GrammarPositionHeatmap.detect_task(df["y"].tolist())

            if max_length is None:
                max_length = GrammarPositionHeatmap.detect_max_length(
                    annotations)

            if task == c.TaskType.MULTI_CLASS_CLASSIFICATION:
                y: List[str] = df["y"].tolist()
            elif task == c.TaskType.MULTI_LABEL_CLASSIFICATION:
                y: List[str] = df["y"].replace(np.nan, "", regex=True).tolist()

            for i, annotation in enumerate(annotations):
                labels: List[str] = None
                if task == c.TaskType.MULTI_CLASS_CLASSIFICATION:
                    labels = [y[i]]
                elif task == c.TaskType.MULTI_LABEL_CLASSIFICATION:
                    if y[i] == "":
                        labels = ["none (background)"]
                    else:
                        labels = y[i].split("|")

                for label in labels:
                    if label not in counts:
                        counts[label] = [0] * max_length

                    for j, letter in enumerate(annotation):
                        if letter == c.PositionType.GRAMMAR:
                            counts[label][j] += 1

            label_column: List[str] = list()
            position_column: List[int] = list()
            grammar_probability_column: List[float] = list()

            for label, grammar_counts in counts.items():
                for i, grammar_count in enumerate(grammar_counts):
                    label_column.append(label)
                    position_column.append(i + 1)
                    grammar_probability_column.append(
                        grammar_count / len(annotations))

            plot_df = pd.DataFrame(
                {"label": label_column,
                 "position": position_column,
                 "grammar_probability": grammar_probability_column})
            plot_df.to_csv(file_name + ".txt", sep="\t", index=False)

            plot_script: str = pkg_resources.resource_filename(
                "seqgra", "simulator/heatmap/heatmap.R")

            cmd = ["Rscript", "--no-save", "--no-restore", "--quiet",
                   plot_script, file_name + ".txt", file_name + ".pdf"]

            try:
                subprocess.call(cmd, universal_newlines=True)
            except subprocess.CalledProcessError as exception:
                logger.warning("failed to create grammar position "
                               "heatmaps: %s", exception.output)
            except FileNotFoundError as exception:
                logger.warning("Rscript not on PATH, skipping "
                               "grammar position heatmaps")
