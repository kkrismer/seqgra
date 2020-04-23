"""
MIT - CSAIL - Gifford Lab - seqgra

- abstract base class for all evaluators

@author: Konstantin Krismer
"""
from __future__ import annotations

import os
import logging
from typing import List
import random

import numpy as np
import pandas as pd

from seqgra.learner.learner import Learner
from seqgra.evaluator.evaluator import Evaluator
from seqgra.evaluator.sis import sis_collection
from seqgra.evaluator.sis import make_empty_boolean_mask_broadcast_over_axis
from seqgra.evaluator.sis import produce_masked_inputs


class SISEvaluator(Evaluator):
    def __init__(self, learner: Learner, output_dir: str) -> None:
        super().__init__("sis", learner, output_dir)

    def evaluate_model(self, set_name: str = "test") -> None:
        labels: List[str] = self.learner.definition.labels

        for i in range(len(labels)):
            sis_results = self.find_sis(
                labels[i], i, set_name, n=10,
                select_randomly=True, threshold=0.5)
            self.save_results(sis_results, set_name + "-" + labels[i])
            print(labels[i])
            print("precision:")
            print(self.calculate_precision(sis_results))
            print("recall:")
            print(self.calculate_recall(sis_results))
    
    def save_results(self, results, name: str) -> None:
        if results is None:
            df = pd.DataFrame([], columns=["input", "annotation", "sis"])
        else:
            tmp = results.copy()
            for i in range(len(tmp)):
                result = tmp[i]
                tmp[i] = (result[0], result[1], ";".join(result[2]))

            df = pd.DataFrame(tmp, columns=["input", "annotation", "sis"])

        df.to_csv(self.output_dir + name + ".txt", sep = "\t", index=False)
    
    def load_results(self, name: str):
        df = pd.read_csv(self.output_dir + name + ".txt", sep = "\t")

        results = [tuple(x) for x in df.values]
        for i in range(len(results)):
            result = results[i]
            results[i] = (result[0], result[1], result[2].split(";"))
        return results

    def plot_sis_heatmap(self, results, name: str, image_format: str = "pdf",
                         grammar_letter: str = "G", background_letter: str = "_", masked_letter: str = "N"):
        pass
        # if results is not None and len(results) > 0:
        #     n = len(results[0][0])

        #     # heatmap = np.zeros((len(results), n), dtype = object)
        #     # for i in range(len(results)):
        #     #     annotation = results[i][1]
        #     #     sis = results[i][2]
        #     #     for j in range(n):
        #     #         if annotation[j] == grammar_letter:
        #     #             # grammar position only, until overridden
        #     #             heatmap[i, j] = "grammar, not part of SIS"

        #     #             for k in range(len(sis)):
        #     #                 if len(sis[k]) == n and sis[k][j] != masked_letter:
        #     #                     # grammar and SIS position only
        #     #                     heatmap[i, j] = "grammar, part of SIS"
        #     #         else:
        #     #             # neither grammar nor SIS position, until overridden
        #     #             heatmap[i, j] = "background, not part of SIS"
                        
        #     #             for k in range(len(sis)):
        #     #                 if len(sis[k]) == n and sis[k][j] != masked_letter:
        #     #                     # SIS position only
        #     #                     heatmap[i, j] = "background, part of SIS"

        #     df = pd.DataFrame(columns=["example", "position", "group"])

        #     for i in range(len(results)):
        #         annotation = results[i][1]
        #         sis = results[i][2]
        #         for j in range(n):
        #             if annotation[j] == grammar_letter:
        #                 # grammar position only, until overridden
        #                 group_label = "grammar, not part of SIS"

        #                 for k in range(len(sis)):
        #                     if len(sis[k]) == n and sis[k][j] != masked_letter:
        #                         # grammar and SIS position only
        #                         group_label = "grammar, part of SIS"
        #             else:
        #                 # neither grammar nor SIS position, until overridden
        #                 group_label = "background, not part of SIS"
                        
        #                 for k in range(len(sis)):
        #                     if len(sis[k]) == n and sis[k][j] != masked_letter:
        #                         # SIS position only
        #                         group_label = "background, part of SIS"
                    
        #             df = df.append({"example": i + 1, "position": j + 1, "group": group_label}, ignore_index=True)

        #     p = ggplot(df, aes(x = "position", y = "example", fill = "factor(group)")) + \
        #         geom_tile() + \
        #         scale_x_discrete(breaks=np.arange(0, n + 1, 25).tolist()) + \
        #         scale_fill_manual(values = ["white", "green", "yellow", "red"], 
        #                           labels = ["background, not part of SIS", "grammar, part of SIS", "grammar, not part of SIS", "background, part of SIS"],
        #                           drop = False) + \
        #         theme(legend_title = element_blank())
        #     ggsave(plot=p, filename = self.output_dir + name + "." + image_format)

            # idx = {"background, not part of SIS": 0,
            #        "grammar, part of SIS": 1,
            #        "grammar, not part of SIS": 2,
            #        "background, part of SIS": 3}

            # aninv =  { val: key for key, val in idx.items()  }
            # f = lambda x: idx[x]
            # fv = np.vectorize(f)
            # Z = fv(heatmap)

            # plt.figure(figsize = (5,5))

            # #cmap = mpl.colors.ListedColormap(["white", "green", "blue", "yellow"])
            # #im = ax.imshow(Z, cmap = cmap)

            # #bounds = [0, 1, 2, 3]
            # #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            # #cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
            # #                               norm=norm,
            # #                               boundaries=[0] + bounds + [13],
            # #                               extend='both',
            # #                               ticks=bounds,
            # #                               spacing='proportional',
            # #                               orientation='horizontal')


            # im = plt.imshow(Z, interpolation='none', aspect='auto')
            # #im = ax.imshow(Z, cmap = plt.cm.get_cmap("Blues", 4), interpolation='none')

            # ax = plt.gca()
            # ax.set_yticks(np.arange(0, len(results)).tolist())
            # ax.set_yticklabels(np.arange(1, len(results) + 1).tolist())
            # ax.set_xlabel("position")
            # ax.set_ylabel("example")

            # ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

            # values = list(idx.values())
            # values2 = list(idx.keys())
            # # get the colors of the values, according to the 
            # # colormap used by imshow
            # colors = [ im.cmap(im.norm(value)) for value in list(idx.values())]
            # # create a patch (proxy artist) for every color 
            # patches = [ mpl.patches.Patch(color=colors[i], label="Level {l}".format(l=values2[i]) ) for i in range(len(idx)) ]
            # # put those patched as legend-handles into the legend
            # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )


            #color_bar = fig.colorbar(im, ticks=np.arange(0, 4), ax=ax)
            #color_bar.set_ticks(list(idx.values()))
            #color_bar.set_ticklabels(list(idx.keys()))

            # We want to show all ticks...
            #ax.set_xticks(np.arange(len(farmers)))
            #ax.set_yticks(np.arange(len(vegetables)))
            # ... and label them with the respective list entries
            #ax.set_xticklabels(farmers)
            #ax.set_yticklabels(vegetables)


            # Loop over data dimensions and create text annotations.
            #for i in range(len(vegetables)):
            #    for j in range(len(farmers)):
            #        text = ax.text(j, i, heatmap[i, j],
            #                    ha="center", va="center", color="w")

            #plt.tight_layout()
            #plt.savefig(self.output_dir + name + "." + image_format, bbox_inches="tight")  

    def select_random_n_examples(self, for_label, for_set, n, threshold):
        input_df, annotation_df = self.__select_examples(for_label, for_set, threshold)

        if len(input_df.index) == 0:
            logging.warn("no correctly labeled examples with label '" + for_label + "' prediction threshold > " + str(threshold) + " in set")
            return (None, None)
        elif n > len(input_df.index):
            logging.warn("n is larger than number of correctly labeled examples with label '" + for_label + "' prediction threshold > " + str(threshold) + " in set")
            n = len(input_df.index)

        idx: List[int] = list(range(len(input_df.index)))
        random.shuffle(idx)
        idx = idx[:n]

        input_df = input_df.iloc[idx]
        annotation_df = annotation_df.iloc[idx]

        return (input_df["x"].tolist(), annotation_df["annotation"].tolist())

    def select_first_n_examples(self, for_label, for_set, n, threshold):
        input_df, annotation_df = self.__select_examples(for_label, for_set, threshold)

        if len(input_df.index) == 0:
            logging.warn("no correctly labeled examples with label '" + for_label + "' prediction threshold > " + str(threshold) + " in set")
            return (None, None)
        elif n > len(input_df.index):
            logging.warn("n is larger than number of correctly labeled examples with label '" + for_label + "' prediction threshold > " + str(threshold) + " in set")
            n = len(input_df.index)

        input_df = input_df.iloc[range(n)]
        annotation_df = annotation_df.iloc[range(n)]

        return (input_df["x"].tolist(), annotation_df["annotation"].tolist())

    def find_sis(self, for_label, label_index, for_set, n=10,
                 select_randomly=False,
                 threshold=0.9):
        if select_randomly:
            examples = self.select_random_n_examples(for_label, for_set, n, threshold)
        else:
            examples = self.select_first_n_examples(for_label, for_set, n, threshold)

        decoded_examples = examples[0]
        annotations = examples[1]

        if decoded_examples is None:
            return None

        encoded_examples = self.learner.encode_x(decoded_examples)

        def sis_predict(x):
            return np.array(self.learner.predict(x, encode=False))[:, label_index]

        input_shape = encoded_examples[0].shape
        fully_masked_input = np.ones(input_shape) * 0.25
        initial_mask = make_empty_boolean_mask_broadcast_over_axis(
            input_shape, 1)

        return [(decoded_examples[i],
                 annotations[i],
                 self.__produce_masked_inputs(encoded_examples[i],
                                              sis_predict,
                                              threshold,
                                              fully_masked_input,
                                              initial_mask))
                for i in range(len(encoded_examples))]

    def calculate_precision(self, x, grammar_letter="G", background_letter="_", masked_letter="N") -> List[float]:
        if x is None:
            precision_values: List[float] = []
        else:
            precision_values: List[float] = [self.__calculate_precision(sis,
                                                                        grammar_letter=grammar_letter,
                                                                        background_letter=background_letter,
                                                                        masked_letter=masked_letter) for sis in x]
        return precision_values

    def calculate_recall(self, x, grammar_letter="G", background_letter="_", masked_letter="N") -> List[float]:
        if x is None:
            recall_values: List[float] = []
        else:
            recall_values: List[float] = [self.__calculate_recall(sis,
                                                                grammar_letter=grammar_letter,
                                                                background_letter=background_letter,
                                                                masked_letter=masked_letter) for sis in x]
        return recall_values

    def __calculate_precision(self, x, grammar_letter="G", background_letter="_", masked_letter="N") -> float:
        annotation: str = x[1]
        sis: str = self.__collapse_sis(x[2], masked_letter=masked_letter)

        if sis == "":
            return 1.0
        else:
            num_selected: int = 0
            num_selected_relevant: int = 0
            for i, c in enumerate(sis):
                if c != masked_letter:
                    num_selected += 1
                    if annotation[i] == grammar_letter:
                        num_selected_relevant += 1

            if num_selected == 0:
                return 1.0

            return num_selected_relevant / num_selected

    def __calculate_recall(self, x, grammar_letter="G", background_letter="_", masked_letter="N") -> float:
        annotation: str = x[1]
        sis: str = self.__collapse_sis(x[2], masked_letter=masked_letter)
        num_relevent: int = 0

        if sis == "":
            for i, c in enumerate(annotation):
                if c == grammar_letter:
                    num_relevent += 1
            if num_relevent == 0:
                return 1.0
            else:
                return 0.0
        else:
            num_relevant_selected: int = 0
            for i, c in enumerate(annotation):
                if c == grammar_letter:
                    num_relevent += 1
                    if sis[i] != masked_letter:
                        num_relevant_selected += 1

            if num_relevent == 0:
                return 1.0

            return num_relevant_selected / num_relevent

    def __collapse_sis(self, sis: List[str], masked_letter: str = "N") -> str:
        if len(sis) == 0:
            return ""
        elif len(sis) == 1:
            return sis[0]
        else:
            collapsed_sis: str = sis[0]
            sis.pop(0)
            for i, c in enumerate(collapsed_sis):
                if c == masked_letter:
                    for s in sis:
                        if s[i] != masked_letter:
                            collapsed_sis = collapsed_sis[:i] + \
                                s[i] + collapsed_sis[(i + 1):]
            return collapsed_sis

    def __produce_masked_inputs(self, x, sis_predict, threshold, fully_masked_input, initial_mask) -> List[str]:
        collection = sis_collection(sis_predict, threshold, x,
                                    fully_masked_input,
                                    initial_mask=initial_mask)

        if len(collection) > 0:
            sis_masked_inputs = produce_masked_inputs(x,
                                                      fully_masked_input,
                                                      [sr.mask for sr in collection])
            return self.learner.decode_x(sis_masked_inputs).tolist()
        else:
            return list()

    def __get_valid_file(self, data_file: str) -> str:
        data_file = data_file.replace("\\", "/").replace("//", "/").strip()
        if os.path.isfile(data_file):
            return data_file
        else:
            raise Exception("file does not exist: " + data_file)

    def __select_examples(self, for_label, set_name, threshold):
        """ 
        Returns all correctly classified examples for a specified label and
        set that exceed the threshold.
  
        Parameters: 
            TODO 

        Returns:
            TODO
        """
        examples_file: str = self.learner.get_examples_file(set_name)
        annotations_file: str = self.learner.get_annotations_file(set_name)

        examples_df = pd.read_csv(examples_file, sep="\t")
        examples_df = examples_df[examples_df.y == for_label]

        annotations_df = pd.read_csv(annotations_file, sep="\t")
        annotations_df = annotations_df[annotations_df.y == for_label]

        # predict with learner and discard misclassified / mislabelled examples

        x = examples_df["x"].tolist()
        y = examples_df["y"].tolist()
        encoded_y = self.learner.encode_y(y)
        y_hat = self.learner.predict(x)

        idx = [i for i in range(len(encoded_y)) if np.argmax(y_hat[i]) == np.argmax(encoded_y[i]) and np.max(y_hat[i]) > threshold]

        examples_df = examples_df.iloc[idx]
        annotations_df = annotations_df.iloc[idx]

        return (examples_df, annotations_df)
