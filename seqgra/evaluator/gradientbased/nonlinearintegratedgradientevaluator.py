"""Non-linear Integrated Gradient Evaluator
"""
from typing import Optional

import numpy as np
from sklearn.neighbors import BallTree
import torch
from torch.autograd import Variable

import seqgra.constants as c
from seqgra.evaluator.gradientbased import AbstractGradientEvaluator
from seqgra.learner import Learner


class NonlinearIntegratedGradientEvaluator(AbstractGradientEvaluator):
    """Non-linear integrated gradient evaluator for PyTorch models
    """

    def __init__(self, learner: Learner, output_dir: str,
                 importance_threshold: Optional[float] = None,
                 data=None, k=5, reference=None,
                 path_generator=None) -> None:
        # TODO NonlinearIntegratedGradExplainer
        # requires other data and how to handle reference (default is None)
        super().__init__(c.EvaluatorID.NONLINEAR_INTEGRATED_GRADIENTS,
                         "Nonlinear Integrated Gradients", learner,
                         output_dir, importance_threshold)
        self.reference = reference
        if path_generator is not None:
            self._path_fnc = path_generator
        else:
            self._path_fnc = lambda args: NonlinearIntegratedGradientEvaluator.sequence_path(args, data, k)

    def explain(self, x, y):
        if self.reference is None:
            self.reference = x.data.clone()
            self.reference = self.reference[:, :, torch.randperm(
                self.reference.size()[2])]

        grad = 0
        x_data = x.data.clone()
        new_data, nsteps = self._path_fnc((x_data.cpu().numpy(),
                                           self.reference.cpu().numpy()))
        for i in range(nsteps):
            new_x = torch.from_numpy(new_data[i])
            new_x = new_x.float()
            new_x = Variable(new_x.unsqueeze(0).to(self.learner.device),
                             requires_grad=True)
            g = self._backprop(new_x, y)
            grad += g

        return grad * x_data / nsteps

    @staticmethod
    def dijkstra(u, v, distances, indices):
        sptset = np.zeros((distances.shape[0]))

        sptset[u] = 1.0
        prev_i = [u]
        dists = np.zeros((indices.shape[0]))
        preds = np.zeros((indices.shape[0]))
        while True:
            next_i = [(j, indices[j, ii], ii)
                      for j in prev_i for ii in range(indices[j, :].shape[0])]

            min_ind = None
            min_min = None
            for ind in next_i:

                if sptset[ind[1]] == 0.0:
                    if min_ind == None:
                        min_ind = ind[1]
                        min_min = dists[ind[0]]+distances[ind[0], ind[2]]
                        preds[ind[1]] = ind[0]
                    elif min_min > (dists[ind[0]]+distances[ind[0], ind[2]]):

                        min_ind = ind[1]
                        min_min = dists[ind[0]]+distances[ind[0], ind[2]]
                        preds[ind[1]] = ind[0]

            sptset[min_ind] = 1.0
            dists[min_ind] = min_min
            prev_i = [j for j in range(distances.shape[0]) if sptset[j] == 1.0]
            if min_ind == v:
                break
            if sum(sptset) == indices.shape[0]:
                break
        return dists, preds

    @staticmethod
    def shortest_path(start, end, distances, indices):
        dists, preds = NonlinearIntegratedGradientEvaluator.dijkstra(start, end, distances, indices)
        v = end
        path = [int(v)]
        while v != start:
            v = preds[int(v)]
            path.append(int(v))
        path.reverse()
        return path, len(path)

    @staticmethod
    def sequence_path(args, data, k):
        """distances = [[1,40],[1,35],[40,3],[35,1],[4,1]]
        indices = [[1,2],[0,3],[0,4],[1,4],[2,3]]
        nddist = np.array([np.array(xi) for xi in distances])
        ndinds = np.array([np.array(xi) for xi in indices])
        sp = shortest_path(0,4,
                        nddist,
                        ndinds)
        print(sp)

        #unit tests change numbers to test path
        # 0-1-3
        # |
        # 2
        # |
        # 4
        """
        start, end = args
        X = np.concatenate([data.reshape((data.shape[0], -1)),
                            start.reshape((1, -1)),
                            end.reshape((1, -1))], axis=0)
        bt = BallTree(X, leaf_size=3, metric='hamming')

        distances, indices = bt.query(X, k=k, return_distance=True)
        path, nsteps = NonlinearIntegratedGradientEvaluator.shortest_path(
            X.shape[0]-2, X.shape[0]-1, distances, indices)

        return [X[p, :].reshape(data.shape[1:]) for p in path], nsteps
