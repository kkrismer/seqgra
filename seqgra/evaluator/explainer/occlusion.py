import torch
import numpy as np
from torch.autograd import Variable
from skimage.util import view_as_windows

from seqgra.learner import Learner

# modified from https://github.com/marcoancona/DeepExplain/blob/master/deepexplain/tensorflow/methods.py#L291-L342
# note the different dim order in pytorch (NCHW) and tensorflow (NHWC)
def occlusion(inp, learner: Learner, window_shape, step=None):
    if isinstance(window_shape, int):
        window_shape = (window_shape, window_shape, 3)
        
    if step is None:
        step = 1
    n, c, h, w = inp.data.size()
    total_dim = c * h * w
    index_matrix = np.arange(total_dim).reshape(h, w, c)
    idx_patches = view_as_windows(index_matrix, window_shape, step).reshape(
        (-1,) + window_shape)
    heatmap = np.zeros((n, h, w, c), dtype=np.float32).reshape((-1), total_dim)
    weights = np.zeros_like(heatmap)
    
    inp_data = inp.data.clone()
    new_inp = Variable(inp_data)
    eval0 = learner.model(new_inp)
    pred_id = eval0.max(1)[1].data[0]
    
    for i, p in enumerate(idx_patches):
        mask = np.ones((h, w, c)).flatten()
        mask[p.flatten()] = 0
        th_mask = torch.from_numpy(mask.reshape(1, h, w, c).transpose(0, 3, 1, 2)).float().to(learner.device)
        masked_xs = Variable(th_mask * inp_data)
        delta = (eval0[0, pred_id] - learner.model(masked_xs)[0, pred_id]).data.cpu().numpy()
        delta_aggregated = np.sum(delta.reshape(n, -1), -1, keepdims=True)
        heatmap[:, p.flatten()] += delta_aggregated
        weights[:, p.flatten()] += p.size
    
    attribution = np.reshape(heatmap / (weights + 1e-10), (n, h, w, c)).transpose(0, 3, 1, 2)
    return attribution