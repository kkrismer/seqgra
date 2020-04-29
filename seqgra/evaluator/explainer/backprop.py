import types

import numpy as np
from torch.autograd import Variable, Function
import torch

import seqgra.evaluator.explainer.path as path


class VanillaDifferenceGradExplainer(object):
    def __init__(self, model):
        self.model = model

    def _backprop(self, inp, ind1, ind2=None):
        output = self.model(inp)
        if ind1 is None:
            ind1 = output.data.max(1)[1]
        if ind2 is None:
            index0 = torch.LongTensor([0]).cuda()
            ind2 = output.data.topk(k=2, sorted=True)[1][0][0].unsqueeze(0)

        grad_out1 = output.data.clone()
        grad_out1.fill_(0.0)

        grad_out2 = output.data.clone()
        grad_out2.fill_(0.0)

        grad_out1.scatter_(1, ind1.unsqueeze(0).t(), 1.0)
        grad_out2.scatter_(1, ind2.unsqueeze(0).t(), 1.0)
        output.backward(grad_out1-grad_out2)
        return inp.grad.data

    def explain(self, inp, ind1=None, ind2=None):
        return self._backprop(inp, ind1, ind2)


class VanillaGradExplainer(object):
    def __init__(self, model):
        self.model = model

    def _backprop(self, inp, ind):
        output = self.model(inp)
        if ind is None:
            ind = output.data.max(1)[1].unsqueeze(0)
        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        grad_out.scatter_(1, ind, 1.0)
        output.backward(grad_out)
        return inp.grad.data

    def explain(self, inp, ind=None, blank=None):
        return self._backprop(inp, ind)


class GradxInputExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(GradxInputExplainer, self).__init__(model)

    def explain(self, inp, ind=None):
        grad = self._backprop(inp, ind)
        return inp.data * grad


class SaliencyExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(SaliencyExplainer, self).__init__(model)

    def explain(self, inp, ind=None):
        grad = self._backprop(inp, ind)
        return grad.abs()


class NonlinearIntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, data, k=5, reference=None, path_generator=None):
        super(NonlinearIntegrateGradExplainer, self).__init__(model)
        self.reference = reference
        if path_generator != None:
            self._path_fnc = path_generator
        else:
            self._path_fnc = lambda args: path.sequence_path(args, data, k)

    def explain(self, inp, ind=None):
        if self.reference == None:
            self.reference = inp.data.clone()
            self.reference = self.reference[:, :, torch.randperm(
                self.reference.size()[2])]

        grad = 0
        inp_data = inp.data.clone()
        new_data, nsteps = self._path_fnc((inp_data.cpu().numpy(),
                                           self.reference.cpu().numpy()))
        for i in range(nsteps):
            new_inp = torch.from_numpy(new_data[i])
            new_inp = new_inp.float()
            new_inp = Variable(new_inp.unsqueeze(0).cuda(), requires_grad=True)
            g = self._backprop(new_inp, ind)
            grad += g

        return grad * inp_data / nsteps


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, steps=100):
        super(IntegrateGradExplainer, self).__init__(model)
        self.steps = steps

    def explain(self, inp, ind=None):
        grad = 0
        inp_data = inp.data.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_inp = Variable(inp_data * alpha, requires_grad=True)
            g = self._backprop(new_inp, ind)
            grad += g

        return grad * inp_data / self.steps


class DeconvExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(DeconvExplainer, self).__init__(model)
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                grad_inp = torch.clamp(grad_output, min=0)
                return grad_inp

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


class GuidedBackpropExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(GuidedBackpropExplainer, self).__init__(model)
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                ctx.save_for_backward(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                output, = ctx.saved_tensors
                mask1 = (output > 0).float()
                mask2 = (grad_output.data > 0).float()
                grad_inp = mask1 * mask2 * grad_output.data
                grad_output.data.copy_(grad_inp)
                return grad_output

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


# modified from https://github.com/PAIR-code/saliency/blob/master/saliency/base.py#L80
class SmoothGradExplainer(object):
    def __init__(self, base_explainer, stdev_spread=0.15,
                 nsamples=25, magnitude=True):
        self.base_explainer = base_explainer
        self.stdev_spread = stdev_spread
        self.nsamples = nsamples
        self.magnitude = magnitude

    def explain(self, inp, ind1=None, ind2=None):
        stdev = self.stdev_spread * (inp.data.max() - inp.data.min())

        total_gradients = 0
        origin_inp_data = inp.data.clone()

        for i in range(self.nsamples):
            noise = torch.randn(inp.size()).cuda() * stdev
            inp.data.copy_(noise + origin_inp_data)
            grad = self.base_explainer.explain(inp, ind1, ind2)

            if self.magnitude:
                total_gradients += grad ** 2
            else:
                total_gradients += grad

        return total_gradients / self.nsamples
