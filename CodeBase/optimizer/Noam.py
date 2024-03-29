import torch
import numpy as np


class NoamOpt(object):
    """
    Optim wrapper that implements rate.
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.epoch=0
        self.best_mad=np.inf
        self.best_fad=np.inf

    def step(self):
        """
        Update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement `lrate` above
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** -0.5) * min(step ** -0.5, step * self.warmup ** -1.5)
    
    def zero_grad(self):
        self.optimizer.zero_grad()


class Noam(object):

    def __init__(self,parameters,emb_size,factor,warmup, lr):
        self.optimizer = NoamOpt(emb_size,factor,warmup,
        torch.optim.Adam(parameters, lr=lr, betas=(0.9,0.98), eps=1e-9) )

    def step(self):
        self.optimizer.step()

    def rate(self,step=None):
        return self.optimizer(step)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def setWarmUpFactor(self, factor):
        self.optimizer.warmup *= factor


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))