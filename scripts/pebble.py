import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools

class Pebble:
    def __init__(self, model, data, lr = 0.001, k = 25):
        self.gen_model = model
        self.data = data
        self.lr = lr
        self.k = k
        self._lrs = torch.nn.ParameterDict({
            k: torch.nn.Parameter(torch.tensor(self.lr), requires_grad = True)
            for k, v in self.gen_model.params.items()
        })
        self._optim = torch.optim.Adam(itertools.chain(self.gen_model.params.values(), self._lrs.values()), lr = self.lr)
    
    def _orcale_reward(self, reward1, reward2):
        return 1.0 * (reward1 > reward2)
    
    def _compute_loss(self):
        pass


