import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
from Data_Prep import DataPreparation as data_prep

class Pebble:
    def __init__(self, model, data = None, lr = 0.001, k = 25):
        self.gen_model = model
        self.data = data
        self.lr = lr
        self.k = k
        self._lrs = torch.nn.ParameterDict({
            k: torch.nn.Parameter(torch.tensor(self.lr), requires_grad = True)
            for k, v in self.gen_model.model.params.items()
        })
        self._optim = torch.optim.Adam(itertools.chain(self.gen_model.model.params.values(), self._lrs.values()), lr = self.lr)
    
    
    def train(self, X, y, task_lengths, epochs = 500):
        plt_x = np.arange(0, epochs)
        plt_y = np.zeros(epochs)
        for epoch in range(epochs):
            self._optim.zero_grad()
            X_b, y_b = data_prep.batchify(X, y, task_lengths, 0)
            H, W = X_b.shape
            loss, accuracy = self.gen_model._compute_loss(X_b, y_b, self.gen_model.model, self.gen_model.model.params)
            plt_y[epoch] = accuracy
            if epoch % 25 == 0:
                plt.plot(plt_x[:epoch+1], plt_y[:epoch+1])
                plt.xlabel("Epochs")
                plt.ylabel("Mean Accuracy")
                plt.show()
                print(f"Epoch: {epoch}, Accuracy: {plt_y[epoch]}")
            loss.backward()
            self._optim.step()



        


