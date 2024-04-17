import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import warnings
import copy
import matplotlib.pyplot as plt
import random
import math
from torch.nn import functional as F
warnings.filterwarnings('ignore')

from Data_Prep import DataPreparation as data_prep

class Model(nn.Module):
    def __init__(
        self,
        hidden_layers=[256, 256, 256],
        act=F.leaky_relu,
        output_act=torch.tanh,
    ):
        super().__init__()
        params = {}
        last_dim = 39 + 4
        self.num_layers = len(hidden_layers) + 1
        for i, dim in enumerate(
            hidden_layers
            + [
                1,
            ]
        ):
            weight = torch.empty(last_dim, dim)
            weight.transpose_(0, 1)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight.transpose_(0, 1)
            params[f"linear_w_{i}"] = nn.Parameter(weight)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(params[f"linear_w_{i}"].T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            params[f"linear_b_{i}"] = nn.Parameter(
                nn.init.uniform_(torch.empty(1, dim, requires_grad=True), -bound, bound)
            )
            last_dim = dim

        self.params = nn.ParameterDict(params)
        self.parameters = self.params.parameters 
        self.act = act
        self.output_act = output_act

    def forward(self, x, params=None):
        if params is None:
            params = self.params
        for i in range(self.num_layers):
            x = params[f"linear_b_{i}"] + torch.matmul( x, params[f"linear_w_{i}"] )
            if i == self.num_layers - 1:
                x = self.output_act(x)
            else:
                x = self.act(x)
        return x.squeeze(-1)
    

###################################################
    

class MAML:
    def __init__(
        self,
        input_size,
        hidden_size1,
        hidden_size2,
        hidden_size3,
        outer_lr = 1e-7,
        inner_lr = 0.001,
        num_support=10,
        num_query=10,
        num_inner_steps=5,
        k = 25,
        num_tasks = 10,
        episode_per_task = 1250,
        output_size = 2750,
        beta1 = 0.9, 
        beta2 = 0.999,
        adam_alpha = 0.001,
        adam_eps = 1e-8,
        **kwargs,
    ):
        self.reward_criterion =  nn.BCELoss()
        self.num_support = num_support
        self.num_query = num_query
        self.num_inner_steps = num_inner_steps
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.k = k
        self.num_tasks = num_tasks
        self.episode_per_task = episode_per_task
        self.num_segments = None
        self.model = Model()
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_alpha = adam_alpha
        self.adam_eps = adam_eps
        self.m_dict = {k: torch.zeros_like(v) for k, v in self.model.params.items()}
        self.v_dict = {k: torch.zeros_like(v) for k, v in self.model.params.items()}
        self._lrs = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(torch.tensor(self.inner_lr), requires_grad=True)
                for k, v in self.model.params.items()
            }
        )
        self.optim = optim.Adam(itertools.chain(self.model.params.values(), self._lrs.values()), lr=0.0001)


    def train(self, X, y, task_lengths, num_epochs):
        plt_x = np.arange(0, num_epochs)
        plt_y = np.zeros(num_epochs)
        acc = np.zeros(num_epochs)
        for epoch in tqdm(range(num_epochs)):
            self.optim.zero_grad()
            loss, accuracy = self._outer_step(self.model, X, y, task_lengths) 
            if loss is None:
                return {}
            # print('loss from train:', loss)
            plt_y[epoch] = loss
            acc[epoch] = accuracy
            idx = 0
            if epoch % 25 == 0:
                plt.plot(plt_x[:epoch+1], plt_y[:epoch+1])
                plt.xlabel("Epochs")
                # plt.ylabel("Mean Loss")
                
                # add the accuracy plot
                plt.plot(plt_x[:epoch+1], acc[:epoch+1])
                plt.xlabel("Epochs")
                # plt.ylabel("Mean Accuracy")
                plt.legend(["Loss", "Accuracy"])
                
                plt.show() 
                print(f"Epoch: {epoch}, Loss: {plt_y[epoch]}, accuracy: {acc[epoch]}")
            loss.backward()
            self.optim.step()
        return plt_x, plt_y, acc


    def _outer_step(self, model, X, y, task_lengths):
        outer_losses = []
        accuracy = 0
        for t in range(self.num_tasks):
            X_b, y_b = data_prep.batchify(X, y, task_lengths, t)
            H, W = X_b.shape
            inner_x = X_b[:H//2, :]
            inner_y = y_b[:H//2]
            outer_x = X_b[H//2:, :]
            outer_y = y_b[H//2:]
            weights = self._inner_loop(inner_x, inner_y, model = model)
            loss, acc= self._compute_loss(outer_x, outer_y, model, parameters=weights)
            outer_losses.append(loss)
            accuracy += acc

        # print(outer_losses)
        if (len(outer_losses) == 0):
            return None
        outer_loss = torch.mean(torch.stack(outer_losses))
        # print("Outer Loss: ", loss)

        return outer_loss, accuracy/self.num_tasks


            
    def _inner_loop(self, X, y, model):
        params_og = model.params
        new_dict = {k: torch.clone(v) for k, v in params_og.items()}

        loss, _ = self._compute_loss(X, y, model, parameters = new_dict)
        grad = torch.autograd.grad(loss, new_dict.values(), create_graph=True)

        idx = 0
        for name, w in params_og.items():
            if name in ['linear_b_0', 'linear_b_1', 'linear_b_2', 'linear_b_3']:
                new_dict[name] = new_dict[name] - 0.001 * grad[idx]
            else:
                new_dict[name] = new_dict[name] - 0.001 * grad[idx]
            idx += 1
        return new_dict

    
    def _compute_loss(self, X, y, model, parameters = None):
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        if parameters is not None:
            output_reward = model.forward(X_tensor, parameters) 
        else:
            print('Params are NONE')
            output_reward = model(X_tensor)
        x =  y_tensor.shape[0]
        N = x//self.k
        output_reward = output_reward.reshape(N, self.k)
        output_reward = torch.sum(output_reward, dim=1)
        y_tensor = y_tensor.reshape(N, self.k)
        y_tensor = torch.sum(y_tensor, dim=1)
        loss = 0
        criterion = nn.BCEWithLogitsLoss()
        loss = []
        for i in range(N):
            for j in range(i+1, N):
                if y_tensor[i] > y_tensor[j]:
                    loss.append(criterion(output_reward[j] - output_reward[i], torch.tensor(0.0, requires_grad= True)))
                else:
                    loss.append(criterion(output_reward[j] - output_reward[i], torch.tensor(1.0, requires_grad= True)))

        loss = torch.mean(torch.stack(loss)) #### sum

        #accuracy 
        accuracy = 0
        count = 0
        with torch.no_grad():
            for i in range(N):
                for j in range(i+1, N):
                    label = torch.tensor(1.0)
                    if y_tensor[i] > y_tensor[j]:
                        label = torch.tensor(0.0)
                    logit = output_reward[j] - output_reward[i]
                    pred = logit > 0
                    accuracy += (pred == label).float()
                    count += 1
        accuracy = accuracy/count
        return loss, accuracy