import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DATASETS = '../datasets/'

def read_file(dir_name):
    main_df = pd.DataFrame()
    directory = DATASETS + dir_name
    for filename in os.listdir(directory):
        data = np.load(os.path.join(directory, filename))
        data_dict = {}
        for keys in data.keys():
            data_dict[keys] = list(data[keys])
        df = pd.DataFrame.from_dict(data_dict)
        main_df = pd.concat([main_df, df])
    return main_df

data = pd.DataFrame()
for name in os.listdir("../datasets/mw"):
    if not (name.startswith('.')):
        dir_name = 'mw/'+name
        print(dir_name)
        df = read_file(dir_name)
        data = pd.concat([data, df])

class Model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class PreferenceMAML:
    def __init__(
        self,
        ml10,
        input_size,
        hidden_size1,
        hidden_size2,
        output_size,
        inner_lr=0.01,
        num_support=10,
        num_query=10,
        num_inner_steps=5,
        **kwargs,
    ):
        self.ml10 = ml10
        self.reward_criterion = nn.BCELoss()
        self.num_support = num_support
        self.num_query = num_query
        self.num_inner_steps = num_inner_steps
        self.inner_lr = inner_lr

        self.model = Model(input_size, hidden_size1, hidden_size2, output_size)

    def construct_episodes(self):
        episodes = []
        episode = []
        for _, row in self.ml10.iterrows():
            episode.append(row)
            if row['done']:
                episodes.append(episode)
                episode = []
        return episodes

    def form_sigma_groups(self, episode, k):
        sigmas = []
        segments = []
        q, r = divmod(len(episode), k)
        for i in range(k):
            segments.append(episode[i*q+min(i, r): (i+1)*q+min(i+1, r)])

        for i in range(k):
            sigma_i = segments[i]
            for j in range(i+1, k):
                sigma_j = segments[j]

                sigmas.append((sigma_i, sigma_j))
        return sigmas

    def compare_probabilities(self, sigma1, sigma2):
        exp_sum_rewards_sigma1 = np.exp(sum(row['reward'] for row in sigma1))
        exp_sum_rewards_sigma2 = np.exp(sum(row['reward'] for row in sigma2))
        prob = exp_sum_rewards_sigma1 / (exp_sum_rewards_sigma1 + exp_sum_rewards_sigma2)
        return [0] if prob > 0.5 else [1]

    def prepare_data(self, k):
        X = []
        y = []
        episodes = self.construct_episodes()
        for episode in episodes:
            sigmas = self.form_sigma_groups(episode, k)
            for _ in range(len(sigmas)):
                sigma1 = sigmas[_][0]
                sigma2 = sigmas[_][1]

                obs_action_sigma1 = []
                for row in sigma1:
                    obs_action = list(row['obs']) + list(row['action'])
                    obs_action_sigma1.append(obs_action)

                obs_action_sigma2 = []
                for row in sigma2:
                    obs_action = list(row['obs']) + list(row['action'])
                    obs_action_sigma2.append(obs_action)

                if len(obs_action_sigma1) > len(obs_action_sigma2):
                    obs_action_sigma1 = obs_action_sigma1[1:]
                elif len(obs_action_sigma1) < len(obs_action_sigma2):
                    obs_action_sigma2 = obs_action_sigma2[1:]
                else:
                    continue

                X.append(np.concatenate((obs_action_sigma1, obs_action_sigma2), axis=1))
                y.append(self.compare_probabilities(sigma1, sigma2))

        return X, y

    def setup_optimizers(self, optim_class, optim_kwargs):
        self.optim = optim_class(self.model.parameters(), **optim_kwargs)

    def _train_step(self, X, y):
        self.optim.zero_grad()
        loss = self._outer_step(X, y)
        loss.backward()
        self.optim.step()
        return loss.item()

    def _outer_step(self, X, y):
        outer_losses = []
        for i in tqdm(range(len(X))):
            if len(X[i]) > self.num_support:
                support_X, support_y, query_X, query_y = self._split_support_query(X[i], y[i])
                # Inner loop (adaptation)
                adapted_model = self._inner_loop(support_X, support_y)
                # Compute loss using the adapted model on query set
                query_loss = self._compute_loss(adapted_model, query_X, query_y)
                outer_losses.append(query_loss)
        return torch.mean(torch.stack(outer_losses))

    def _inner_loop(self, support_X, support_y):
        adapted_model = Model(self.model.fc1.in_features, self.model.fc1.out_features,
                              self.model.fc2.out_features, self.model.fc3.out_features)
        adapted_model.load_state_dict(self.model.state_dict())

        inner_optimizer = optim.Adam(adapted_model.parameters(), lr=self.inner_lr)

        for _ in range(self.num_inner_steps):
            inner_optimizer.zero_grad()
            loss = self._compute_loss(adapted_model, support_X, support_y)
            loss.backward()
            inner_optimizer.step()

        return adapted_model

    def _compute_loss(self, model, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        op = model(X_tensor)
        loss = self.reward_criterion(op, y_tensor)
        return loss

    def _split_support_query(self, X, y):
        num_samples = len(X)
        all_indices = np.arange(num_samples)
        # Randomly sample support indices
        support_indices = np.random.choice(num_samples, self.num_support, replace=False)
        query_indices = np.setdiff1d(all_indices, support_indices)
        support_X = X[support_indices]
        query_X = X[query_indices]
        support_y = [y] * self.num_support
        query_y = [y] * len(query_indices)

        return support_X, support_y, query_X, query_y

ml10 = data.copy()
input_size = 86  # Assuming obs has 39 numbers and action has 4 numbers * 2 for pair of sigmas
hidden_size1 = 128
hidden_size2 = 128
output_size = 1
num_epochs = 5
outer_lr = 0.001

model = PreferenceMAML(ml10, input_size, hidden_size1, hidden_size2, output_size)
model.setup_optimizers(optim.Adam, {"lr": outer_lr})

print('Preparing Data.')
X, y = model.prepare_data(k=4)
print('Data Preparation Done.\n')

# Train the model
for epoch in range(num_epochs):
    print(f'\nBeginning Training - Epoch [{epoch+1}/{num_epochs}]')
    loss = model._train_step(X, y)
    print(f"Epoch {epoch+1}, Loss: {loss}")

test = pd.DataFrame()
for name in os.listdir("../datasets/mw_valid"):
    if not (name.startswith('.')):
        dir_name = 'mw_valid/'+name
        print(dir_name)
        df = read_file(dir_name)
        test = pd.concat([data, df])

test.reset_index(drop=True, inplace=True)
Test = PreferenceMAML(test, input_size, hidden_size1, hidden_size2, output_size)
test_X, test_y = Test.prepare_data(k=4)

def evaluate_model(model, X, y):
    predictions = []
    with torch.no_grad():
        for i in range(len(X)):
            X_tensor = torch.tensor(X[i], dtype=torch.float32)
            output = model.model(X_tensor.unsqueeze(0))
            predictions.append(output.squeeze().numpy())

    preds = []
    for _ in range(len(predictions)):
        preds.append((np.array(predictions[_]).mean()))

    pred_label = []
    for i in range(len(preds)):
        pred_label.append([0] if preds[i] > 0.5 else [1])

    sum = 0
    for _ in range(len(y)):
        sum += pred_label[_] == y[_]
    accuracy = sum / len(y)
    return accuracy, pred_label

test_accuracy, pred_labels = evaluate_model(model, test_X, test_y)
print(f'\nTest Accuracy: {test_accuracy}')
