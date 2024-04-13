import numpy as np
import random
import pandas as pd
import os

class DataPreparation:
    def __init__(self):
        self,
        self.k = 25

    def read_file(self, dir_name):
        DATASETS = '../datasets/'
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

    def get_data(self, validation=False, task = None):
        data = []
        if not validation:
            for name in os.listdir("../datasets/mw"):
                inner_df = pd.DataFrame()
                if not (name.startswith('.')):
                    dir_name = 'mw/'+name
                    print(dir_name)
                    df = self.read_file(dir_name)
                    inner_df = pd.concat([inner_df, df])    
                data.append(inner_df)
            data = np.array(data, dtype=object)
        else:
            dir_name = 'mw_valid_policy_v1/'+task+'-v1'
            print(dir_name)
            inner_df = pd.DataFrame()
            df = self.read_file(dir_name)
            inner_df = pd.concat([inner_df, df])    
            data.append(inner_df)
            dir_name = 'mw_valid/'+task+'-v2'
            print(dir_name)
            inner_df = pd.DataFrame()
            df = self.read_file(dir_name)
            inner_df = pd.concat([inner_df, df])    
            data.append(inner_df)
            data = np.array(data, dtype=object)

        return data

    def construct_episodes(self, ml10):
        self.episodes = []
        for task in ml10:
            if(len(task)==0):
                continue
            task_episodes=[]
            row_index = task[task['done'] == True].index.tolist()
            prev=0
            for x in row_index:
                task_episodes.append(task[prev:x+1])
                prev=x+1
            task_episodes = np.array(task_episodes,dtype=object)
            self.episodes.append(task_episodes)
        self.episodes = np.array(self.episodes,dtype=object)

    def form_sigma_groups(self, episode):
        split_indices = np.arange(self.k, episode.shape[0], self.k)
        if len(split_indices) != 0:
            l_segment = np.array_split(episode.iloc[::-1][:(self.k*(episode.shape[0] // self.k))], split_indices)
            for i in range(len(l_segment)):
                l_segment[i] = l_segment[i].iloc[::-1]
            if(len(l_segment[-1])<25):
                l_segment=l_segment[:-1]
            return l_segment
        else:
            return None

    def prepare_data(self):
        X = []
        y = []
        episodes = self.episodes
        sigmas = []
        for task in episodes:
            sigma = []
            for episode in task:
                segment = self.form_sigma_groups(episode)
                if segment is not None:
                    sigma.append(segment)
            sigmas.append(sigma)
        sigmas = np.array(sigmas, dtype=object)
        all_lengths = [len(episode) for task in sigmas for episode in task]
        self.num_segments = min(all_lengths)
        if(self.num_segments<3):
            self.num_segments=3
        for task in sigmas:
            task_list=[]
            for episode in task:
                ep_list=[]
                if(len(episode)<self.num_segments):
                    continue
                for i in range(self.num_segments):
                    y.append(episode[i]["reward"])
                    ep_list.append(episode[i].drop('reward', axis=1))
                task_list.append(ep_list)
            task_list=np.array(task_list, dtype=object).reshape(-1,1) # convert row vector to col vector
            X.append(task_list)
        X_new=[]
        task_counter = []
        for task in X:
            counter = 0
            for i in range(0,len(task),4):
                X_new.append(np.concatenate((task[i][0], task[i+1][0])))
                counter += 1
            task_counter.append(counter)
        X = np.array(X_new, dtype=object)
        task_counts = np.array(task_counter, dtype=object)
        y=np.array(y,dtype=object)
        y=y.flatten()
        return X, y, task_counts

    def batchify(X, y, task_lengths, task_no, num=110):
        if task_no == 0:
            task_beg = 0  
            task_end = task_lengths[0]
        else:    
            task_beg = np.sum(task_lengths[0:task_no])    
            task_end = np.sum(task_lengths[0:task_no+1])
        rand = random.sample(range(0,400), 110)
        X_task = X[task_beg:task_end]
        y_task = y[task_beg:task_end]
        X_random = []
        y_random = []
        for r in rand:
            X_random.append(X_task[r*25 : r*25+25])
            y_random.append(y_task[r*25 : r*25+25])
        X_random = np.array(X_random, dtype=np.float32)
        y_random = np.array(y_random, dtype=np.float32)
        return X_random.reshape(2750,43), y_random.reshape(2750,)
