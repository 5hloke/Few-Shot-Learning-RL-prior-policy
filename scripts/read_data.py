import numpy as np
import pandas as pd
import os 
import sys

DATASETS = '../datasets/'
def read_file(dir_name):
    directory = DATASETS + dir_name

    for filename in os.listdir(directory):
        # if filename.endswith(".csv"):
            # data = pd.read_csv(os.path.join(directory, filename))
        data = np.load(os.path.join(directory, filename))
        print(data.keys())
        print(data['obs'].shape)
        data_dict = {}
        for keys in data.keys():
            # if keys == 'obs': 
            #     obs_data = list(data[keys])
            #     data_dict[keys] = obs_data
            # else:
            #     data_dict[keys] = data[keys]
            data_dict[keys] = list(data[keys])
            

        df = pd.DataFrame.from_dict(data_dict)
        print(df['done'].unique())
        print(df)

    return data

if __name__ == '__main__':
    dir_name = 'panda_reach_valid'
    read_file(dir_name)