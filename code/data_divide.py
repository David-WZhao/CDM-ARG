import torch.utils.data as tud
import random
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import torch
# from tqdm import tqdm
import json
import os
import sys
from sklearn.model_selection import train_test_split

# 获取当前脚本的目录（而不是当前工作目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {current_dir}")

# 创建必要的目录（相对于脚本目录）
os.makedirs(os.path.join(current_dir, 'data', 'train_val'), exist_ok=True)
os.makedirs(os.path.join(current_dir, 'data', 'test'), exist_ok=True)

print(f"Created directories: {os.path.join(current_dir, 'data', 'train_val')}")
print(f"Created directories: {os.path.join(current_dir, 'data', 'test')}")

def save_KFold_data(data, K):
    data = data.reset_index(drop=True)
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    cross = 1
    for train_index, val_index in kf.split(data):
        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]
        # train_data.to_pickle('data/train_val/cross_' + str(cross) +'_train.pickle')
        # train_data.to_pickle('data/data_no_anti/train_val/cross_' + str(cross) +'_train.pickle')
        train_data.to_pickle('data/train_val/cross_' + str(cross) +'_train.pickle')
        # train_data.to_pickle('data/data_no_mech/train_val/cross_' + str(cross) +'_train.pickle')
        # val_data.to_pickle('data/train_val/cross_' + str(cross) +'_val.pickle')
        # val_data.to_pickle('data/data_no_anti/train_val/cross_' + str(cross) +'_val.pickle')
        val_data.to_pickle('data/train_val/cross_' + str(cross) +'_val.pickle')
        # val_data.to_pickle('data/data_no_mech/train_val/cross_' + str(cross) +'_val.pickle')
        print('cross_' + str(cross) +' train_val data saved...')
        cross += 1

def save_test_data(test_data):
    # test_data.to_pickle('./data/test/test.pickle')
    # test_data.to_pickle('./data/data_no_anti/test/test.pickle')
    test_data.to_pickle('./data/test/test.pickle')
    # test_data.to_pickle('./data/data_no_mech/test/test.pickle')

    print('test data saved...')

def load_data():
    # data = pd.read_pickle('./data/data_no_anti/res_no_seq_no_twoanti.pickle')
    data = pd.read_pickle('./data/arg_v5_processed.pickle')
    # data = pd.read_pickle('./data/data_no_mech/res_no_seq_no_twomech.pickle')
    anti_count, mech_count, type_count = 15, 6, 2
    #anti_count, mech_count, type_count = 14, 6, 2
    # anti_count, mech_count, type_count = 13, 6, 2
    return data, anti_count, mech_count, type_count

def init_data(data, train_rate):
    train_data = data.sample(int(len(data) * train_rate), random_state=42)
    test_data = data.drop(labels=train_data.index)
    return train_data, test_data

if __name__ == '__main__':
    data, anti_count, mech_count, type_count = load_data()
    train_data, test_data = train_test_split(
    data, 
    test_size=0.2, 
    stratify=data['anti_label'], 
    random_state=42)
    save_test_data(test_data)
    save_KFold_data(train_data, 5)
