# -*- coding: utf-8 -*- 
# @File : utils.py
# @Time : 2024/10/13 22:10:40
# @Author :  
# @Software : Visual Studio Code
import numpy as np
import torch
import random
import os
import torch.backends.cudnn
import dgl


def build_graph(df, num_users, num_items):
    u = torch.LongTensor(df['user_id'].values)
    i = torch.LongTensor(df['item_id'].values) + num_users
    graph = dgl.graph((u, i), num_nodes=num_users + num_items)
    return graph


def set_seed(seed=2000):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
