# -*- coding: utf-8 -*- 
# @File : test.py
# @Time : 2024/10/22 20:35:23
# @Author :  
# @Software : Visual Studio Code
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .metrics import TopK


def test_BCE(device, dataset_name, test_loader, test_df, model_name, model):
    model.eval()
    test_pred = []
    test_true = []
    with torch.no_grad():
        for user_ids, item_ids, time_diffs, user_trends, item_trends, labels in tqdm(test_loader):
            user_ids, item_ids, time_diffs, user_trends, item_trends = user_ids.to(device), item_ids.to(device), time_diffs.to(device), user_trends.to(device), item_trends.to(device)
            if model_name == "MF":
                pred = model.predict(user_ids, item_ids)
            elif model_name == "LightGCN":
                pred = model.predict(user_ids, item_ids)
            elif model_name == "MF_CRIB":
                pred = model.predict(user_ids, item_ids, time_diffs, user_trends, item_trends)
            elif model_name == "LightGCN_CRIB":
                pred = model.predict(user_ids, item_ids, time_diffs, user_trends, item_trends)
            pred = pred.detach().cpu()
            test_pred.append(pred)
            test_true.append(labels)
    test_pred = torch.cat(test_pred).squeeze()
    test_df['pred'] = test_pred.tolist()
    TopK(5, test_df).run()
    TopK(10, test_df).run()
    TopK(20, test_df).run()
