# -*- coding: utf-8 -*- 
# @File : main.py
# @Time : 2024/10/13 21:55:57
# @Author : 
# @Software : Visual Studio Code
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
import os
import argparse
import numpy as np
import pandas as pd

from src.train import train_and_valid_BCE
from src.test import test_BCE
from src.utils import set_seed, build_graph
from src.model import MF, LightGCN, MF_CRIB, LightGCN_CRIB
from src.data_processing import load_data, build_interaction_trend, BCEDataset, TestDataset


def main(args):
    # ------------ Config and hyper-parameters ------------ #
    dataset_name = args.dataset_name  # ["kuairand_pure", "kuairand_1k", "kuairec_small", "kuairec_big", "kuaisar_small", "kuaisar"]
    model_name = args.model_name  # ["MF", "LightGCN", "MF_CRIB", "LightGCN_CRIB"]
    loss_name = args.loss_name  # "BCE"
    pre_trained = args.pre_trained  # [False, True]
    embedding_dim = args.embedding_dim
    num_layers = args.num_layers  # [2, 3]
    lr = args.lr
    epochs =args.epochs
    batch_size = args.batch_size
    lamb = args.lamb  # 0.0001
    test_only = args.test_only  # [False, True]
    alpha = args.alpha
    # ------------ Config and hyper-parameters ------------ #

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_file_name = "Top-K.log"
    log_file = logger.add(os.path.join(os.getcwd(), 'log', log_file_name), encoding='utf-8')
    logger.info(f"Dataset: {dataset_name}, Model: {model_name}, Loss: {loss_name}, pretrained: {pre_trained}, embedding_dim: {embedding_dim}")
    logger.info(f"num_layers: {num_layers}, lr: {lr}, epochs: {epochs}, batch_size: {batch_size}, lamb: {lamb}, test_only: {test_only}, alpha: {alpha}, log_file: {os.path.join(os.getcwd(), 'log', log_file_name)}")

    train_df, valid_df, test_df, num_users, num_items, num_interactions = load_data(dataset_name, loss_name)
    user_trend_df, item_trend_df, train_time_diff, max_time_diff, global_trend = build_interaction_trend(dataset_name)
    train_df = pd.merge(train_df, user_trend_df, on='user_id', how='left')
    train_df = pd.merge(train_df, item_trend_df, on='item_id', how='left')
    valid_df = pd.merge(valid_df, user_trend_df, on='user_id', how='left')
    valid_df = pd.merge(valid_df, item_trend_df, on='item_id', how='left')
    test_df = pd.merge(test_df, user_trend_df, on='user_id', how='left')
    test_df = pd.merge(test_df, item_trend_df, on='item_id', how='left')
    logger.info(f"Dataset basic info: #Users = {num_users}, #Items = {num_items}, #Interactions = {num_interactions}, Max_time_diff = {max_time_diff}")
    user_trends = torch.tensor(np.array(user_trend_df['user_trend'].tolist(), dtype=np.float32), dtype=torch.float32, device=device)  # p_u
    item_trends = torch.tensor(np.array(item_trend_df['item_trend'].tolist(), dtype=np.float32), dtype=torch.float32, device=device)  # p_v
    global_trend = torch.tensor(global_trend, dtype=torch.float32, device=device)  # g

    # build graph
    if model_name == "LightGCN" or model_name == "LightGCN_CRIB":
        train_graph = build_graph(train_df, num_users, num_items).to(device)

    train_dataset = BCEDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataset = TestDataset(valid_df)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset = TestDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    if model_name == "MF":
        model = MF(num_users, num_items, embedding_dim)
    elif model_name == "LightGCN":
        model = LightGCN(num_users, num_items, embedding_dim, train_graph, num_layers)
    elif model_name == "MF_CRIB":
        model = MF_CRIB(num_users, num_items, embedding_dim, user_trends, item_trends, train_time_diff, max_time_diff, global_trend, alpha)
    elif model_name == "LightGCN_CRIB":
        model = LightGCN_CRIB(num_users, num_items, embedding_dim, user_trends, item_trends, train_time_diff, max_time_diff, global_trend, train_graph, num_layers, alpha)
    if pre_trained or test_only:
        model.load_state_dict(torch.load(f"./save_model/{dataset_name}-{model_name}-{loss_name}-{lamb}.pt", weights_only=True))
    model.to(device)

    if test_only:
        test_BCE(device, dataset_name, test_loader, test_df, model_name, model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_and_valid_BCE(device, dataset_name, train_loader, valid_loader, valid_df, model_name, model, optimizer, epochs, lamb)
        test_BCE(device, dataset_name, test_loader, test_df, model_name, model)
    logger.remove(log_file)
    with open(os.path.join(os.getcwd(), 'log', log_file_name), 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write('\n')


def parser():
    parser = argparse.ArgumentParser(description="argument parser")
    parser.add_argument('--dataset_name', type=str, default='kuairand_pure', choices=['kuairand_pure', 'kuairand_1k', 'kuairec_small', 'kuairec_big', 'kuaisar_small', 'kuaisar'])
    parser.add_argument('--model_name', type=str, default='MF_CRIB', choices=['MF', 'LightGCN', 'MF_CRIB', 'LightGCN_CRIB'])
    parser.add_argument('--loss_name', type=str, default='BCE')
    parser.add_argument('--pre_trained', type=bool, default=False, choices=[False, True])
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2, choices=[2, 3])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lamb', type=float, default=0.01)
    parser.add_argument('--test_only', type=bool, default=False, choices=[False, True])
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed()
    args = parser()
    main(args)
