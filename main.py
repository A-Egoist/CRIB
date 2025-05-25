import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time
from loguru import logger
import os
import argparse
import numpy as np
import pandas as pd
import time

from src.train import train_and_valid
from src.test import test
from src.utils import set_seed, build_graph
from src.model import *
from src.data_processing import load_data, build_interaction_trend, BCEDataset, TestDataset


def main(args):
    # ------------ Config and hyper-parameters ------------ #
    dataset_name = args.dataset_name  # ["kuairand_pure", "kuairand_1k", "kuairec_small", "kuairec_big", "kuaisar_small", "kuaisar"]
    model_name = args.model_name  # ["MF", "LightGCN", "DRIMF", "DRIGNN"]
    # loss_name = args.loss_name  # ["BPR", "BCE"]
    pre_trained = args.pre_trained  # [False, True]
    embedding_dim = args.embedding_dim  # [16, 32, 64, 128], 128 for MF-based model, 32 for LightGCN-based model
    num_layers = args.num_layers  # [2, 3]
    lr = args.lr  # [0.0001, 0.001]
    epochs =args.epochs  # [20, 40, 60]
    batch_size = args.batch_size  # [1024, 2048, 4096, 8192]
    lamb = args.lamb  # 0.01
    test_only = args.test_only  # [False, False]
    CRIB_alpha = args.CRIB_alpha
    LDRI_alpha = args.LDRI_alpha
    LDRI_beta = args.LDRI_beta
    # ------------ Config and hyper-parameters ------------ #

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if not os.path.exists(f'./logs/{dataset_name}'):
        os.mkdir(f'./logs/{dataset_name}')
    log_file = logger.add(f'./logs/{dataset_name}/{model_name}-{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.log', encoding='utf-8')
    logger.info(f"Dataset: {dataset_name}, Model: {model_name}, Loss: BCE, pretrained: {pre_trained}, embedding_dim: {embedding_dim}")
    logger.info(f"num_layers: {num_layers}, lr: {lr}, epochs: {epochs}, batch_size: {batch_size}, lamb: {lamb}, test_only: {test_only}, alpha: {CRIB_alpha}")

    train_df, valid_df, test_df, num_users, num_items, num_interactions = load_data(dataset_name)
    user_trend_df, item_trend_df, train_time_diff, max_time_diff, global_trend = build_interaction_trend(dataset_name)
    train_df = pd.merge(train_df, user_trend_df, on='user_id', how='left')
    train_df = pd.merge(train_df, item_trend_df, on='item_id', how='left')
    valid_df = pd.merge(valid_df, user_trend_df, on='user_id', how='left')
    valid_df = pd.merge(valid_df, item_trend_df, on='item_id', how='left')
    test_df = pd.merge(test_df, user_trend_df, on='user_id', how='left')
    test_df = pd.merge(test_df, item_trend_df, on='item_id', how='left')
    logger.info(f"Dataset basic info: #Users = {num_users}, #Items = {num_items}, #Interactions = {num_interactions}, Max_time_diff = {max_time_diff}")
    user_trends = torch.tensor(np.array(user_trend_df['user_trend'].tolist(), dtype=np.float32), dtype=torch.float32, device=device)
    item_trends = torch.tensor(np.array(item_trend_df['item_trend'].tolist(), dtype=np.float32), dtype=torch.float32, device=device)
    global_trend = torch.tensor(global_trend, dtype=torch.float32, device=device)

    # build graph
    train_graph = None
    if model_name.split('-')[0] == "LightGCN":
        train_graph = build_graph(train_df, num_users, num_items).to(device)

    train_dataset = BCEDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataset = TestDataset(valid_df)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset = TestDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    if model_name == "MF-Base":
        model = MF(num_users, num_items, embedding_dim)
    elif model_name == "LightGCN-Base":
        model = LightGCN(num_users, num_items, embedding_dim, train_graph, num_layers)
    elif model_name == "MF-CRIB":
        model = DRIMFBCE(num_users, num_items, embedding_dim, user_trends, item_trends, train_time_diff, max_time_diff, global_trend, CRIB_alpha)
    elif model_name == "LightGCN-CRIB":
        model = DRIGNNBCE(num_users, num_items, embedding_dim, user_trends, item_trends, train_time_diff, max_time_diff, global_trend, train_graph, num_layers, CRIB_alpha)
    if pre_trained or test_only:
        model.load_state_dict(torch.load(f"./save_model/{dataset_name}/{model_name}_{args.embedding_dim}_{args.lr}_{args.lamb}_final.pt", weights_only=True))
    model.to(device)

    if not test_only:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model = train_and_valid(args, device, dataset_name, train_loader, valid_loader, valid_df, model_name, model, optimizer, epochs, lamb, LDRI_alpha=LDRI_alpha, LDRI_beta=LDRI_beta, CRIB_alpha=CRIB_alpha)

    if model_name == 'MF-LDRI' or model_name == 'LightGCN-LDRI':
        test(device, dataset_name, test_loader, test_df, model_name, model, pred_mode='policy_1')
    test(device, dataset_name, test_loader, test_df, model_name, model, pred_mode='policy_2')
    logger.remove(log_file)


def parser():
    parser = argparse.ArgumentParser(description="argument parser")
    parser.add_argument('--dataset_name', type=str, default='kuairand_pure', choices=['kuairand_pure', 'kuairand_1k', 'kuairec_big', 'kuaisar_small', 'kuaisar'])
    parser.add_argument('--model_name', type=str, default='DRIMF', choices=['MF-Base', 'LightGCN-Base', 'MF-CRIB', 'LightGCN-CRIB'])
    parser.add_argument('--pre_trained', type=bool, default=False, choices=[False, True])
    parser.add_argument('--embedding_dim', type=int, default=16, choices=[16, 32, 64, 128], help='128 for MF-based model, 32 for LightGCN-based model')
    parser.add_argument('--num_layers', type=int, default=2, choices=[2, 3])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048, choices=[1024, 2048, 4096])
    parser.add_argument('--lamb', type=float, default=0.01)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--test_only', type=bool, default=False, choices=[False, True])
    parser.add_argument('--CRIB_alpha', type=float, default=0.5)
    parser.add_argument('--LDRI_alpha', type=float, default=0.6)
    parser.add_argument('--LDRI_beta', type=float, default=0.5)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_seed()
    args = parser()
    main(args)
