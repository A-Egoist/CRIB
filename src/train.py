# -*- coding: utf-8 -*- 
# @File : train.py
# @Time : 2024/10/22 19:59:47
# @Author :  
# @Software : Visual Studio Code
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from loguru import logger

from .metrics import TopK


def train_and_valid_BCE(device, dataset_name, train_loader, valid_loader, valid_df, model_name, model, optimizer, epochs, lamb, graph=None, num_layers=2):
    # train
    bce_loss = nn.BCELoss()
    loss_min = 0x3f3f3f3f
    reg_loss_min = 0x3f3f3f3f
    for epoch in range(epochs):
        model.train()
        loss_sum = torch.tensor([0], dtype=torch.float32).to(device)
        reg_loss_sum = torch.tensor([0], dtype=torch.float32).to(device)
        for user_ids, item_ids, time_diffs, user_trends, item_trends, labels in tqdm(train_loader):
            user_ids, item_ids, time_diffs, user_trends, item_trends, labels = user_ids.to(device), item_ids.to(device), time_diffs.to(device), user_trends.to(device), item_trends.to(device), labels.to(device)
            
            if model_name == "MF":
                pred, reg_loss = model(user_ids, item_ids)
                loss = bce_loss(pred, labels.float()) + lamb * reg_loss
            elif model_name == "LightGCN":
                pred, reg_loss = model(user_ids, item_ids)
                loss = bce_loss(pred, labels.float()) + lamb * reg_loss
            elif model_name == "MF_CRIB":
                global_match, time_match, reg_loss = model(user_ids, item_ids, time_diffs, user_trends, item_trends)
                loss = bce_loss(global_match, labels.float()) + bce_loss(time_match, labels.float()) + lamb * reg_loss
            elif model_name == "LightGCN_CRIB":
                global_match, time_match, reg_loss = model(user_ids, item_ids, time_diffs, user_trends, item_trends)
                loss = bce_loss(global_match, labels.float()) + bce_loss(time_match, labels.float()) + lamb * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            reg_loss_sum += reg_loss.item()
            
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {np.round(loss_sum.item(), 4)}, Reg_loss: {np.round(reg_loss_sum.item(), 4)}")
        if ((loss_sum.item() < loss_min) and ((epoch + 1) % 5 == 0)) or (epoch + 1) == epochs:
            torch.save(model.state_dict(), f"./save_model/{dataset_name}-{model_name}-BCE-{lamb}.pt")
        loss_min = loss_sum.item()
        # reg_loss_min = reg_loss_sum.item()

        # valid
        # if (epoch + 1) % 5 == 0 and (epoch + 1) != epochs:
        #     model.eval()
        #     valid_pred = []
        #     valid_true = []
        #     for user_ids, item_ids, time_diffs, user_trends, item_trends, labels in tqdm(valid_loader):
        #         user_ids, item_ids, time_diffs, user_trends, item_trends = user_ids.to(device), item_ids.to(device), time_diffs.to(device), user_trends.to(device), item_trends.to(device)
        #         if model_name == "MF" or model_name == "LightGCN":
        #             pred = model.predict(user_ids, item_ids)
        #         elif model_name == "MF_CRIB" or model_name == "LightGCN_CRIB":
        #             pred = model.predict(user_ids, item_ids, time_diffs, user_trends, item_trends)
        #         pred = pred.detach().cpu()
        #         valid_pred.append(pred)
        #         valid_true.append(labels)
        #     valid_pred = torch.cat(valid_pred).squeeze()
        #     valid_df['pred'] = valid_pred.tolist()
        #     TopK(5, valid_df).run()
        #     TopK(10, valid_df).run()
