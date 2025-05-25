import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from loguru import logger
import os

from .metrics import TopK


def train_and_valid(args, device, dataset_name, train_loader, valid_loader, valid_df, model_name, model, optimizer, epochs, lamb, graph=None, num_layers=2, LDRI_alpha=0.6, LDRI_beta=0.5, CRIB_alpha=0.5):
    # train
    bce_loss = nn.BCELoss()

    best_valid_res_value = {'Recall@10': -1, 'NDCG@10': -1}
    best_valid_res_epoch = {'Recall@10': -1, 'NDCG@10': -1}
    bad_count = 0

    for epoch in range(epochs):
        model.train()
        flag_update_metric = 0  # used in valid
        loss_sum = torch.tensor([0], dtype=torch.float32).to(device)
        reg_loss_sum = torch.tensor([0], dtype=torch.float32).to(device)
        # user_id, item_id, time_diff, user_trend, item_trend, label
        for user_ids, item_ids, time_diffs, user_trends, item_trends, labels in tqdm(train_loader):
            user_ids, item_ids, time_diffs, user_trends, item_trends, labels = user_ids.to(device), item_ids.to(device), time_diffs.to(device), user_trends.to(device), item_trends.to(device), labels.to(device)
            
            if model_name == "MF-Base":
                pred, reg_loss = model(user_ids, item_ids)
                loss = bce_loss(torch.sigmoid(pred), labels.float()) + lamb * reg_loss
            elif model_name == "LightGCN-Base":
                pred, reg_loss = model(user_ids, item_ids)
                loss = bce_loss(torch.sigmoid(pred), labels.float()) + lamb * reg_loss
            elif model_name == "MF-CRIB":
                global_match, time_match, reg_loss = model(user_ids, item_ids, time_diffs, user_trends, item_trends)
                loss = bce_loss(global_match, labels.float()) + bce_loss(time_match, labels.float()) + lamb * reg_loss
            elif model_name == "LightGCN-CRIB":
                global_match, time_match, reg_loss = model(user_ids, item_ids, time_diffs, user_trends, item_trends)
                loss = bce_loss(global_match, labels.float()) + bce_loss(time_match, labels.float()) + lamb * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            reg_loss_sum += reg_loss.item()
            
        logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {np.round(loss_sum.item(), 4)}, Reg_loss: {np.round(reg_loss_sum.item(), 4)}")

        # valid
        if epoch != 0 and epoch % args.eval_interval == 0:
            model.eval()
            valid_pred = []
            valid_true = []
            for user_ids, item_ids, time_diffs, user_trends, item_trends, labels in tqdm(valid_loader):
                user_ids, item_ids, time_diffs, user_trends, item_trends = user_ids.to(device), item_ids.to(device), time_diffs.to(device), user_trends.to(device), item_trends.to(device)
                if model_name == "MF-Base":
                    pred = model.predict(user_ids, item_ids)
                elif model_name == "LightGCN-Base":
                    pred = model.predict(user_ids, item_ids)
                elif model_name == "MF-CRIB":
                    pred = model.predict(user_ids, item_ids, time_diffs, user_trends, item_trends)
                elif model_name == "LightGCN-CRIB":
                    pred = model.predict(user_ids, item_ids, time_diffs, user_trends, item_trends)
                pred = pred.detach().cpu()
                valid_pred.append(pred)
                valid_true.append(labels)
            valid_pred = torch.cat(valid_pred).squeeze()
            valid_df['pred'] = valid_pred.tolist()

            valid_res = TopK(10, valid_df).run(False)
            for key in best_valid_res_value.keys():
                if valid_res[key] > best_valid_res_value[key]:
                    flag_update_metric = 1
                    bad_count = 0
                    best_valid_res_value[key] = valid_res[key]
                    best_valid_res_epoch[key] = epoch
                    if not os.path.exists(f'./save_model/{dataset_name}'):
                        os.mkdir(f'./save_model/{dataset_name}')
                    torch.save(model.state_dict(), f"./save_model/{dataset_name}/{model_name}_{args.embedding_dim}_{args.lr}_{args.lamb}_Epoch{epoch}.pt")
            
            if flag_update_metric == 0:
                bad_count += 1
            
            print(f'flag_update_metric={flag_update_metric}')
            print(f'bad_count={bad_count}')

            if bad_count >= args.patience:
                logger.info('Early stopped.')
                break

    if not os.path.exists(f'./save_model/{dataset_name}'):
        os.mkdir(f'./save_model/{dataset_name}')
    torch.save(model.state_dict(), f"./save_model/{dataset_name}/{model_name}_{args.embedding_dim}_{args.lr}_{args.lamb}_final.pt")
    return model
