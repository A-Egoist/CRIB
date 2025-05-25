import torch
from tqdm import tqdm

from .metrics import TopK


def test(device, dataset_name, test_loader, test_df, model_name, model, pred_mode='policy_2'):
    model.eval()
    test_pred = []
    test_true = []
    with torch.no_grad():
        for user_ids, item_ids, time_diffs, user_trends, item_trends, labels in tqdm(test_loader):
            user_ids, item_ids, time_diffs, user_trends, item_trends = user_ids.to(device), item_ids.to(device), time_diffs.to(device), user_trends.to(device), item_trends.to(device)
            if model_name == "MF-Base":
                pred = model.predict(user_ids, item_ids)
            elif model_name == "LightGCN-Base":
                pred = model.predict(user_ids, item_ids)
            elif model_name == "MF-CRIB":
                pred = model.predict(user_ids, item_ids, time_diffs, user_trends, item_trends)
            elif model_name == "LightGCN-CRIB":
                pred = model.predict(user_ids, item_ids, time_diffs, user_trends, item_trends)
            elif model_name == "MF-LDRI" or model_name == "LightGCN-LDRI":
                pred = model.predict(user_ids, item_ids, time_diffs, pred_mode=pred_mode)
            elif model_name == "MF-TaFR" or model_name == "LightGCN-TaFR":
                pred = model.predict(user_ids, item_ids, time_diffs)
            elif model_name == "MF-DCR-MoE" or model_name == "LightGCN-DCR-MoE":
                pred = model.predict(user_ids, item_ids, time_diffs)
            elif model_name in ['MF-CRIB_without_USE', 'LightGCN-CRIB_without_USE', 'MF-CRIB_without_VT', 'LightGCN-CRIB_without_VT']:
                pred = model.predict(user_ids, item_ids, time_diffs, user_trends, item_trends)
            pred = pred.detach().cpu()
            test_pred.append(pred)
            test_true.append(labels)
    test_pred = torch.cat(test_pred).squeeze()
    test_df['pred'] = test_pred.tolist()
    TopK(5, test_df).run()
    TopK(10, test_df).run()
    TopK(20, test_df).run()
    TopK(50, test_df).run()
    TopK(100, test_df).run()
