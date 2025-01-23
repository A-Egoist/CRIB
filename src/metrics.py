# -*- coding: utf-8 -*- 
# @File : evaluate.py
# @Time : 2024/10/14 15:45:18
# @Author :  
# @Software : Visual Studio Code
import numpy as np
import pandas as pd
from loguru import logger


class TopK:
    def __init__(self, topk, df_result):
        self.topk = topk
        self.df = df_result.copy()
    
    def get_result(self, R, T):
        res = {}
        decimals = 4
        res[f"Precision@{self.topk}"] = np.round(self.get_Precision(R, T), decimals)
        res[f"Recall@{self.topk}"] = np.round(self.get_Recall(R, T), decimals)
        res[f"HR@{self.topk}"] = np.round(self.get_HR(R, T), decimals)
        res[f"MRR@{self.topk}"] = np.round(self.get_MRR(R, T), decimals)
        res[f"MAP@{self.topk}"] = np.round(self.get_MAP(R, T), decimals)
        res[f"NDCG@{self.topk}"] = np.round(self.get_NDCG(R, T), decimals)
        return res

    def run(self):
        df = self.df[['user_id', 'item_id', 'label', 'pred']]
        
        # get ground truth
        truth_df = df[df['label'] == 1].copy()
        truth_df = truth_df.groupby('user_id')['item_id'].apply(list).reset_index()
        truth_df.rename(columns={'item_id': 'truth_list'}, inplace=True)

        # get pred
        pred_df = df.groupby('user_id').apply(lambda x: x.sort_values('pred', ascending=False)['item_id'].tolist()).reset_index(name='pred_list')

        # [user_id, truth_list, pred_list]
        df = pd.merge(truth_df, pred_df, on='user_id', how='left')
        df = df[['user_id', 'truth_list', 'pred_list']]
        df['pred_list'] = df['pred_list'].apply(lambda x: x[:self.topk])

        T = df['truth_list'].tolist()  # 2d, [[1, 2, 3, 5], [1, 3], ..., [5, 2, 3]]
        R = df['pred_list'].tolist()  # 2d, [[2, 3, 4], [1, 2, 5], ..., [4, 1, 6]]

        res = self.get_result(R, T)
        logger.info('Precision@{:d} {:.4f} | Recall@{:d} {:.4f} | HR@{:d} {:.4f} | MRR@{:d} {:.4f} | MAP@{:d} {:.4f} | NDCG@{:d} {:.4f}'
                    .format(self.topk, res['Precision@' + str(self.topk)],
                            self.topk, res['Recall@' + str(self.topk)],
                            self.topk, res['HR@' + str(self.topk)],
                            self.topk, res['MRR@' + str(self.topk)],
                            self.topk, res['MAP@' + str(self.topk)],
                            self.topk, res['NDCG@' + str(self.topk)]))
        
    def get_Precision(self, R, T):
        assert len(R) == len(T)
        res = 0
        for i in range(len(R)):
            res += len(set(R[i]) & set(T[i])) / len(R[i])
        return res / len(R)

    def get_Recall(self, R, T):
        assert len(R) == len(T)
        res = 0
        for i in range(len(R)):
            if len(T[i]) > 0:
                res += len(set(R[i]) & set(T[i])) / len(T[i])
        return res / len(R)

    def get_HR(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            if len(set(R[i]) & set(T[i])) > 0:
                up += 1
        return up / down

    def get_MRR(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            index = -1
            for j in range(len(R[i])):
                if R[i][j] in T[i]:
                    index = R[i].index(R[i][j])
                    break
            if index != -1:
                up += 1 / (index + 1)
        return up / down
    
    def get_MAP(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        for i in range(len(R)):
            temp = 0
            hit = 0
            for j in range(len(R[i])):
                if R[i][j] in T[i]:
                    hit += 1
                    temp += hit / (j + 1)
            if hit > 0:
                up += temp / len(T[i])
        return up / down

    def get_NDCG(self, R, T):
        assert len(R) == len(T)
        up = 0
        down = len(R)
        def dcg(hits):
            res = 0
            for i in range(len(hits)):
                res += (hits[i] / np.log2(i + 2))
            return res
        for i in range(len(R)):
            hits = []
            for j in range(len(R[i])):
                if R[i][j] in T[i]:
                    hits += [1.0]
                else:
                    hits += [0.0]
            if sum(hits) > 0:
                up += (dcg(hits) / (dcg([1.0 for i in range(len(T[i]))]) + 1))  # from wiki
        return up / down
