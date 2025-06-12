import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import date


def load_data(dataset_name):
    train_df = pd.read_csv(os.path.join(os.getcwd(), 'data', dataset_name + '-train.csv'))
    train_df = train_df.drop_duplicates()
    valid_df = pd.read_csv(os.path.join(os.getcwd(), 'data', dataset_name + '-valid.csv'))
    valid_df = valid_df.drop_duplicates()
    test_df = pd.read_csv(os.path.join(os.getcwd(), 'data', dataset_name + '-test.csv'))
    test_df = test_df.drop_duplicates()
    num_users = max(train_df['user_id'].max(), valid_df['user_id'].max(), test_df['user_id'].max()) + 1  # user id from 0 to max
    num_items = max(train_df['item_id'].max(), valid_df['item_id'].max(), test_df['item_id'].max()) + 1  # item id from 0 to max
    num_interactions = train_df.shape[0] + valid_df.shape[0] + test_df.shape[0]

    return train_df, valid_df, test_df, num_users, num_items, num_interactions


class BCEDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.user_id = torch.tensor(np.array(df.iloc[:, 0].tolist()), dtype=torch.long)  # user id
        self.item_id = torch.tensor(np.array(df.iloc[:, 1].tolist()), dtype=torch.long)  # item id
        self.time_diff = torch.tensor(np.array(df.iloc[:, 2].tolist()), dtype=torch.long)  # time_diff
        self.user_trend = torch.tensor(np.array(df.iloc[:, 4].tolist()), dtype=torch.float32)  # user intercation trend
        self.item_trend = torch.tensor(np.array(df.iloc[:, 5].tolist()), dtype=torch.float32)  # item interaction trend
        self.label = torch.tensor(np.array(df.iloc[:, 3].tolist()), dtype=torch.float32)  # interaction label
        assert self.user_id.shape[0] == self.item_id.shape[0] == self.time_diff.shape[0] == self.user_trend.shape[0] == self.item_trend.shape[0] == self.label.shape[0]

    def __getitem__(self, index):
        return self.user_id[index], self.item_id[index], self.time_diff[index], self.user_trend[index], self.item_trend[index], self.label[index]
    
    def __len__(self):
        return self.label.shape[0]


class TestDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.user_id = torch.tensor(np.array(df.iloc[:, 0].tolist()), dtype=torch.long)  # user id
        self.item_id = torch.tensor(np.array(df.iloc[:, 1].tolist()), dtype=torch.long)  # item id
        self.time_diff = torch.tensor(np.array(df.iloc[:, 2].tolist()), dtype=torch.long)  # time_diff
        self.user_trend = torch.tensor(np.array(df.iloc[:, 4].tolist()), dtype=torch.float32)  # user intercation trend
        self.item_trend = torch.tensor(np.array(df.iloc[:, 5].tolist()), dtype=torch.float32)  # item interaction trend
        self.label = torch.tensor(np.array(df.iloc[:, 3].tolist()), dtype=torch.float32)  # interaction label
        assert self.user_id.shape[0] == self.item_id.shape[0] == self.time_diff.shape[0] == self.user_trend.shape[0] == self.item_trend.shape[0] == self.label.shape[0]

    def __getitem__(self, index):
        return self.user_id[index], self.item_id[index], self.time_diff[index], self.user_trend[index], self.item_trend[index], self.label[index]
    
    def __len__(self):
        return self.user_id.shape[0]


def KuaiRand_preprocess(dataset):
    if dataset == "kuairand_pure":
        log_sd1 = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiRand-Pure', 'data', 'log_standard_4_08_to_4_21_pure.csv'))
        log_sd2 = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiRand-Pure', 'data', 'log_standard_4_22_to_5_08_pure.csv'))
        video_feat = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiRand-Pure', 'data', 'video_features_basic_pure.csv'))
        # user_info = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiRand-Pure', 'data', 'user_features_pure.csv'))
    elif dataset == "kuairand_1k":
        log_sd1 = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiRand-1K', 'data', 'log_standard_4_08_to_4_21_1k.csv'))
        log_sd2 = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiRand-1K', 'data', 'log_standard_4_22_to_5_08_1k.csv'))
        video_feat = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiRand-1K', 'data', 'video_features_basic_1k.csv'))
        # user_info = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiRand-1K', 'data', 'user_features_1k.csv'))
    
    log_df = pd.concat([log_sd1, log_sd2], axis=0)
    video_feat = video_feat[['video_id', 'upload_dt']].drop_duplicates()
    df = pd.merge(left=log_df, right=video_feat, how='left', on=['video_id']).dropna(how='any')
    
    if dataset == 'kuairand_pure':
        df['label'] = ((df['is_click'] | df['is_like'] | df['is_comment'] | df['is_follow'] | df['is_forward'] | df['is_profile_enter']) & (df['is_hate'] == 0)).astype('int32')
    elif dataset == 'kuairand_1k':
        df['watch_ratio'] = df['play_time_ms'] / df['duration_ms']
        df['label'] = ((df['is_click'] | (df['watch_ratio'] > 0.8) | df['is_like'] | df['is_comment'] | df['is_follow'] | df['is_forward'] | df['is_profile_enter']) & (df['is_hate'] == 0)).astype('int32')
    
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date
    df['upload_dt'] = pd.to_datetime(df['upload_dt'], format='%Y-%m-%d').dt.date
    df = df[['user_id', 'video_id', 'date', 'upload_dt', 'label']]
    # rename columns
    df.columns = ['user_id', 'item_id', 'date', 'upload_dt', 'label']
    print(df.head(10))
    df.to_csv(os.path.join(os.path.join(os.getcwd(), 'data', dataset + '.csv')), index=False)

    train_df = df[(df['date'] >= date(2022, 4, 8)) & (df['date'] <= date(2022, 4, 28))].copy()
    valid_df = df[(df['date'] >= date(2022, 4, 29)) & (df['date'] <= date(2022, 5, 1))].copy()
    test_df = df[(df['date'] >= date(2022, 5, 2)) & (df['date'] <= date(2022, 5, 8))].copy()

    train_df['time_diff'] = (train_df['date'] - train_df['upload_dt']).dt.days
    valid_df['time_diff'] = (valid_df['date'] - valid_df['upload_dt']).dt.days
    test_df['time_diff'] = test_df['upload_dt'].apply(lambda x: (date(2022, 5, 1) - x).days if x <= date(2022, 5, 1) else 0)
    
    # limit the scope of time diff
    train_df['time_diff'] = train_df['time_diff'].clip(0, (date(2022, 4, 28) - date(2022, 4, 8)).days)
    valid_df['time_diff'] = valid_df['time_diff'].clip(0, (date(2022, 5, 1) - date(2022, 4, 8)).days)
    test_df['time_diff'] = test_df['time_diff'].clip(0, (date(2022, 5, 8) - date(2022, 4, 8)).days)

    train_df = train_df[['user_id', 'item_id', 'time_diff', 'label']]
    valid_df = valid_df[['user_id', 'item_id', 'time_diff', 'label']]
    test_df = test_df[['user_id', 'item_id', 'time_diff', 'label']]
    train_df.to_csv(os.path.join(os.getcwd(), 'data', dataset + '-train.csv'), index=False)
    valid_df.to_csv(os.path.join(os.getcwd(), 'data', dataset + '-valid.csv'), index=False)
    test_df.to_csv(os.path.join(os.getcwd(), 'data', dataset + '-test.csv'), index=False)


def KuaiRec_preprocess(dataset):
    log_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiRec', 'data', 'big_matrix.csv'))
    video_feat = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiRec', 'data', 'item_daily_features.csv'))
    video_feat = video_feat[['video_id', 'upload_dt']].drop_duplicates()
    df = pd.merge(left=log_df, right=video_feat, how='left', on=['video_id']).dropna(how='any')
    
    df['label'] = (df['watch_ratio'] >= 1.0).astype('int32')

    df['date'] = pd.to_datetime(df['time']).dt.date
    df['upload_dt'] = pd.to_datetime(df['upload_dt'], format='%Y-%m-%d').dt.date
    df = df[['user_id', 'video_id', 'date', 'upload_dt', 'label']]
    df.columns = ['user_id', 'item_id', 'date', 'upload_dt', 'label']
    print(df.head(10))
    df.to_csv(os.path.join(os.path.join(os.getcwd(), 'data', dataset + '.csv')), index=False)

    train_df = df[(df['date'] >= date(2020, 7, 7)) & (df['date'] <= date(2020, 8, 14))].copy()
    valid_df = df[(df['date'] >= date(2020, 8, 15)) & (df['date'] <= date(2020, 8, 21))].copy()
    test_df = df[(df['date'] >= date(2020, 8, 22)) & (df['date'] <= date(2020, 9, 5))].copy()

    train_df['time_diff'] = (train_df['date'] - train_df['upload_dt']).dt.days
    valid_df['time_diff'] = (valid_df['date'] - valid_df['upload_dt']).dt.days
    test_df['time_diff'] = test_df['upload_dt'].apply(lambda x: (date(2020, 8, 21) - x).days if x <= date(2020, 8, 21) else 0)

    # TODO
    # limit the scope of time diff
    train_df['time_diff'] = train_df['time_diff'].clip(0, (date(2020, 8, 14) - date(2020, 7, 7)).days)
    valid_df['time_diff'] = valid_df['time_diff'].clip(0, (date(2020, 8, 21) - date(2020, 7, 7)).days)
    test_df['time_diff'] = test_df['time_diff'].clip(0, (date(2020, 9, 5) - date(2020, 7, 7)).days)

    train_df = train_df[['user_id', 'item_id', 'time_diff', 'label']]
    valid_df = valid_df[['user_id', 'item_id', 'time_diff', 'label']]
    test_df = test_df[['user_id', 'item_id', 'time_diff', 'label']]
    train_df.to_csv(os.path.join(os.getcwd(), 'data', dataset + '-train.csv'), index=False)
    valid_df.to_csv(os.path.join(os.getcwd(), 'data', dataset + '-valid.csv'), index=False)
    test_df.to_csv(os.path.join(os.getcwd(), 'data', dataset + '-test.csv'), index=False)


def KuaiSAR_preprocess(dataset):
    # TODO
    if dataset == "kuaisar_small":
        log_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiSAR-small', 'rec_inter.csv'))
        item_feat = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiSAR-small', 'item_features.csv'))
    elif dataset == "kuaisar":
        log_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiSAR', 'rec_inter.csv'))
        item_feat = pd.read_csv(os.path.join(os.getcwd(), 'data', 'KuaiSAR', 'item_features.csv'))
    # print(log_df.columns)
    # print(item_feat.columns)
    item_feat['upload_dt'] = pd.to_datetime(item_feat['upload_time'], format='%Y-%m-%d', errors='coerce').dt.date
    item_feat.dropna(how='any')
    item_feat = item_feat[['item_id', 'upload_dt']].drop_duplicates()
    df = pd.merge(left=log_df, right=item_feat, how='left', on=['item_id']).dropna(how='any')

    df['label'] = df['click'].astype('int32')

    df['date'] = pd.to_datetime(df['time']).dt.date
    df = df[['user_id', 'item_id', 'date', 'upload_dt', 'label']]
    print(df.head(10))
    df.to_csv(os.path.join(os.path.join(os.getcwd(), 'data', dataset + '.csv')), index=False)

    if dataset == 'kuaisar_small':
        train_df = df[(df['date'] >= date(2023, 5, 22)) & (df['date'] <= date(2023, 5, 27))].copy()
        valid_df = df[(df['date'] >= date(2023, 5, 28)) & (df['date'] <= date(2023, 5, 28))].copy()
        test_df = df[(df['date'] >= date(2023, 5, 29)) & (df['date'] <= date(2023, 5, 31))].copy()
    elif dataset == 'kuaisar':
        train_df = df[(df['date'] >= date(2023, 5, 22)) & (df['date'] <= date(2023, 6, 2))].copy()
        valid_df = df[(df['date'] >= date(2023, 6, 3)) & (df['date'] <= date(2023, 6, 4))].copy()
        test_df = df[(df['date'] >= date(2023, 6, 5)) & (df['date'] <= date(2023, 6, 10))].copy()

    train_df['time_diff'] = (train_df['date'] - train_df['upload_dt']).dt.days
    valid_df['time_diff'] = (valid_df['date'] - valid_df['upload_dt']).dt.days
    if dataset == 'kuaisar_small':
        test_df['time_diff'] = test_df['upload_dt'].apply(lambda x: (date(2023, 5, 28) - x).days if x <= date(2023, 5, 28) else 0)
    elif dataset == 'kuaisar':
        test_df['time_diff'] = test_df['upload_dt'].apply(lambda x: (date(2023, 6, 4) - x).days if x <= date(2023, 6, 4) else 0)

    # limit the scope of time diff
    if dataset == "kuaisar_small":
        train_df['time_diff'] = train_df['time_diff'].clip(0, (date(2023, 5, 27) - date(2023, 5, 22)).days)
        valid_df['time_diff'] = valid_df['time_diff'].clip(0, (date(2023, 5, 28) - date(2023, 5, 22)).days)
        test_df['time_diff'] = test_df['time_diff'].clip(0, (date(2023, 5, 31) - date(2023, 5, 22)).days)
    elif dataset == "kuaisar":
        train_df['time_diff'] = train_df['time_diff'].clip(0, (date(2023, 6, 2) - date(2023, 5, 22)).days)
        valid_df['time_diff'] = valid_df['time_diff'].clip(0, (date(2023, 6, 4) - date(2023, 5, 22)).days)
        test_df['time_diff'] = test_df['time_diff'].clip(0, (date(2023, 6, 10) - date(2023, 5, 22)).days)

    train_df = train_df[['user_id', 'item_id', 'time_diff', 'label']]
    valid_df = valid_df[['user_id', 'item_id', 'time_diff', 'label']]
    test_df = test_df[['user_id', 'item_id', 'time_diff', 'label']]
    train_df.to_csv(os.path.join(os.getcwd(), 'data', dataset + '-train.csv'), index=False)
    valid_df.to_csv(os.path.join(os.getcwd(), 'data', dataset + '-valid.csv'), index=False)
    test_df.to_csv(os.path.join(os.getcwd(), 'data', dataset + '-test.csv'), index=False)


def build_interaction_trend(dataset):
    train_df = pd.read_csv(os.path.join(os.getcwd(), 'data', dataset + '-train.csv'))
    valid_df = pd.read_csv(os.path.join(os.getcwd(), 'data', dataset + '-valid.csv'))
    test_df = pd.read_csv(os.path.join(os.getcwd(), 'data', dataset + '-test.csv'))
    num_users = max(train_df['user_id'].max(), valid_df['user_id'].max(), test_df['user_id'].max()) + 1  # user id from 0 to max
    num_items = max(train_df['item_id'].max(), valid_df['item_id'].max(), test_df['item_id'].max()) + 1  # item id from 0 to max

    if dataset == 'kuairand_pure' or dataset == 'kuairand_1k':
        train_time_diff = (date(2022, 4, 28) - date(2022, 4, 8)).days
        max_time_diff = (date(2022, 5, 8) - date(2022, 4, 8)).days
    elif dataset == 'kuairec_big':
        train_time_diff = (date(2020, 8, 14) - date(2020, 7, 7)).days
        max_time_diff = (date(2020, 9, 5) - date(2020, 7, 7)).days
    elif dataset == 'kuaisar_small':
        train_time_diff = (date(2023, 5, 27) - date(2023, 5, 22)).days
        max_time_diff = (date(2023, 5, 31) - date(2023, 5, 22)).days
    elif dataset == 'kuaisar':
        train_time_diff = (date(2023, 6, 2) - date(2023, 5, 22)).days
        max_time_diff = (date(2023, 6, 10) - date(2023, 5, 22)).days

    # user trend
    user_count = train_df.groupby(['user_id', 'time_diff']).size().unstack(fill_value=0)
    user_count = user_count.reindex(columns=range(train_time_diff), fill_value=0)  # extend
    user_sum = user_count.sum(axis=1)
    user_sum[user_sum < 1e-6] = 1e-6
    user_trend = user_count.values / user_sum.values[:, None]
    user_trend[np.isnan(user_trend)] = 0  # replace NaN to 0
    user_trend = np.round(user_trend, 4)
    global_user_trend = user_trend.mean(axis=0)
    user_trend_df = pd.DataFrame(user_trend, index=user_count.index, columns=range(train_time_diff)).reset_index()
    user_trend_df.columns = ['user_id'] + [f'time_diff_{i}' for i in range(train_time_diff)]
    user_trend_df['user_trend'] = user_trend_df[[f'time_diff_{i}' for i in range(train_time_diff)]].values.tolist()
    user_trend_df = user_trend_df[['user_id', 'user_trend']]

    # item trend
    item_count = train_df.groupby(['item_id', 'time_diff']).size().unstack(fill_value=0)
    item_count = item_count.reindex(columns=range(train_time_diff), fill_value=0)  # extend
    item_sum = item_count.sum(axis=1)
    item_sum[item_sum < 1e-6] = 1e-6
    item_trend = item_count.values / item_sum.values[:, None]
    item_trend[np.isnan(item_trend)] = 0  # replace NaN to 0
    item_trend = np.round(item_trend, 4)
    global_item_trend = item_trend.mean(axis=0)
    item_trend_df = pd.DataFrame(item_trend, index=item_count.index, columns=range(train_time_diff)).reset_index()
    item_trend_df.columns = ['item_id'] + [f'time_diff_{i}' for i in range(train_time_diff)]
    item_trend_df['item_trend'] = item_trend_df[[f'time_diff_{i}' for i in range(train_time_diff)]].values.tolist()
    item_trend_df = item_trend_df[['item_id', 'item_trend']]

    # cold start user
    noise_level = 0.01
    all_user_ids = pd.DataFrame({'user_id': list(range(num_users))})
    user_trend_df = pd.merge(all_user_ids, user_trend_df, on='user_id', how='left')
    user_trend_df['user_trend'] = user_trend_df['user_trend'].apply(
        lambda x: x if isinstance(x, list) else np.round((global_user_trend + np.random.normal(0, noise_level, train_time_diff)).tolist(), 4)
    )

    # cold start item
    all_item_ids = pd.DataFrame({'item_id': list(range(num_items))})
    item_trend_df = pd.merge(all_item_ids, item_trend_df, on='item_id', how='left')
    item_trend_df['item_trend'] = item_trend_df['item_trend'].apply(
        lambda x: x if isinstance(x, list) else np.round((global_item_trend + np.random.normal(0, noise_level, train_time_diff)).tolist(), 4)
    )

    # global trend
    global_count = train_df.groupby('time_diff').size()
    num_interactions = global_count.sum()
    global_trend = global_count.values / num_interactions
    global_trend[np.isnan(global_trend)] = 0  # replace NaN to 0
    global_trend = np.round(global_trend, 4)

    return user_trend_df, item_trend_df, train_time_diff, max_time_diff, global_trend


if __name__ == "__main__":
    dataset = "kuairand_pure"  # ['kuairand_pure', 'kuairand_1k', 'kuairec_big', 'kuaisar_small', 'kuaisar']
    if dataset == "kuairand_pure" or dataset == "kuairand_1k":
        KuaiRand_preprocess(dataset)
    elif dataset == "kuairec_big":
        KuaiRec_preprocess(dataset)
    elif dataset == "kuaisar_small" or dataset == "kuaisar":
        KuaiSAR_preprocess(dataset)
