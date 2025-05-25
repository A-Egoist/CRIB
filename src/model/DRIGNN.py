import torch
from torch import nn
import dgl


class DRIGNNBCE(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, user_trends, item_trends, train_time_diff, max_time_diff, global_trend, graph, num_layers, alpha=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_trends = user_trends
        self.item_trends = item_trends
        self.train_time_diff = train_time_diff
        self.max_time_diff = max_time_diff
        self.global_trend = global_trend
        self.graph = graph
        self.num_layers = num_layers
        self.alpha = alpha

        # user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # user and item time embedding
        self.user_time_embedding = nn.Embedding(num_users, max_time_diff)
        self.item_time_embedding = nn.Embedding(num_items, max_time_diff)

        # MLP (train_time_diff) -> (max_time_diff - train_time_diff)
        self.mlp = nn.Sequential(nn.Linear(train_time_diff, 32),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(32, max_time_diff - train_time_diff))

        # initialization
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.user_time_embedding.weight, std=0.01)
        nn.init.normal_(self.item_time_embedding.weight, std=0.01)
        self.global_time_embedding = torch.nn.functional.interpolate(self.global_trend.unsqueeze(0).unsqueeze(0), size=self.max_time_diff, mode='linear', align_corners=True).squeeze(0).squeeze(0)

    def _reg_loss(self, user_indices, item_indices):
        user_emb = self.user_embedding.weight[user_indices]
        item_emb = self.item_embedding.weight[item_indices]
        user_time_emb = self.user_time_embedding.weight[user_indices]
        item_time_emb = self.item_time_embedding.weight[item_indices]
        reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) + item_emb.norm(2).pow(2) + user_time_emb.norm(2).pow(2) + item_time_emb.norm(2).pow(2)) / float(len(user_indices))
        return reg_loss
    
    def _graph_forward(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        final_emb = torch.cat([user_emb, item_emb])
        emb_list = [final_emb]
        for _ in range(self.num_layers):
            self.graph.ndata['h'] = final_emb
            self.graph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h'))
            final_emb = self.graph.ndata['h']
            emb_list.append(final_emb)
        final_emb = torch.mean(torch.stack(emb_list, dim=1), dim=1)

        user_emb, item_emb = torch.split(final_emb, [self.num_users, self.num_items])
        return user_emb, item_emb
    
    def forward(self, user_indices, item_indices, time_diffs, user_trends, item_trends):
        # matching score
        user_embeddings, item_embeddings = self._graph_forward()
        user_emb = user_embeddings[user_indices]
        item_emb = item_embeddings[item_indices]
        global_match = torch.sum(user_emb * item_emb, dim=-1)  # m_{u,v}

        # time matching score
        user_time_embeddings, item_time_embeddings = self.user_time_embedding.weight, self.item_time_embedding.weight
        user_time_emb = user_time_embeddings[user_indices]
        item_time_emb = item_time_embeddings[item_indices]

        # (batch_size, self.train_time_diff) -> ()
        user_trends = torch.cat((user_trends, self.mlp(user_trends)), dim=1)
        item_trends = torch.cat((item_trends, self.mlp(item_trends)), dim=1)

        # method-4
        user_time_match = torch.sum(user_time_emb * user_trends, dim=-1)
        item_time_match = torch.sum(item_time_emb * item_trends, dim=-1)

        reg_loss = self._reg_loss(user_indices, item_indices)
        return torch.sigmoid(global_match), torch.sigmoid(user_time_match + item_time_match), reg_loss
    
    def predict(self, user_indices, item_indices, time_diffs, user_trends, item_trends):
        # matching score
        user_embeddings, item_embeddings = self._graph_forward()
        user_emb = user_embeddings[user_indices]
        item_emb = item_embeddings[item_indices]
        global_match = torch.sum(user_emb * item_emb, dim=-1)  # m_{u,v}

        # time matching score
        user_time_embeddings, item_time_embeddings = self.user_time_embedding.weight, self.item_time_embedding.weight
        user_time_emb = user_time_embeddings[user_indices]
        item_time_emb = item_time_embeddings[item_indices]

        user_trends = torch.cat((user_trends, self.mlp(user_trends)), dim=1)
        item_trends = torch.cat((item_trends, self.mlp(item_trends)), dim=1)

        user_time_match = torch.sum(user_time_emb * user_trends, dim=-1).unsqueeze(1)
        item_time_match = torch.sum(item_time_emb * item_trends, dim=-1).unsqueeze(1)

        alpha = self.alpha
        pred = alpha * torch.sigmoid(global_match) + (1 - alpha) * torch.sigmoid(torch.sum((user_time_match + item_time_match) * self.global_time_embedding, dim=-1))
        
        return pred
