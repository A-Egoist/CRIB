import torch
from torch import nn


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def _reg_loss(self, user_indices, item_indices):
        user_emb = self.user_embedding.weight[user_indices]
        item_emb = self.item_embedding.weight[item_indices]
        reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) + item_emb.norm(2).pow(2)) / float(len(user_indices))
        return reg_loss

    def forward(self, user_indices, item_indices):
        user_embedding, item_embedding = self.user_embedding.weight, self.item_embedding.weight
        user_emb = user_embedding[user_indices]
        item_emb = item_embedding[item_indices]
        
        pred = torch.sum(user_emb * item_emb, dim=-1)
        reg_loss = self._reg_loss(user_indices, item_indices)
        # return torch.sigmoid(pred), reg_loss
        return pred, reg_loss
    
    def predict(self, user_indices, item_indices):
        user_embeddings, item_embeddings = self.user_embedding.weight, self.item_embedding.weight
        user_emb = user_embeddings[user_indices]
        item_emb = item_embeddings[item_indices]
        
        pred = torch.sum(user_emb * item_emb, dim=-1)
        return pred