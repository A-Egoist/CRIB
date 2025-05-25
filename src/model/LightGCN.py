import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, graph, num_layers):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.graph = graph
        self.num_layers = num_layers

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def _reg_loss(self, user_indices, item_indices):
        user_emb = self.user_embedding.weight[user_indices]
        item_emb = self.item_embedding.weight[item_indices]
        reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) + item_emb.norm(2).pow(2)) / float(len(user_indices))
        return reg_loss
    
    def _graph_forward(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        
        final_embs = torch.cat([user_embeddings, item_embeddings])
        emb_list = [final_embs]
        for _ in range(self.num_layers):
            self.graph.ndata['h'] = final_embs
            self.graph.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h'))
            final_embs = self.graph.ndata['h']
            emb_list.append(final_embs)
        final_embs = torch.mean(torch.stack(emb_list, dim=1), dim=1)

        user_embeddings, item_embeddings = torch.split(final_embs, [self.num_users, self.num_items])
        return user_embeddings, item_embeddings
    
    def forward(self, user_indices, item_indices):
        user_embeddings, item_embeddings = self._graph_forward()
        user_emb = user_embeddings[user_indices]
        item_emb = item_embeddings[item_indices]

        pred = torch.sum(user_emb * item_emb, dim=-1)
        reg_loss = self._reg_loss(user_indices, item_indices)
        # return torch.sigmoid(pred), reg_loss
        return pred, reg_loss
    
    def predict(self, user_indices, item_indices):
        user_embeddings, item_embeddings = self._graph_forward()
        user_emb = user_embeddings[user_indices]
        item_emb = item_embeddings[item_indices]
        
        pred = torch.sum(user_emb * item_emb, dim=-1)
        return pred