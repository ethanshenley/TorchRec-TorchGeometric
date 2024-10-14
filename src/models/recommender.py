import torch
import torch.nn as nn
from .embedding import GraphAwareEmbeddingBagCollection
from .gnn import GNNStack

class GraphRecommender(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers, gnn_type='GCN'):
        super().__init__()
        ebc_config = {
            "tables": [
                {
                    "name": "user",
                    "embedding_dim": embedding_dim,
                    "num_embeddings": num_embeddings,
                    "feature_names": ["user_id"],
                },
                {
                    "name": "item",
                    "embedding_dim": embedding_dim,
                    "num_embeddings": num_embeddings,
                    "feature_names": ["item_id"],
                },
            ],
        }
        self.gnn = GNNStack(embedding_dim, hidden_dim, embedding_dim, num_layers, gnn_type)
        self.graph_aware_ebc = GraphAwareEmbeddingBagCollection(ebc_config, self.gnn)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, edge_index):
        embeddings = self.graph_aware_ebc(features, edge_index)
        user_emb = embeddings['user']
        item_emb = embeddings['item']
        
        # Assuming the first half of the embeddings are users and the second half are items
        user_item_embeddings = torch.cat([user_emb, item_emb], dim=1)
        
        return self.predictor(user_item_embeddings)
