import torch
import torch.nn as nn
from torchrec.modules.embedding_modules import EmbeddingBagCollection

class GraphAwareEmbeddingBagCollection(nn.Module):
    def __init__(self, ebc_config, gnn):
        super().__init__()
        self.ebc = EmbeddingBagCollection(**ebc_config)
        self.gnn = gnn

    def forward(self, features, edge_index):
        # Get embeddings from EmbeddingBagCollection
        ebc_embeddings = self.ebc(features)
        
        # Combine embeddings into a single tensor
        combined_embeddings = torch.cat([emb.values() for emb in ebc_embeddings.values()], dim=0)
        
        # Apply GNN to get graph-aware embeddings
        graph_embeddings = self.gnn(combined_embeddings, edge_index)
        
        # Split graph embeddings back into separate tensors for each key
        split_embeddings = torch.split(graph_embeddings, [emb.values().shape[0] for emb in ebc_embeddings.values()])
        
        # Create a new dictionary with graph-aware embeddings
        return {key: emb for key, emb in zip(ebc_embeddings.keys(), split_embeddings)}

def create_ebc_config(num_embeddings, embedding_dim):
    return {
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