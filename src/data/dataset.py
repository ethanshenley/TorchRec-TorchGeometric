import torch
from torch_geometric.data import Dataset, Data
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

class GraphRecDataset(Dataset):
    def __init__(self, user_item_interactions, num_users, num_items, transform=None):
        super().__init__(transform)
        self.user_item_interactions = user_item_interactions
        self.num_users = num_users
        self.num_items = num_items

    def len(self):
        return len(self.user_item_interactions)

    def get(self, idx):
        user, item, rating = self.user_item_interactions[idx]
        
        # Create graph structure
        edge_index = torch.tensor([[user, self.num_users + item],
                                   [self.num_users + item, user]], dtype=torch.long)
        
        # Create node features (you might want to extend this)
        x = torch.zeros((self.num_users + self.num_items, 1))
        
        # Create KeyedJaggedTensor for TorchRec
        keys = ["user", "item"]
        values = torch.tensor([user, item], dtype=torch.long)
        lengths = torch.tensor([1, 1], dtype=torch.long)
        features = KeyedJaggedTensor.from_lengths_sync(keys=keys, values=values, lengths=lengths)
        
        return Data(x=x, edge_index=edge_index, features=features, y=torch.tensor([rating], dtype=torch.float))