import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from src.data import HeteroGraphRecDataset, load_and_preprocess_hetero_data
from src.models import create_ebc_config
from src.utils import evaluate_recommendations, plot_training_curve
from torchrec.modules.embedding_modules import EmbeddingBagCollection

class HeteroGNNRecommender(nn.Module):
    def __init__(self, metadata, embedding_dim, hidden_dim):
        super().__init__()
        
        # Create embeddings for each node type
        self.embedding = nn.ModuleDict()
        for node_type in metadata[0]:
            ebc_config = create_ebc_config(metadata[1][node_type], embedding_dim)
            self.embedding[node_type] = EmbeddingBagCollection(**ebc_config)
        
        # Heterogeneous GNN layers
        self.convs = nn.ModuleList()
        for _ in range(2):  # 2-layer GNN
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_dim)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        # Final prediction layer
        self.predictor = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_dict, edge_index_dict):
        # Get initial embeddings for each node type
        for node_type in x_dict:
            x_dict[node_type] = self.embedding[node_type](x_dict[node_type])['ids'].values()
        
        # Apply GNN layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Predict ratings
        user_emb = x_dict['user']
        item_emb = x_dict['item']
        
        # Assuming the first dimension of user_emb and item_emb are aligned
        combined = torch.cat([user_emb, item_emb], dim=1)
        return self.predictor(combined)

# Load and preprocess heterogeneous graph data
data, num_nodes_dict = load_and_preprocess_hetero_data('path_to_your_hetero_data.csv')

# Create datasets and data loaders
train_dataset = HeteroGraphRecDataset(data['train'], num_nodes_dict)
test_dataset = HeteroGraphRecDataset(data['test'], num_nodes_dict)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, optimizer, and loss function
model = HeteroGNNRecommender(data['train'].metadata(), embedding_dim=64, hidden_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data['train'] = data['train'].to(device)
data['test'] = data['test'].to(device)
num_epochs = 50
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = criterion(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict)
            val_loss += criterion(out.squeeze(), batch.y).item()
    val_losses.append(val_loss / len(test_loader))
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# Evaluate the model
test_metrics = evaluate_recommendations(model, test_loader, device)
print("Test Metrics:", test_metrics)

# Plot training curve
plot_training_curve(train_losses, val_losses)
