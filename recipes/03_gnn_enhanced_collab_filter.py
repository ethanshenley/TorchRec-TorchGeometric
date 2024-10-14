import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from src.data import GraphRecDataset, load_and_preprocess_data, create_adjacency_matrix
from src.models import create_ebc_config
from src.utils import evaluate_recommendations, plot_training_curve
from torchrec.modules.embedding_modules import EmbeddingBagCollection

class GNNCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        # User and Item embeddings
        ebc_config = create_ebc_config(max(num_users, num_items), embedding_dim)
        self.embedding = EmbeddingBagCollection(**ebc_config)
        
        # GNN layers
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Final prediction layer
        self.predictor = nn.Linear(hidden_dim * 2, 1)

    def forward(self, features, edge_index):
        # Get embeddings
        embeddings = self.embedding(features)
        user_emb = embeddings['user'].values()
        item_emb = embeddings['item'].values()
        
        # Combine user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=0)
        
        # Apply GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Split back into user and item embeddings
        user_emb, item_emb = x[:self.num_users], x[self.num_users:]
        
        # Get relevant user and item embeddings
        user_indices = features['user'].values().squeeze()
        item_indices = features['item'].values().squeeze()
        user_emb = user_emb[user_indices]
        item_emb = item_emb[item_indices]
        
        # Predict ratings
        combined = torch.cat([user_emb, item_emb], dim=1)
        return self.predictor(combined)

# Load and preprocess data
train_interactions, test_interactions, num_users, num_items = load_and_preprocess_data('path_to_your_data.csv')

# Create adjacency matrix
edge_index = create_adjacency_matrix(train_interactions, num_users, num_items)

# Create datasets and data loaders
train_dataset = GraphRecDataset(train_interactions, num_users, num_items)
test_dataset = GraphRecDataset(test_interactions, num_users, num_items)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, optimizer, and loss function
model = GNNCollaborativeFiltering(num_users, num_items, embedding_dim=64, hidden_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
edge_index = edge_index.to(device)
num_epochs = 50
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.features, edge_index)
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
            out = model(batch.features, edge_index)
            val_loss += criterion(out.squeeze(), batch.y).item()
    val_losses.append(val_loss / len(test_loader))
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# Evaluate the model
test_metrics = evaluate_recommendations(model, test_loader, device)
print("Test Metrics:", test_metrics)

# Plot training curve
plot_training_curve(train_losses, val_losses)
