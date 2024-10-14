import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import TemporalDataLoader
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from src.data import TemporalGraphRecDataset, load_and_preprocess_temporal_data
from src.models import create_ebc_config
from src.utils import evaluate_temporal_recommendations, plot_training_curve

class TemporalGNNRecommender(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.embedding = EmbeddingBagCollection(
            tables=[create_ebc_config(num_nodes, embedding_dim)]
        )
        self.conv1 = GATConv(embedding_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.predictor = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index, edge_attr):
        # Get node embeddings
        x = self.embedding(x)['ids'].values()
        
        # Apply GAT layers
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        
        return x

    def predict(self, x, user_index, item_index):
        # Apply LSTM to user and item sequences
        user_seq = x[user_index].unsqueeze(0)
        item_seq = x[item_index].unsqueeze(0)
        _, (user_hidden, _) = self.lstm(user_seq)
        _, (item_hidden, _) = self.lstm(item_seq)
        
        # Concatenate user and item representations
        combined = torch.cat([user_hidden.squeeze(0), item_hidden.squeeze(0)], dim=1)
        
        # Make prediction
        return self.predictor(combined)

# Load and preprocess temporal graph data
data, num_nodes = load_and_preprocess_temporal_data('path_to_your_temporal_data.csv')

# Create dataset and data loader
dataset = TemporalGraphRecDataset(data)
train_loader = TemporalDataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model, optimizer, and loss function
model = TemporalGNNRecommender(num_nodes, embedding_dim=64, hidden_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
num_epochs = 50
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        pred = model.predict(out, batch.user_index, batch.item_index)
        loss = criterion(pred.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    
    # Validation (assuming we have a validation set)
    val_loss = evaluate_temporal_recommendations(model, dataset.val_mask, device)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# Evaluate the model
test_metrics = evaluate_temporal_recommendations(model, dataset.test_mask, device)
print("Test Metrics:", test_metrics)

# Plot training curve
plot_training_curve(train_losses, val_losses)