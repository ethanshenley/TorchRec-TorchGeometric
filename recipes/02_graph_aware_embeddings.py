import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from src.data import GraphRecDataset, load_and_preprocess_data
from src.models import GraphAwareEmbeddingBagCollection, GNNStack, create_ebc_config
from src.utils import evaluate_recommendations, plot_training_curve, visualize_embeddings
from sklearn.manifold import TSNE

class GraphAwareRecommender(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        ebc_config = create_ebc_config(num_embeddings, embedding_dim)
        self.gnn = GNNStack(embedding_dim, hidden_dim, embedding_dim, num_layers)
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
        combined = torch.cat([user_emb, item_emb], dim=1)
        return self.predictor(combined)

# Load and preprocess data
train_interactions, test_interactions, num_users, num_items = load_and_preprocess_data('path_to_your_data.csv')

# Create datasets and data loaders
train_dataset = GraphRecDataset(train_interactions, num_users, num_items)
test_dataset = GraphRecDataset(test_interactions, num_users, num_items)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, optimizer, and loss function
model = GraphAwareRecommender(num_embeddings=max(num_users, num_items), 
                              embedding_dim=64, 
                              hidden_dim=32, 
                              num_layers=2)
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
        out = model(batch.features, batch.edge_index)
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
            out = model(batch.features, batch.edge_index)
            val_loss += criterion(out.squeeze(), batch.y).item()
    val_losses.append(val_loss / len(test_loader))
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# Evaluate the model
test_metrics = evaluate_recommendations(model, test_loader, device)
print("Test Metrics:", test_metrics)

# Plot training curve
plot_training_curve(train_losses, val_losses)

# Visualize embeddings
model.eval()
with torch.no_grad():
    batch = next(iter(test_loader)).to(device)
    embeddings = model.graph_aware_ebc(batch.features, batch.edge_index)
    user_emb = embeddings['user'].cpu().numpy()
    item_emb = embeddings['item'].cpu().numpy()

# Use t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
user_emb_2d = tsne.fit_transform(user_emb)
item_emb_2d = tsne.fit_transform(item_emb)

# Visualize user and item embeddings
visualize_embeddings(user_emb_2d, labels=range(len(user_emb_2d)), title="User Embeddings")
visualize_embeddings(item_emb_2d, labels=range(len(item_emb_2d)), title="Item Embeddings")