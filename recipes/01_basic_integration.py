import torch
from torch_geometric.loader import DataLoader
from src.data import GraphRecDataset, load_and_preprocess_data
from src.models import GraphRecommender
from src.utils import evaluate_recommendations, plot_training_curve

# Load and preprocess data
train_interactions, test_interactions, num_users, num_items = load_and_preprocess_data('path_to_your_data.csv')

# Create datasets
train_dataset = GraphRecDataset(train_interactions, num_users, num_items)
test_dataset = GraphRecDataset(test_interactions, num_users, num_items)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model
model = GraphRecommender(num_embeddings=max(num_users, num_items), 
                         embedding_dim=64, 
                         hidden_dim=32, 
                         num_layers=2)

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = torch.nn.MSELoss()

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