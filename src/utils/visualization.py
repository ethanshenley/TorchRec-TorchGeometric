import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_networkx

def visualize_graph(edge_index, num_nodes):
    G = to_networkx(torch.tensor(edge_index), num_nodes=num_nodes)
    plt.figure(figsize=(10, 10))
    nx.draw(G, node_size=50, with_labels=False)
    plt.title("User-Item Interaction Graph")
    plt.show()

def plot_training_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def visualize_embeddings(embeddings, labels, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
