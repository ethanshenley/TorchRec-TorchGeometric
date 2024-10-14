import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from src.data import LargeGraphRecDataset, load_and_preprocess_large_data
from src.models import create_large_ebc_config
from src.utils import distributed_evaluate_recommendations, plot_training_curve

class LargeGNNRecommender(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = create_large_ebc_config(num_nodes, embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index):
        x = self.embedding(x)['ids'].values()
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

def train(rank, world_size, data):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = LargeGNNRecommender(data.num_nodes, embedding_dim=64, hidden_dim=32).to(device)
    model = DistributedModelParallel(
        module=model,
        device=device,
        sharders=[EmbeddingBagCollectionSharder(
            kernel=EmbeddingComputeKernel.DENSE
        )]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=1024,
        input_nodes=data.train_mask,
        sampler=DistributedSampler(data.train_mask, num_replicas=world_size, rank=rank)
    )

    num_epochs = 50
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            user_emb = out[batch.user_index]
            item_emb = out[batch.item_index]
            pred = model.predictor(torch.cat([user_emb, item_emb], dim=1))
            loss = criterion(pred.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_losses.append(total_loss / len(train_loader))
        
        if rank == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}')

    if rank == 0:
        test_metrics = distributed_evaluate_recommendations(model, data, device)
        print("Test Metrics:", test_metrics)
        plot_training_curve(train_losses)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    data = load_and_preprocess_large_data('path_to_your_large_data.csv')
    torch.multiprocessing.spawn(train, args=(world_size, data), nprocs=world_size, join=True)
