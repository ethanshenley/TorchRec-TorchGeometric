# TorchRec and PyTorch Geometric Cookbook

This repository provides a comprehensive set of recipes for building advanced recommendation systems using TorchRec and PyTorch Geometric. It demonstrates how to leverage graph neural networks and large-scale embedding techniques for various recommendation tasks.

## Features

- Basic integration of TorchRec and PyTorch Geometric
- Graph-aware embeddings for enhanced recommendations
- GNN-enhanced collaborative filtering
- Heterogeneous graph recommendations
- Distributed training for large-scale graphs
- Temporal graph recommendations

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/ethanshenley/torchrec-torchgeometric-cookbook.git
   cd torchrec-torchgeometric
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the package and its dependencies:
   ```
   pip install -e .
   ```

## Usage

Each recipe is available as both a Python script in the `recipes/` directory and as a Jupyter notebook in the `notebooks/` directory.

To run a recipe script:
```
python recipes/01_basic_integration.py
```

To use a notebook, start Jupyter and open the desired notebook:
```
jupyter notebook
```

## Recipes

1. [Basic Integration](notebooks/01_basic_integration.ipynb)
2. [Graph-Aware Embeddings](notebooks/02_graph_aware_embeddings.ipynb)
3. [GNN-Enhanced Collaborative Filtering](notebooks/03_gnn_enhanced_collab_filter.ipynb)
4. [Heterogeneous Graph Recommendations](notebooks/04_heterogeneous_graph_recommendations.ipynb)
5. [Distributed Training](notebooks/05_distributed_training.ipynb)
6. [Temporal Graph Recommendations](notebooks/06_temporal_graph_recommendations.ipynb)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
