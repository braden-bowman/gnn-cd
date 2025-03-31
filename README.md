# GNN Community Detection Framework

A comprehensive framework for comparing different community detection methods, including traditional algorithms, graph neural networks (GNNs), dynamic methods, and overlapping community detection techniques.

## Overview

This framework provides a structured approach to:

1. Generate or load graph data with known community structure
2. Apply various community detection methods
3. Evaluate and compare results across different methods
4. Visualize communities and evaluation metrics
5. Generate comprehensive evaluation reports
6. Apply community detection to cybersecurity data (UNSW-NB15 dataset)

## Features

- **Multiple Detection Methods**: Traditional (Louvain, Leiden, Label Propagation), GNN-based (GCN, GraphSAGE, GAT, VGAE), dynamic GNNs, and overlapping community detection
- **Synthetic Graph Generation**: Create graphs with known community structure for benchmarking
- **Comprehensive Evaluation**: Metrics like NMI, ARI, and modularity to compare methods
- **Advanced Visualization**: Visualize communities and their evolution over time
- **GPU Acceleration**: Support for GPU-accelerated computation with CUDA
- **Modular Design**: Easy to extend with new methods and customizations
- **Cybersecurity Analysis**: Apply community detection to identify network attack patterns in the UNSW-NB15 dataset

## Installation

### Requirements

The framework requires Python 3.8+ and the following packages:

```bash
pip install -r requirements.txt
```

Core dependencies:
- RustworkX (for efficient graph operations)
- Polars (for data processing)
- PyTorch (for neural networks)
- scikit-learn (for evaluation metrics)

For traditional methods:
- python-louvain (for Louvain method)
- cdlib (for additional algorithms and evaluation metrics)

For GNN-based methods:
- PyTorch Geometric

### Installation Steps

1. Clone this repository:
```bash
git clone https://github.com/yourusername/gnn-cd.git
cd gnn-cd
```

2. Install core dependencies:
```bash
pip install -r requirements.txt
```

3. Install in development mode:
```bash
pip install -e .
```

4. For using GNN-based methods, ensure PyTorch Geometric is installed correctly (see [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)).

## Framework Structure

The framework consists of Python modules and Jupyter notebooks:

### Python Modules

- `data_prep.py`: Functions for loading, generating, and preprocessing graph data
- `traditional_methods.py`: Implementation of traditional community detection algorithms
- `gnn_community_detection.py`: GNN-based community detection methods
- `dynamic_gnn.py`: Dynamic GNN models for evolving networks
- `overlapping_community_detection.py`: Methods for detecting overlapping communities
- `evaluation.py`: Utilities for evaluating and comparing detection results
- `visualization.py`: Advanced visualization functions for community structures

### Jupyter Notebooks

See the [notebooks README](notebooks/README.md) for detailed descriptions of each notebook.

## Quick Start

1. Install the package as described above
2. Run the notebooks in sequence, starting with data preparation:
```bash
jupyter notebook notebooks/1_Data_Preperation.ipynb
```

## Using the Framework in Your Own Code

You can import the framework modules in your own code:

```python
from community_detection import (
    generate_synthetic_graph, 
    run_louvain, 
    run_gnn_community_detection,
    evaluate_against_ground_truth
)

# Generate a synthetic graph
G, ground_truth = generate_synthetic_graph('sbm', n_nodes=100, n_communities=5)

# Run community detection methods
louvain_communities, _ = run_louvain(G)
gcn_results = run_gnn_community_detection(G, model_type='gcn')

# Evaluate results
louvain_metrics = evaluate_against_ground_truth(G, louvain_communities, 'community')
print(f"Louvain NMI: {louvain_metrics['nmi']}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use this framework in your research, please cite:

```
@software{gnn-community-detection,
  author = {Your Name},
  title = {GNN Community Detection Framework},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/yourusername/gnn-cd}
}
```

## Contact

[Your Contact Information]
