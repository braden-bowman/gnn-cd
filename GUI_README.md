# GNN-CD GUI Documentation

## Overview

The GNN-CD GUI provides a user-friendly web interface for:
- Loading and processing network data
- Running various community detection algorithms
- Training and fine-tuning GNN models
- Visualizing network graphs and communities
- Evaluating and comparing different methods

## Installation

To install all requirements including the GUI frontend:

```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

Or install directly with:

```bash
pip install -e ".[all]"
pip install "nicegui>=1.3.0"
```

## Running the GUI

### Command-line options

Start the GUI with default settings:

```bash
python run_gui.py
```

Or use command-line options to customize:

```bash
python run_gui.py --host localhost --port 8888 --debug
```

Available options:
- `--host`: Host address to bind to (default: 0.0.0.0, which means all interfaces)
- `--port`: Port to run the GUI on (default: 8080)
- `--debug`: Run in debug mode with additional logging

### Accessing the GUI

After starting, access the GUI by opening your browser to:
- If running locally: `http://localhost:8080/`
- If connecting from another machine: `http://<your-ip-address>:8080/`

## Using the GUI

The GUI is organized into tabs that follow the typical workflow:

### Data Upload

Upload network data in various formats:
- Edge lists (CSV or Parquet)
- Pre-existing graph files (GraphML or gpickle)
- Labeled data (for ground truth communities)
- Node features (for GNN models)

Advanced options for:
- Directed/undirected graphs
- Weighted/unweighted edges
- Node limits for large graphs

### Community Detection

Run various detection algorithms:
- **Traditional methods**: Louvain, Leiden, Label Propagation, Infomap, etc.
- **GNN-based methods**: GCN, GraphSAGE, GAT, VGAE
- **Overlapping community detection**: BigCLAM, DEMON, SLPA

Each method has configurable parameters.

### Model Training

Train new GNN models:
- Various architectures (GCN, GraphSAGE, GAT, VGAE, EvolveGCN, DySAT)
- Configurable parameters (embedding dimension, hidden layers, etc.)
- Training progress visualization

Fine-tune existing models:
- Upload pre-trained model
- Configure fine-tuning parameters (epochs, learning rate)
- Freeze base layers option

### Evaluation

Evaluate and compare methods:
- Performance metrics (NMI, ARI, Modularity)
- Execution time comparison
- Comparative charts and tables

Run multiple methods in batch for side-by-side comparison.

### Visualization

Visualize network data:
- Interactive network graph with community coloring
- Community size distribution
- Node embedding visualization (t-SNE)

Customization options:
- Node coloring schemes (Community, Degree, Node Type)
- Layout algorithms (Force-Directed, Circular, Spectral, etc.)
- Node limit for large graphs

### Results and Export

Export results for further analysis:
- Community assignments (CSV, Parquet, JSON)
- Trained models
- Node embeddings

View experiment summary with key statistics.

## Deployment Options

The GUI can be deployed in different ways:

### Local Development

```bash
python run_gui.py --host localhost --port 8080
```

Only accessible from the local machine.

### Local Network

```bash
python run_gui.py --host 0.0.0.0 --port 8080
```

Accessible from any machine on the local network.

### Internet Deployment

For internet deployment:

1. Set up a reverse proxy (like Nginx or Caddy)
2. Add HTTPS support
3. Consider authentication for security

Example with Caddy:

```caddyfile
your-domain.com {
    reverse_proxy localhost:8080
}
```

## Troubleshooting

Common issues:

- **NiceGUI not found**: Ensure you've installed it with `pip install nicegui>=1.3.0`
- **Port already in use**: Change the port with `--port` option
- **Large graph performance**: Use node limit in visualization settings
- **Module errors**: Make sure you've installed all dependencies with `pip install -e ".[all]"`
