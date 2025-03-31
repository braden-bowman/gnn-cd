# Dynamic Graph Neural Networks for Community Detection
# ====================================================

import os
import torch
import polars as pl
import rustworkx as rx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

# For GPU acceleration
try:
    import cudf
    import cugraph
    CUGRAPH_AVAILABLE = True
except ImportError:
    CUGRAPH_AVAILABLE = False

# For PyTorch and PyTorch Geometric
try:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Dynamic GNN methods will be skipped.")

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_networkx, from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. Dynamic GNN methods will be skipped.")

# Import functions from other modules
from .gnn_community_detection import (rwx_to_pyg, extract_embeddings,
                                  detect_communities_from_embeddings,
                                  evaluate_communities, plot_embeddings,
                                  add_communities_to_graph)


# Function to generate a sequence of dynamic graphs
def generate_dynamic_graphs(n_time_steps: int = 5, n_nodes: int = 100, 
                          n_communities: int = 5, change_fraction: float = 0.1, 
                          seed: int = 42) -> List[rx.PyGraph]:
    """
    Generate a sequence of dynamic graphs with evolving community structure
    
    Parameters:
    -----------
    n_time_steps: int
        Number of time steps
    n_nodes: int
        Number of nodes in each graph
    n_communities: int
        Number of communities
    change_fraction: float
        Fraction of nodes that change communities between time steps
    seed: int
        Random seed
        
    Returns:
    --------
    graphs: list
        List of RustWorkX graphs for each time step
    """
    torch.manual_seed(seed)
    
    # Initialize graphs list
    graphs = []
    
    # Generate initial graph (using stochastic block model)
    import networkx as nx  # For SBM generation
    
    # Calculate sizes for each community (handling the case when n_nodes is not divisible by n_communities)
    base_size = n_nodes // n_communities
    remainder = n_nodes % n_communities
    community_sizes = []
    for i in range(n_communities):
        if i < remainder:
            community_sizes.append(base_size + 1)
        else:
            community_sizes.append(base_size)
    
    # Create probability matrix
    p_matrix = torch.ones((n_communities, n_communities)) * 0.05  # p_out
    p_matrix.fill_diagonal_(0.3)  # p_in
    
    G_nx = nx.stochastic_block_model(
        community_sizes,
        p_matrix.numpy(),
        seed=seed
    )
    
    # Calculate the starting node for each community
    community_start_nodes = [0]
    for i in range(1, n_communities):
        community_start_nodes.append(community_start_nodes[i-1] + community_sizes[i-1])
    
    # Add community labels
    for i in range(n_nodes):
        # Find which community this node belongs to
        comm_idx = 0
        for j in range(1, n_communities):
            if i >= community_start_nodes[j]:
                comm_idx = j
        G_nx.nodes[i]['community'] = comm_idx
    
    # Get initial community assignments
    community_assignments = torch.tensor([G_nx.nodes[i]['community'] for i in range(n_nodes)])
    
    # Convert to RustWorkX
    G_init = rx.PyGraph()
    
    # Add nodes with attributes
    node_mapping = {}
    for node in G_nx.nodes():
        attrs = G_nx.nodes[node]
        node_mapping[node] = G_init.add_node(attrs)
    
    # Add edges with attributes
    for u, v, data in G_nx.edges(data=True):
        G_init.add_edge(node_mapping[u], node_mapping[v], data if data else None)
    
    # Add first graph to list
    graphs.append(G_init)
    
    # Generate subsequent graphs with evolving community structure
    for t in range(1, n_time_steps):
        # Decide which nodes change communities
        n_changes = int(n_nodes * change_fraction)
        change_indices = torch.randperm(n_nodes)[:n_changes]
        
        # Update community assignments
        for idx in change_indices:
            # Assign a different community
            current_comm = community_assignments[idx].item()
            available_comms = [c for c in range(n_communities) if c != current_comm]
            new_comm = available_comms[torch.randint(0, len(available_comms), (1,)).item()]
            community_assignments[idx] = new_comm
        
        # Create new graph with updated communities
        # First, create probability matrix for SBM based on updated communities
        sizes = [(community_assignments == c).sum().item() for c in range(n_communities)]
        
        # Create probability matrix
        p_in = 0.3  # probability within community
        p_out = 0.05  # probability between communities
        p_matrix = torch.ones((n_communities, n_communities)) * p_out
        p_matrix.fill_diagonal_(p_in)
        
        # Generate new graph with NetworkX
        G_nx_t = nx.stochastic_block_model(sizes, p_matrix.numpy(), seed=seed+t)
        
        # Add community assignments as node attributes
        node_index = 0
        for comm, size in enumerate(sizes):
            for _ in range(size):
                G_nx_t.nodes[node_index]['community'] = comm
                node_index += 1
        
        # Convert to RustWorkX
        G_t = rx.PyGraph()
        
        # Add nodes with attributes
        node_mapping = {}
        for node in G_nx_t.nodes():
            attrs = G_nx_t.nodes[node]
            node_mapping[node] = G_t.add_node(attrs)
        
        # Add edges with attributes
        for u, v, data in G_nx_t.edges(data=True):
            G_t.add_edge(node_mapping[u], node_mapping[v], data if data else None)
        
        # Add to graphs list
        graphs.append(G_t)
    
    return graphs


# Function to visualize dynamic communities
def visualize_dynamic_communities(graphs: List[rx.PyGraph], 
                                community_attr: str = 'community', 
                                figsize: Tuple[int, int] = (15, 10),
                                layout: str = 'fixed', 
                                title: str = "Communities Over Time"):
    """
    Visualize the evolution of communities in dynamic graphs
    
    Parameters:
    -----------
    graphs: list
        List of RustWorkX graphs for each time step
    community_attr: str
        Node attribute for community assignment
    figsize: tuple
        Figure size
    layout: str
        Layout type ('fixed' or 'separate')
    title: str
        Plot title
        
    Returns:
    --------
    None
    """
    n_time_steps = len(graphs)
    
    # Convert to NetworkX for visualization
    import networkx as nx
    graphs_nx = []
    
    for G in graphs:
        G_nx = nx.Graph()
        
        # Add nodes with data
        for i in range(len(G)):
            node_data = G.get_node_data(i)
            G_nx.add_node(i, **({} if node_data is None else node_data))
        
        # Add edges
        for edge in G.edge_list():
            source, target = edge[0], edge[1]
            edge_data = G.get_edge_data(source, target)
            G_nx.add_edge(source, target, **({} if edge_data is None else edge_data))
            
        graphs_nx.append(G_nx)
    
    # Set up the figure
    fig, axes = plt.subplots(1, n_time_steps, figsize=figsize)
    if n_time_steps == 1:
        axes = [axes]
    
    # Calculate fixed layout for all time steps if requested
    if layout == 'fixed':
        pos = nx.spring_layout(graphs_nx[0], seed=42)
    
    # Visualize each time step
    for t, G_nx in enumerate(graphs_nx):
        # Get communities
        communities = [G_nx.nodes[i].get(community_attr, 0) for i in range(len(G_nx))]
        
        # Calculate layout (use same layout for all time steps if fixed)
        if layout != 'fixed':
            pos = nx.spring_layout(G_nx, seed=42)
        
        # Plot
        ax = axes[t]
        nx.draw_networkx(G_nx, pos=pos, node_color=communities, cmap='rainbow',
                        with_labels=False, node_size=80, ax=ax)
        ax.set_title(f'Time Step {t+1}')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


# 1. EvolveGCN model
class EvolveGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super(EvolveGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Use PyTorch Geometric GCN layers instead of standard linear layers
        self.convs = nn.ModuleList()
        
        # Input layer - use a GCNConv which handles the graph structure
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # RNN for weight evolution
        # Since GCNConv weight has shape [out_channels, in_channels]
        # We need to track it for the last layer
        self.weight_rnn = nn.GRU(
            input_size=1,  # Treating each weight as a sequence
            hidden_size=1,
            batch_first=True
        )
        
        # Initialize weights
        self.reset_parameters()
        
        # Debug info
        print(f"EvolveGCN initialized with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def _evolve_weights(self, layer):
        """Helper method to evolve weights using RNN"""
        # Get the weight from the layer
        weight = layer.lin.weight
        
        # Reshape for sequence processing [batch=out_features, seq_len=in_features, features=1]
        weight_seq = weight.reshape(weight.size(0), weight.size(1), 1)
        
        # Run through RNN
        evolved_weight_seq, _ = self.weight_rnn(weight_seq)
        
        # Reshape back and update
        evolved_weight = evolved_weight_seq.reshape(weight.size(0), weight.size(1))
        
        # Update the weight in the layer
        layer.lin.weight.data = evolved_weight
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
              prev_weights: Optional[Any] = None):
        """
        Forward pass for EvolveGCN.
        
        Args:
            x: Node features tensor of shape [num_nodes, input_dim]
            edge_index: Edge indices for the graph
            prev_weights: Flag indicating if this is after first timestep
        """
        # Silently fix dimension mismatch (feature dimensions should be standardized already)
        if x.size(1) != self.input_dim:
            if x.size(1) > self.input_dim:
                # Truncate
                x = x[:, :self.input_dim]
            else:
                # Pad with zeros
                padding = torch.zeros(x.size(0), self.input_dim - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        # Evolve weights of the last layer if we're past the first timestep
        if prev_weights is not None:
            self._evolve_weights(self.convs[-1])
        
        # Apply GCN layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        return x


# 2. DySAT: Dynamic Self-Attention Network (simplified)
class DySAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                num_heads: int = 8, num_layers: int = 2):
        super(DySAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Structural attention (simplified to GAT)
        self.struct_attention = nn.ModuleList()
        self.struct_attention.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads))
        
        for _ in range(num_layers - 2):
            self.struct_attention.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads))
        
        self.struct_attention.append(
            GATConv(hidden_dim, output_dim // num_heads, heads=num_heads, concat=True))
        
        # Temporal attention would go here in a full implementation
        # Simplified for this example
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # Apply structural attention
        for i, attention in enumerate(self.struct_attention[:-1]):
            x = attention(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Output layer
        x = self.struct_attention[-1](x, edge_index)
        
        return x


# Function to train dynamic GNN model on a sequence of graphs
def train_dynamic_gnn(model_type: str, graphs: List[rx.PyGraph], 
                    embedding_dim: int = 16, epochs: int = 100, 
                    lr: float = 0.01) -> Tuple[nn.Module, List[Data]]:
    """
    Train a dynamic GNN model on a sequence of graphs
    
    Parameters:
    -----------
    model_type: str
        Type of dynamic GNN model ('evolvegcn' or 'dysat')
    graphs: list
        List of RustWorkX graphs for each time step
    embedding_dim: int
        Dimension of node embeddings
    epochs: int
        Number of training epochs
    lr: float
        Learning rate
        
    Returns:
    --------
    model: torch.nn.Module
        Trained dynamic GNN model
    data_list: list
        List of PyTorch Geometric Data objects for each time step
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric not available")
    
    # Convert graphs to PyG Data objects and standardize feature dimensions
    temp_data_list = [rwx_to_pyg(G) for G in graphs]
    
    # Find the maximum feature dimension across all time steps
    max_features = max(data.x.size(1) for data in temp_data_list)
    print(f"Standardizing feature dimensions to {max_features} across all time steps")
    
    # Standardize all data objects to have the same feature dimension
    data_list = []
    for i, data in enumerate(temp_data_list):
        if data.x.size(1) < max_features:
            # Pad with zeros to match max_features
            padding = torch.zeros(data.x.size(0), max_features - data.x.size(1))
            standardized_x = torch.cat([data.x, padding], dim=1)
            data.x = standardized_x
        elif data.x.size(1) > max_features:
            # Truncate to match max_features
            data.x = data.x[:, :max_features]
        
        data_list.append(data)
    
    # Print data_list shapes for debugging
    print("Data features shapes after standardization:")
    for i, data in enumerate(data_list):
        print(f"Time step {i}: x={data.x.shape}, edge_index={data.edge_index.shape}")
        if hasattr(data, 'y'):
            print(f"  y={data.y.shape}")
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data_list = [data.to(device) for data in data_list]
    
    # Get dimensions
    input_dim = data_list[0].x.size(1)
    print(f"Detected input dimension: {input_dim}")
    
    # Initialize model
    try:
        if model_type.lower() == 'evolvegcn':
            model = EvolveGCN(input_dim, hidden_dim=32, output_dim=embedding_dim).to(device)
        elif model_type.lower() == 'dysat':
            model = DySAT(input_dim, hidden_dim=32, output_dim=embedding_dim).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Process each time step
        prev_weights = None
        total_loss = 0
        
        try:
            for t, data in enumerate(data_list):
                # Forward pass
                if model_type.lower() == 'evolvegcn':
                    output = model(data.x, data.edge_index, prev_weights if t > 0 else None)
                    # We don't need to track prev_weights with the new implementation
                    prev_weights = True  # Just a flag to indicate we've had a previous iteration
                else:
                    output = model(data.x, data.edge_index)
                
                # If we have ground truth communities, use supervised loss
                if hasattr(data, 'y') and data.y is not None:
                    loss = F.cross_entropy(output, data.y)
                else:
                    # For unsupervised setting, use a simplified contrastive loss
                    # Create negative samples by permuting nodes
                    neg_edge_index = data.edge_index[:, torch.randperm(data.edge_index.size(1))]
                    
                    # Positive loss (real edges should have high similarity)
                    pos_z = output[data.edge_index[0]] * output[data.edge_index[1]]
                    pos_loss = -torch.log(torch.sigmoid(pos_z.sum(dim=1))).mean()
                    
                    # Negative loss (fake edges should have low similarity)
                    neg_z = output[neg_edge_index[0]] * output[neg_edge_index[1]]
                    neg_loss = -torch.log(1 - torch.sigmoid(neg_z.sum(dim=1))).mean()
                    
                    loss = pos_loss + neg_loss
                
                total_loss += loss
        except Exception as e:
            print(f"Error during forward/backward pass: {e}")
            raise
        
        # Backward pass
        try:
            total_loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"Error during optimizer step: {e}")
            raise
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}')
    
    return model, data_list


# Function to extract temporal embeddings from trained dynamic GNN
def extract_temporal_embeddings(model: nn.Module, data_list: List[Data], 
                              model_type: str = 'evolvegcn') -> List[torch.Tensor]:
    """
    Extract node embeddings for each time step
    
    Parameters:
    -----------
    model: torch.nn.Module
        Trained dynamic GNN model
    data_list: list
        List of PyTorch Geometric Data objects for each time step
    model_type: str
        Type of dynamic GNN model ('evolvegcn' or 'dysat')
        
    Returns:
    --------
    embeddings_list: list
        List of node embeddings for each time step
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data_list = [data.to(device) for data in data_list]
    
    # Ensure the model is in evaluation mode
    model.eval()
    embeddings_list = []
    prev_weights = None
    
    print("Extracting temporal embeddings...")
    
    # Get expected input dimension from the model
    expected_dim = model.input_dim if hasattr(model, 'input_dim') else None
    if expected_dim:
        print(f"Model expects input dimension: {expected_dim}")
    
    with torch.no_grad():
        try:
            for t, data in enumerate(data_list):
                print(f"Processing time step {t+1}/{len(data_list)}")
                
                # Forward pass
                if model_type.lower() == 'evolvegcn':
                    # No need to warn about dimensions - already standardized
                    embeddings = model(data.x, data.edge_index, prev_weights if t > 0 else None)
                    prev_weights = True  # Just a flag to indicate we've had a previous iteration
                else:
                    embeddings = model(data.x, data.edge_index)
                
                # Validate embeddings 
                if torch.isnan(embeddings).any():
                    print("WARNING: NaN values detected in embeddings!")
                
                # Convert to CPU
                embeddings_list.append(embeddings.cpu())
                print(f"Extracted embeddings shape: {embeddings.shape}")
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            raise
    
    return embeddings_list


# Function to detect communities from temporal embeddings
def detect_temporal_communities(embeddings_list: List[torch.Tensor], 
                              n_clusters: Optional[int] = None, 
                              method: str = 'kmeans') -> List[torch.Tensor]:
    """
    Detect communities from temporal embeddings
    
    Parameters:
    -----------
    embeddings_list: list
        List of node embeddings for each time step
    n_clusters: int
        Number of clusters (communities)
    method: str
        Clustering method ('kmeans' or 'spectral')
        
    Returns:
    --------
    communities_list: list
        List of community assignments for each time step
    """
    communities_list = []
    
    for t, embeddings in enumerate(embeddings_list):
        # Detect communities for this time step
        communities = detect_communities_from_embeddings(
            embeddings, n_clusters=n_clusters, method=method)
        
        communities_list.append(communities)
    
    return communities_list


# Function to evaluate temporal communities against ground truth
def evaluate_temporal_communities(communities_list: List[torch.Tensor], 
                                graphs: List[rx.PyGraph], 
                                ground_truth_attr: str = 'community') -> List[Dict[str, float]]:
    """
    Evaluate detected communities against ground truth for each time step
    
    Parameters:
    -----------
    communities_list: list
        List of community assignments for each time step
    graphs: list
        List of RustWorkX graphs for each time step
    ground_truth_attr: str
        Node attribute for ground truth communities
        
    Returns:
    --------
    metrics_list: list
        List of evaluation metrics for each time step
    """
    metrics_list = []
    
    for t, (communities, G) in enumerate(zip(communities_list, graphs)):
        # Get ground truth
        ground_truth = []
        for i in range(len(G)):
            node_data = G.get_node_data(i)
            if node_data and ground_truth_attr in node_data:
                ground_truth.append(node_data[ground_truth_attr])
            else:
                ground_truth.append(0)  # Default
                
        ground_truth = torch.tensor(ground_truth, dtype=torch.long)
        
        # Evaluate
        metrics = evaluate_communities(communities, ground_truth)
        metrics_list.append(metrics)
    
    return metrics_list


# Function to visualize community evolution
def visualize_community_evolution(communities_list: List[torch.Tensor], 
                                n_nodes: int, figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize the evolution of community assignments over time
    
    Parameters:
    -----------
    communities_list: list
        List of community assignments for each time step
    n_nodes: int
        Number of nodes
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    n_time_steps = len(communities_list)
    
    # Prepare data for visualization
    community_data = torch.zeros((n_nodes, n_time_steps))
    
    for t, communities in enumerate(communities_list):
        community_data[:, t] = communities.float()
    
    # Visualize
    plt.figure(figsize=figsize)
    plt.imshow(community_data.numpy(), aspect='auto', cmap='rainbow')
    plt.colorbar(label='Community')
    plt.title('Community Evolution Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Node ID')
    plt.tight_layout()
    plt.show()


# Function to run dynamic community detection
def run_dynamic_community_detection(graphs: List[rx.PyGraph], model_type: str = 'evolvegcn', 
                                  embedding_dim: int = 16, n_clusters: Optional[int] = None, 
                                  epochs: int = 100) -> Dict[str, Any]:
    """
    Run dynamic community detection using a dynamic GNN model
    
    Parameters:
    -----------
    graphs: list
        List of RustWorkX graphs for each time step
    model_type: str
        Type of dynamic GNN model ('evolvegcn' or 'dysat')
    embedding_dim: int
        Dimension of node embeddings
    n_clusters: int
        Number of clusters (communities)
    epochs: int
        Number of training epochs
        
    Returns:
    --------
    results: dict
        Results including embeddings, communities, and evaluation metrics
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric not available")
    
    # Train dynamic GNN model
    print(f"Training {model_type.upper()} model...")
    start_time = time.time()
    model, data_list = train_dynamic_gnn(model_type, graphs, embedding_dim, epochs)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Extract temporal embeddings
    print("Extracting temporal embeddings...")
    embeddings_list = extract_temporal_embeddings(model, data_list, model_type)
    
    # If n_clusters not specified, try to infer from ground truth
    if n_clusters is None:
        has_communities = False
        for i in range(len(graphs[0])):
            node_data = graphs[0].get_node_data(i)
            if node_data and 'community' in node_data:
                has_communities = True
                break
                
        if has_communities:
            # Extract communities from first graph
            communities = []
            for i in range(len(graphs[0])):
                node_data = graphs[0].get_node_data(i)
                if node_data and 'community' in node_data:
                    communities.append(node_data['community'])
                else:
                    communities.append(0)  # Default
            
            n_clusters = len(set(communities))
    
    # Detect communities
    print(f"Detecting communities with KMeans (n_clusters={n_clusters})...")
    communities_list = detect_temporal_communities(embeddings_list, n_clusters)
    
    # Visualize embeddings for each time step
    print("Visualizing node embeddings for each time step...")
    for t, embeddings in enumerate(embeddings_list):
        plot_embeddings(embeddings, communities_list[t], 
                      title=f"Time Step {t+1} - Node Embeddings")
    
    # Add communities to graphs
    for t, (G, communities) in enumerate(zip(graphs, communities_list)):
        add_communities_to_graph(G, communities, attr_name='detected_community')
    
    # Visualize detected communities
    print("Visualizing detected communities...")
    visualize_dynamic_communities(graphs, community_attr='detected_community',
                               title=f"{model_type.upper()} Detected Communities")
    
    # Visualize community evolution
    print("Visualizing community evolution...")
    visualize_community_evolution(communities_list, len(graphs[0]))
    
    # Evaluate against ground truth if available
    results = {
        'model_type': model_type,
        'embeddings_list': embeddings_list,
        'communities_list': communities_list,
        'training_time': training_time
    }
    
    # Check if any graph has community attribute
    has_ground_truth = False
    for G in graphs:
        for i in range(len(G)):
            node_data = G.get_node_data(i)
            if node_data and 'community' in node_data:
                has_ground_truth = True
                break
        if has_ground_truth:
            break
    
    if has_ground_truth:
        metrics_list = evaluate_temporal_communities(communities_list, graphs)
        results['metrics_list'] = metrics_list
        
        # Print average metrics
        avg_nmi = torch.tensor([m['nmi'] for m in metrics_list]).mean().item()
        avg_ari = torch.tensor([m['ari'] for m in metrics_list]).mean().item()
        
        print("\nEvaluation against ground truth:")
        print(f"Average NMI: {avg_nmi:.4f}")
        print(f"Average ARI: {avg_ari:.4f}")
        
        # Plot metrics over time
        plt.figure(figsize=(10, 6))
        time_steps = range(1, len(metrics_list) + 1)
        plt.plot(time_steps, [m['nmi'] for m in metrics_list], 'o-', label='NMI')
        plt.plot(time_steps, [m['ari'] for m in metrics_list], 's-', label='ARI')
        plt.title('Community Detection Performance Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return results


# Function to compare dynamic GNN models
def compare_dynamic_gnn_models(graphs: List[rx.PyGraph], embedding_dim: int = 16, 
                             n_clusters: Optional[int] = None, 
                             epochs: int = 100) -> pl.DataFrame:
    """
    Compare different dynamic GNN models for community detection
    
    Parameters:
    -----------
    graphs: list
        List of RustWorkX graphs for each time step
    embedding_dim: int
        Dimension of node embeddings
    n_clusters: int
        Number of clusters (communities)
    epochs: int
        Number of training epochs
        
    Returns:
    --------
    results_df: DataFrame
        DataFrame with comparison results
    """
    model_types = ['evolvegcn', 'dysat']
    results_list = []
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Running {model_type.upper()}")
        print(f"{'='*50}")
        
        try:
            results = run_dynamic_community_detection(graphs, model_type=model_type,
                                                   embedding_dim=embedding_dim,
                                                   n_clusters=n_clusters, epochs=epochs)
            
            # Calculate average metrics if available
            if 'metrics_list' in results:
                avg_nmi = torch.tensor([m['nmi'] for m in results['metrics_list']]).mean().item()
                avg_ari = torch.tensor([m['ari'] for m in results['metrics_list']]).mean().item()
            else:
                avg_nmi = float('nan')
                avg_ari = float('nan')
            
            # Collect results
            result_entry = {
                'Model': model_type.upper(),
                'Training Time (s)': results['training_time'],
                'Avg NMI': avg_nmi,
                'Avg ARI': avg_ari
            }
            
            results_list.append(result_entry)
            
        except Exception as e:
            print(f"Error running {model_type}: {e}")
    
    # Create DataFrame
    results_df = pl.DataFrame(results_list)
    
    # Visualize comparison
    plt.figure(figsize=(12, 6))
    
    # Plot metrics
    if 'Avg NMI' in results_df.columns and not results_df['Avg NMI'].is_null().all():
        plt.subplot(1, 2, 1)
        
        # Convert to pandas for plotting
        pd_df = results_df.to_pandas()
        pd_df.plot(x='Model', y=['Avg NMI', 'Avg ARI'], kind='bar', ax=plt.gca())
        plt.title('Average Community Detection Quality')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
    
    # Plot training time
    plt.subplot(1, 2, 2)
    pd_df = results_df.to_pandas()
    pd_df.plot(x='Model', y='Training Time (s)', kind='bar', ax=plt.gca(), color='green')
    plt.title('Training Time')
    plt.ylabel('Seconds')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df