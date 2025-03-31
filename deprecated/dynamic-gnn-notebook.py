# Dynamic Graph Neural Networks for Community Detection
# ====================================================

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# For PyTorch and PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_networkx, from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. Please install it for GNN-based methods.")

# For community evaluation
try:
    from cdlib import evaluation
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False

# Import utility functions from other notebooks
# In a real scenario, these would be imported from a module
from data_prep import (load_data, create_graph_from_edgelist, generate_synthetic_graph, plot_graph)
from gnn_community_detection import (nx_to_pyg, extract_embeddings, detect_communities_from_embeddings,
                                   evaluate_communities, plot_embeddings, add_communities_to_graph)


# Function to generate a sequence of dynamic graphs
def generate_dynamic_graphs(n_time_steps=5, n_nodes=100, n_communities=5, 
                            change_fraction=0.1, seed=42):
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
        List of NetworkX graphs for each time step
    """
    np.random.seed(seed)
    
    # Initialize graphs list
    graphs = []
    
    # Generate initial graph
    G_init, _ = generate_synthetic_graph('sbm', n_nodes=n_nodes, n_communities=n_communities,
                                       p_in=0.3, p_out=0.05)
    
    # Get initial community assignments
    community_assignments = np.array([G_init.nodes[i]['community'] for i in range(n_nodes)])
    
    # Add first graph to list
    graphs.append(G_init)
    
    # Generate subsequent graphs with evolving community structure
    for t in range(1, n_time_steps):
        # Decide which nodes change communities
        n_changes = int(n_nodes * change_fraction)
        change_indices = np.random.choice(n_nodes, n_changes, replace=False)
        
        # Update community assignments
        for idx in change_indices:
            # Assign a different community
            current_comm = community_assignments[idx]
            available_comms = [c for c in range(n_communities) if c != current_comm]
            new_comm = np.random.choice(available_comms)
            community_assignments[idx] = new_comm
        
        # Create probability matrix for SBM based on updated communities
        sizes = [np.sum(community_assignments == c) for c in range(n_communities)]
        p_in = 0.3  # probability within community
        p_out = 0.05  # probability between communities
        
        # Create probability matrix
        p_matrix = np.ones((n_communities, n_communities)) * p_out
        np.fill_diagonal(p_matrix, p_in)
        
        # Generate new graph
        G_t = nx.stochastic_block_model(sizes, p_matrix, seed=seed+t)
        
        # Add community assignments as node attributes
        node_index = 0
        for comm, size in enumerate(sizes):
            for _ in range(size):
                G_t.nodes[node_index]['community'] = comm
                node_index += 1
        
        # Add to graphs list
        graphs.append(G_t)
    
    return graphs


# Function to visualize dynamic communities
def visualize_dynamic_communities(graphs, community_attr='community', figsize=(15, 10)):
    """
    Visualize the evolution of communities in dynamic graphs
    
    Parameters:
    -----------
    graphs: list
        List of NetworkX graphs for each time step
    community_attr: str
        Node attribute for community assignment
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    n_time_steps = len(graphs)
    
    # Set up the figure
    fig, axes = plt.subplots(1, n_time_steps, figsize=figsize)
    if n_time_steps == 1:
        axes = [axes]
    
    # Visualize each time step
    for t, G in enumerate(graphs):
        # Get communities
        communities = [G.nodes[i][community_attr] for i in range(len(G))]
        
        # Calculate layout (use same layout for all time steps)
        if t == 0:
            pos = nx.spring_layout(G, seed=42)
        
        # Plot
        ax = axes[t]
        nx.draw_networkx(G, pos=pos, node_color=communities, cmap='rainbow',
                         with_labels=False, node_size=80, ax=ax)
        ax.set_title(f'Time Step {t+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# 1. EvolveGCN model
class EvolveGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(EvolveGCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Initial weights for the GCN layers
        self.weights = nn.ParameterList()
        
        # Input layer
        self.weights.append(nn.Parameter(torch.Tensor(input_dim, hidden_dim)))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.weights.append(nn.Parameter(torch.Tensor(hidden_dim, hidden_dim)))
        
        # Output layer
        self.weights.append(nn.Parameter(torch.Tensor(hidden_dim, output_dim)))
        
        # RNN for weight evolution
        self.weight_rnn = nn.GRUCell(hidden_dim * output_dim, hidden_dim * output_dim)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
    
    def forward(self, x, edge_index, prev_weights=None):
        # If no previous weights, use current weights
        if prev_weights is None:
            prev_weights = self.weights
        
        # Evolve weights if previous weights are provided
        else:
            # Evolve last layer weights (simplified for demonstration)
            last_weight = prev_weights[-1]
            flat_weight = last_weight.view(-1)
            evolved_weight = self.weight_rnn(flat_weight, flat_weight)
            self.weights[-1] = nn.Parameter(evolved_weight.view(self.hidden_dim, self.output_dim))
        
        # Apply GCN with evolved weights
        for i, weight in enumerate(self.weights[:-1]):
            x = F.relu(F.linear(x, weight))
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Output layer
        x = F.linear(x, self.weights[-1])
        
        return x


# 2. DySAT: Dynamic Self-Attention Network (simplified)
class DySAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=2):
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
        
    def forward(self, x, edge_index):
        # Apply structural attention
        for i, attention in enumerate(self.struct_attention[:-1]):
            x = attention(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Output layer
        x = self.struct_attention[-1](x, edge_index)
        
        return x


# Function to train dynamic GNN model on a sequence of graphs
def train_dynamic_gnn(model_type, graphs, embedding_dim=16, epochs=100, lr=0.01):
    """
    Train a dynamic GNN model on a sequence of graphs
    
    Parameters:
    -----------
    model_type: str
        Type of dynamic GNN model ('evolvegcn' or 'dysat')
    graphs: list
        List of NetworkX graphs for each time step
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
    
    # Convert graphs to PyG Data objects
    data_list = [nx_to_pyg(G) for G in graphs]
    
    # Get dimensions
    input_dim = data_list[0].x.size(1)
    
    # Initialize model
    if model_type.lower() == 'evolvegcn':
        model = EvolveGCN(input_dim, hidden_dim=32, output_dim=embedding_dim)
    elif model_type.lower() == 'dysat':
        model = DySAT(input_dim, hidden_dim=32, output_dim=embedding_dim)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Process each time step
        prev_weights = None
        total_loss = 0
        
        for t, data in enumerate(data_list):
            # Forward pass
            if model_type.lower() == 'evolvegcn':
                output = model(data.x, data.edge_index, prev_weights)
                prev_weights = model.weights
            else:
                output = model(data.x, data.edge_index)
            
            # If we have ground truth communities, use supervised loss
            if data.y is not None:
                loss = F.cross_entropy(output, data.y)
            else:
                # For unsupervised setting, use a simplified loss
                # In a real implementation, this would be more sophisticated
                loss = torch.tensor(0.0, requires_grad=True)
            
            total_loss += loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}')
    
    return model, data_list


# Function to extract temporal embeddings from trained dynamic GNN
def extract_temporal_embeddings(model, data_list, model_type='evolvegcn'):
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
    model.eval()
    embeddings_list = []
    prev_weights = None
    
    with torch.no_grad():
        for t, data in enumerate(data_list):
            # Forward pass
            if model_type.lower() == 'evolvegcn':
                embeddings = model(data.x, data.edge_index, prev_weights)
                prev_weights = model.weights
            else:
                embeddings = model(data.x, data.edge_index)
            
            # Convert to numpy
            embeddings_np = embeddings.detach().cpu().numpy()
            embeddings_list.append(embeddings_np)
    
    return embeddings_list


# Function to detect communities from temporal embeddings
def detect_temporal_communities(embeddings_list, n_clusters=None, method='kmeans'):
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
def evaluate_temporal_communities(communities_list, graphs, ground_truth_attr='community'):
    """
    Evaluate detected communities against ground truth for each time step
    
    Parameters:
    -----------
    communities_list: list
        List of community assignments for each time step
    graphs: list
        List of NetworkX graphs for each time step
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
        ground_truth = np.array([G.nodes[i][ground_truth_attr] for i in range(len(G))])
        
        # Evaluate
        metrics = evaluate_communities(communities, ground_truth)
        metrics_list.append(metrics)
    
    return metrics_list


# Function to visualize community evolution
def visualize_community_evolution(communities_list, n_nodes, figsize=(12, 8)):
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
    community_data = np.zeros((n_nodes, n_time_steps))
    
    for t, communities in enumerate(communities_list):
        community_data[:, t] = communities
    
    # Visualize
    plt.figure(figsize=figsize)
    plt.imshow(community_data, aspect='auto', cmap='rainbow')
    plt.colorbar(label='Community')
    plt.title('Community Evolution Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Node ID')
    plt.tight_layout()
    plt.show()


# Function to run dynamic community detection
def run_dynamic_community_detection(graphs, model_type='evolvegcn', embedding_dim=16, 
                                   n_clusters=None, epochs=100):
    """
    Run dynamic community detection using a dynamic GNN model
    
    Parameters:
    -----------
    graphs: list
        List of NetworkX graphs for each time step
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
    if n_clusters is None and 'community' in graphs[0].nodes[0]:
        communities = [graphs[0].nodes[i]['community'] for i in range(len(graphs[0]))]
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
        add_communities_to_graph(G, communities)
    
    # Visualize detected communities
    print("Visualizing detected communities...")
    visualize_dynamic_communities(graphs, community_attr='detected_community')
    
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
    
    if 'community' in graphs[0].nodes[0]:
        metrics_list = evaluate_temporal_communities(communities_list, graphs)
        results['metrics_list'] = metrics_list
        
        # Print average metrics
        avg_nmi = np.mean([m['nmi'] for m in metrics_list])
        avg_ari = np.mean([m['ari'] for m in metrics_list])
        
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
def compare_dynamic_gnn_models(graphs, embedding_dim=16, n_clusters=None, epochs=100):
    """
    Compare different dynamic GNN models for community detection
    
    Parameters:
    -----------
    graphs: list
        List of NetworkX graphs for each time step
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
                avg_nmi = np.mean([m['nmi'] for m in results['metrics_list']])
                avg_ari = np.mean([m['ari'] for m in results['metrics_list']])
            else:
                avg_nmi = np.nan
                avg_ari = np.nan
            
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
    results_df = pd.DataFrame(results_list)
    
    # Visualize comparison
    plt.figure(figsize=(12, 6))
    
    # Plot metrics
    if 'Avg NMI' in results_df.columns and not results_df['Avg NMI'].isna().all():
        plt.subplot(1, 2, 1)
        results_df.plot(x='Model', y=['Avg NMI', 'Avg ARI'], kind='bar', ax=plt.gca())
        plt.title('Average Community Detection Quality')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
    
    # Plot training time
    plt.subplot(1, 2, 2)
    results_df.plot(x='Model', y='Training Time (s)', kind='bar', ax=plt.gca(), color='green')
    plt.title('Training Time')
    plt.ylabel('Seconds')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df


# Main execution example
if __name__ == "__main__":
    print("Dynamic Graph Neural Networks for Community Detection")
    print("="*50)
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("Error: PyTorch Geometric not available. Please install it first.")
        import sys
        sys.exit(1)
    
    # Generate a sequence of dynamic graphs
    print("\n1. Generating a sequence of dynamic graphs...")
    n_time_steps = 5
    n_communities = 3
    graphs = generate_dynamic_graphs(n_time_steps=n_time_steps, n_nodes=100, 
                                    n_communities=n_communities, change_fraction=0.1)
    
    # Visualize original graphs with ground truth communities
    print("\n2. Visualizing ground truth communities over time...")
    visualize_dynamic_communities(graphs, community_attr='community')
    
    # Run dynamic community detection with a specific model
    print("\n3. Running EvolveGCN-based dynamic community detection...")
    results_evolvegcn = run_dynamic_community_detection(graphs, model_type='evolvegcn',
                                                      embedding_dim=16, n_clusters=n_communities,
                                                      epochs=50)
    
    # Compare different dynamic GNN models
    print("\n4. Comparing different dynamic GNN models...")
    comparison_results = compare_dynamic_gnn_models(graphs, embedding_dim=16,
                                                  n_clusters=n_communities, epochs=50)
    
    print("\n5. Summary of results:")
    print(comparison_results)
    
    print("\nAnalysis complete!")
