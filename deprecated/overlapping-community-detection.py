# Overlapping Community Detection Methods
# ======================================

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import time
import warnings
warnings.filterwarnings('ignore')

# For PyTorch and PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data
    from torch_geometric.utils import to_networkx, from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. Please install it for GNN-based methods.")

# For community evaluation
try:
    from cdlib import algorithms, evaluation
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False
    print("cdlib not available. Please install it for community detection algorithms.")

# Import utility functions from the data preparation notebook
# In a real scenario, these would be imported from a module
from data_prep import (load_data, create_graph_from_edgelist, generate_synthetic_graph, plot_graph)


# Function to generate a synthetic graph with overlapping communities
def generate_synthetic_overlapping_graph(n_nodes=100, n_communities=5, 
                                        overlap_size=10, p_in=0.3, p_out=0.05, seed=42):
    """
    Generate a synthetic graph with overlapping communities
    
    Parameters:
    -----------
    n_nodes: int
        Number of nodes
    n_communities: int
        Number of communities
    overlap_size: int
        Number of nodes that belong to multiple communities
    p_in: float
        Probability of edges within communities
    p_out: float
        Probability of edges between communities
    seed: int
        Random seed
        
    Returns:
    --------
    G: NetworkX Graph
        Graph with overlapping communities
    ground_truth: list of lists
        List of overlapping communities (each node can appear in multiple lists)
    """
    np.random.seed(seed)
    
    # Initialize communities
    sizes = [(n_nodes - overlap_size) // n_communities] * n_communities
    total_non_overlap = sum(sizes)
    
    # Create non-overlapping portion using SBM
    G = nx.stochastic_block_model(sizes, p_in * np.eye(n_communities) + p_out * (1 - np.eye(n_communities)), seed=seed)
    
    # Add overlapping nodes
    overlap_nodes = []
    for i in range(total_non_overlap, total_non_overlap + overlap_size):
        G.add_node(i)
        # Randomly assign to multiple communities
        n_assigned = np.random.randint(2, n_communities + 1)
        assigned_communities = np.random.choice(n_communities, n_assigned, replace=False)
        overlap_nodes.append((i, assigned_communities))
    
    # Add edges for overlapping nodes
    for node, communities in overlap_nodes:
        # For each community the node belongs to
        for comm in communities:
            # Connect to nodes in that community
            comm_nodes = [i for i in range(sizes[comm]) if i < total_non_overlap]
            for target in comm_nodes:
                if np.random.random() < p_in:
                    G.add_edge(node, target)
        
        # Connect to other overlapping nodes
        for other_node, other_comms in overlap_nodes:
            if node != other_node and any(c in other_comms for c in communities):
                if np.random.random() < p_in:
                    G.add_edge(node, other_node)
            else:
                if np.random.random() < p_out:
                    G.add_edge(node, other_node)
    
    # Create ground truth communities
    ground_truth = []
    node_offset = 0
    
    for i, size in enumerate(sizes):
        community = list(range(node_offset, node_offset + size))
        # Add overlapping nodes
        for node, communities in overlap_nodes:
            if i in communities:
                community.append(node)
        ground_truth.append(community)
        node_offset += size
    
    # Add ground truth to node attributes
    for i, comm_list in enumerate(ground_truth):
        for node in comm_list:
            if f'community_{i}' not in G.nodes[node]:
                G.nodes[node][f'community_{i}'] = 1
    
    return G, ground_truth


# Function to visualize overlapping communities
def plot_overlapping_communities(G, community_lists, pos=None, figsize=(12, 10), alpha=0.6):
    """
    Visualize overlapping communities in a graph
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph to visualize
    community_lists: list of lists
        List of overlapping communities
    pos: dict
        Node positions
    figsize: tuple
        Figure size
    alpha: float
        Transparency of node colors
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    # Plot graph structure
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Plot communities with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(community_lists)))
    
    # Create separate plots for each community
    plt.figure(figsize=figsize)
    
    # First draw all nodes in gray
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=100)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Then draw each community
    for i, (community, color) in enumerate(zip(community_lists, colors)):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=[color], alpha=alpha, 
                             node_size=100, label=f'Community {i+1}')
    
    plt.title('Overlapping Communities')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Visualize overlapping nodes differently
    plt.figure(figsize=figsize)
    
    # Count membership
    membership_count = {}
    for comm in community_lists:
        for node in comm:
            membership_count[node] = membership_count.get(node, 0) + 1
    
    # Node colors based on number of communities they belong to
    node_colors = [membership_count.get(node, 0) for node in G.nodes()]
    
    # Draw
    nx.draw_networkx(G, pos, node_color=node_colors, cmap=plt.cm.viridis, 
                    with_labels=True, node_size=100, font_size=8)
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), 
               label='Number of Communities')
    plt.title('Nodes Colored by Community Membership Count')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 1. BigCLAM algorithm for overlapping community detection
def run_bigclam(G, k=None):
    """
    Run the BigCLAM algorithm for overlapping community detection
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph to analyze
    k: int
        Number of communities
        
    Returns:
    --------
    communities: list of lists
        Detected overlapping communities
    execution_time: float
        Execution time in seconds
    """
    if not CDLIB_AVAILABLE:
        raise ImportError("cdlib is required for BigCLAM algorithm")
    
    start_time = time.time()
    
    if k is None:
        # Estimate number of communities
        try:
            louvain_communities = algorithms.louvain(G).communities
            k = len(louvain_communities)
        except:
            k = 5  # Default
    
    # Run BigCLAM
    bigclam_result = algorithms.big_clam(G, k)
    communities = bigclam_result.communities
    
    execution_time = time.time() - start_time
    return communities, execution_time


# 2. DEMON algorithm for overlapping community detection
def run_demon(G, epsilon=0.25):
    """
    Run the DEMON algorithm for overlapping community detection
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph to analyze
    epsilon: float
        Community threshold
        
    Returns:
    --------
    communities: list of lists
        Detected overlapping communities
    execution_time: float
        Execution time in seconds
    """
    if not CDLIB_AVAILABLE:
        raise ImportError("cdlib is required for DEMON algorithm")
    
    start_time = time.time()
    
    # Run DEMON
    demon_result = algorithms.demon(G, epsilon=epsilon)
    communities = demon_result.communities
    
    execution_time = time.time() - start_time
    return communities, execution_time


# 3. SLPA algorithm for overlapping community detection
def run_slpa(G, t=21, r=0.1):
    """
    Run the SLPA algorithm for overlapping community detection
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph to analyze
    t: int
        Number of iterations
    r: float
        Community threshold
        
    Returns:
    --------
    communities: list of lists
        Detected overlapping communities
    execution_time: float
        Execution time in seconds
    """
    if not CDLIB_AVAILABLE:
        raise ImportError("cdlib is required for SLPA algorithm")
    
    start_time = time.time()
    
    # Run SLPA
    slpa_result = algorithms.slpa(G, t=t, r=r)
    communities = slpa_result.communities
    
    execution_time = time.time() - start_time
    return communities, execution_time


# GNN-based overlapping community detection (simplified)
class GNN_Overlapping(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_communities, dropout=0.5):
        super(GNN_Overlapping, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_communities)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc(x)
        
        # For overlapping communities, we don't use softmax but sigmoid
        # This allows nodes to belong to multiple communities
        return torch.sigmoid(x)


# Function to convert NetworkX graph to PyTorch Geometric format for overlapping communities
def nx_to_pyg_overlapping(G, community_lists):
    """
    Convert NetworkX graph to PyTorch Geometric Data object for overlapping communities
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph to convert
    community_lists: list of lists
        List of overlapping communities
        
    Returns:
    --------
    data: torch_geometric.data.Data
        PyTorch Geometric Data object
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric not available")
    
    # Get edges
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add reverse edges for undirected
    
    # Node features (use degree as default feature)
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    x = torch.zeros((len(G), max_degree + 1), dtype=torch.float)
    for node_id, degree in degrees.items():
        x[node_id, degree] = 1.0
    
    # Create multi-hot encoding for overlapping community membership
    n_communities = len(community_lists)
    y = torch.zeros((len(G), n_communities), dtype=torch.float)
    
    for comm_idx, community in enumerate(community_lists):
        for node in community:
            y[node, comm_idx] = 1.0
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data


# Function to train GNN for overlapping community detection
def train_gnn_overlapping(model, data, epochs=100, lr=0.01, weight_decay=5e-4):
    """
    Train a GNN model for overlapping community detection
    
    Parameters:
    -----------
    model: torch.nn.Module
        GNN model
    data: torch_geometric.data.Data
        PyTorch Geometric data
    epochs: int
        Number of training epochs
    lr: float
        Learning rate
    weight_decay: float
        Weight decay for regularization
        
    Returns:
    --------
    model: torch.nn.Module
        Trained model
    losses: list
        List of training losses
    """
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Binary cross-entropy loss for multi-label classification
        loss = F.binary_cross_entropy(out, data.y)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model, losses


# Function to predict overlapping communities using GNN
def predict_gnn_overlapping(model, data, threshold=0.5):
    """
    Predict overlapping communities using a trained GNN model
    
    Parameters:
    -----------
    model: torch.nn.Module
        Trained GNN model
    data: torch_geometric.data.Data
        PyTorch Geometric data
    threshold: float
        Probability threshold for community membership
        
    Returns:
    --------
    communities: list of lists
        Predicted overlapping communities
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    
    # Convert to numpy
    probs = out.detach().cpu().numpy()
    
    # Apply threshold to determine community membership
    membership = (probs >= threshold).astype(int)
    
    # Convert to list of communities
    communities = []
    for j in range(membership.shape[1]):
        comm = [i for i in range(membership.shape[0]) if membership[i, j] == 1]
        if len(comm) > 0:  # Only add non-empty communities
            communities.append(comm)
    
    return communities


# Function to run GNN-based overlapping community detection
def run_gnn_overlapping(G, community_lists, hidden_dim=64, epochs=100, threshold=0.5):
    """
    Run GNN-based overlapping community detection
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph to analyze
    community_lists: list of lists
        List of overlapping communities (for training)
    hidden_dim: int
        Hidden dimension of GNN
    epochs: int
        Number of training epochs
    threshold: float
        Probability threshold for community membership
        
    Returns:
    --------
    communities: list of lists
        Detected overlapping communities
    execution_time: float
        Execution time in seconds
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric not available")
    
    start_time = time.time()
    
    # Convert to PyG format
    data = nx_to_pyg_overlapping(G, community_lists)
    
    # Initialize model
    n_communities = len(community_lists)
    model = GNN_Overlapping(data.x.size(1), hidden_dim, n_communities)
    
    # Train model
    model, losses = train_gnn_overlapping(model, data, epochs=epochs)
    
    # Predict communities
    communities = predict_gnn_overlapping(model, data, threshold=threshold)
    
    execution_time = time.time() - start_time
    return communities, execution_time, losses


# Function to evaluate overlapping communities
def evaluate_overlapping_communities(detected, ground_truth):
    """
    Evaluate detected overlapping communities against ground truth
    
    Parameters:
    -----------
    detected: list of lists
        Detected overlapping communities
    ground_truth: list of lists
        Ground truth overlapping communities
        
    Returns:
    --------
    metrics: dict
        Evaluation metrics
    """
    if not CDLIB_AVAILABLE:
        raise ImportError("cdlib is required for evaluation metrics")
    
    # Convert to cdlib format
    detected_obj = evaluation.NodeClustering(detected, None)
    ground_truth_obj = evaluation.NodeClustering(ground_truth, None)
    
    # Calculate overlap metrics
    metrics = {}
    
    # Standard metrics can still be used
    metrics['nmi'] = evaluation.overlapping_normalized_mutual_information(detected_obj, ground_truth_obj).score
    metrics['omega'] = evaluation.omega(detected_obj, ground_truth_obj).score
    
    # Overlapping specific metrics
    metrics['f1'] = evaluation.f1(detected_obj, ground_truth_obj).score
    
    return metrics


# Function to compare overlapping community detection methods
def compare_overlapping_methods(G, ground_truth):
    """
    Compare different overlapping community detection methods
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph to analyze
    ground_truth: list of lists
        Ground truth overlapping communities
        
    Returns:
    --------
    results: DataFrame
        Comparison results
    """
    methods = {
        'BigCLAM': lambda g: run_bigclam(g, k=len(ground_truth)),
        'DEMON': run_demon,
        'SLPA': run_slpa
    }
    
    results = []
    
    for method_name, method_func in methods.items():
        print(f"Running {method_name}...")
        try:
            communities, execution_time = method_func(G)
            
            # Evaluate
            metrics = evaluate_overlapping_communities(communities, ground_truth)
            
            # Store results
            results.append({
                'Method': method_name,
                'Num Communities': len(communities),
                'Avg Community Size': np.mean([len(comm) for comm in communities]),
                'NMI': metrics['nmi'],
                'Omega': metrics['omega'],
                'F1': metrics['f1'],
                'Execution Time (s)': execution_time
            })
            
            # Visualize detected communities
            print(f"Visualizing communities detected by {method_name}...")
            plot_overlapping_communities(G, communities, 
                                       figsize=(12, 10),
                                       alpha=0.6)
            
        except Exception as e:
            print(f"Error running {method_name}: {e}")
    
    # Add GNN-based method if possible
    if TORCH_GEOMETRIC_AVAILABLE:
        print("Running GNN-based method...")
        try:
            # For GNN, we need to provide ground truth for training
            # In a real scenario, you would split the data and evaluate properly
            communities, execution_time, _ = run_gnn_overlapping(G, ground_truth, epochs=50)
            
            # Evaluate
            metrics = evaluate_overlapping_communities(communities, ground_truth)
            
            # Store results
            results.append({
                'Method': 'GNN',
                'Num Communities': len(communities),
                'Avg Community Size': np.mean([len(comm) for comm in communities]),
                'NMI': metrics['nmi'],
                'Omega': metrics['omega'],
                'F1': metrics['f1'],
                'Execution Time (s)': execution_time
            })
            
            # Visualize detected communities
            print("Visualizing communities detected by GNN...")
            plot_overlapping_communities(G, communities, 
                                       figsize=(12, 10),
                                       alpha=0.6)
        except Exception as e:
            print(f"Error running GNN-based method: {e}")
    
    return pd.DataFrame(results)


# Main execution
if __name__ == "__main__":
    print("Overlapping Community Detection Methods")
    print("="*50)
    
    # Generate a synthetic graph with overlapping communities
    print("\n1. Generating a synthetic graph with overlapping communities...")
    n_communities = 4
    G, ground_truth = generate_synthetic_overlapping_graph(
        n_nodes=100, n_communities=n_communities, overlap_size=20)
    
    # Visualize ground truth communities
    print("\n2. Visualizing ground truth overlapping communities...")
    plot_overlapping_communities(G, ground_truth)
    
    # Compare different overlapping community detection methods
    print("\n3. Comparing different overlapping community detection methods...")
    results = compare_overlapping_methods(G, ground_truth)
    
    # Display results
    print("\n4. Results summary:")
    print(results)
    
    # Visualize comparison
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    plt.subplot(2, 1, 1)
    results.plot(x='Method', y=['NMI', 'Omega', 'F1'], kind='bar', ax=plt.gca())
    plt.title('Quality Metrics by Method')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    
    # Plot execution time
    plt.subplot(2, 1, 2)
    results.plot(x='Method', y='Execution Time (s)', kind='bar', ax=plt.gca(), color='green')
    plt.title('Execution Time by Method')
    plt.ylabel('Time (seconds)')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # If GNN is available, show learning curve
    if TORCH_GEOMETRIC_AVAILABLE:
        print("\n5. Training GNN model with learning curve...")
        _, _, losses = run_gnn_overlapping(G, ground_truth, epochs=100)
        
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('GNN Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    print("\nAnalysis complete!")
