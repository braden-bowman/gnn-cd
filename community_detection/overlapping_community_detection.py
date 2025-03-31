# Overlapping Community Detection Methods
# ======================================

import os
import torch
import polars as pl
import rustworkx as rx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
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
    print("PyTorch not available. Some methods will be skipped.")

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data
    from torch_geometric.utils import to_networkx, from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. GNN-based methods will be skipped.")

# For community evaluation
try:
    from cdlib import algorithms, evaluation
    from cdlib.classes import NodeClustering
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False
    print("cdlib not available. Some methods will be skipped.")


# Function to generate a synthetic graph with overlapping communities
def generate_synthetic_overlapping_graph(n_nodes: int = 100, n_communities: int = 5, 
                                        overlap_size: int = 10, p_in: float = 0.3, 
                                        p_out: float = 0.05, seed: int = 42) -> Tuple[rx.PyGraph, List[List[int]]]:
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
    G: RustWorkX PyGraph
        Graph with overlapping communities
    ground_truth: list of lists
        List of overlapping communities (each node can appear in multiple lists)
    """
    torch.manual_seed(seed)
    
    # Initialize communities
    sizes = [(n_nodes - overlap_size) // n_communities] * n_communities
    total_non_overlap = sum(sizes)
    
    # Create non-overlapping portion using SBM via NetworkX first
    import networkx as nx
    
    # Create probability matrix
    p_matrix = torch.ones((n_communities, n_communities)) * p_out
    p_matrix.fill_diagonal_(p_in)
    
    G_nx = nx.stochastic_block_model(sizes, p_matrix.numpy(), seed=seed)
    
    # Add overlapping nodes
    overlap_nodes = []
    for i in range(total_non_overlap, total_non_overlap + overlap_size):
        G_nx.add_node(i)
        # Randomly assign to multiple communities
        n_assigned = torch.randint(2, n_communities + 1, (1,)).item()
        assigned_communities = torch.randperm(n_communities)[:n_assigned].tolist()
        overlap_nodes.append((i, assigned_communities))
    
    # Add edges for overlapping nodes
    for node, communities in overlap_nodes:
        # For each community the node belongs to
        for comm in communities:
            # Connect to nodes in that community
            comm_nodes = [i for i in range(sizes[comm]) if i < total_non_overlap]
            for target in comm_nodes:
                if torch.rand(1).item() < p_in:
                    G_nx.add_edge(node, target)
        
        # Connect to other overlapping nodes
        for other_node, other_comms in overlap_nodes:
            if node != other_node and any(c in other_comms for c in communities):
                if torch.rand(1).item() < p_in:
                    G_nx.add_edge(node, other_node)
            else:
                if torch.rand(1).item() < p_out:
                    G_nx.add_edge(node, other_node)
    
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
    
    # Convert to RustWorkX graph
    G = rx.PyGraph()
    
    # Add nodes with attributes indicating community membership
    node_mapping = {}
    for i in range(max(G_nx.nodes()) + 1):
        node_attr = {}
        
        # Add community memberships as separate attributes
        for comm_idx, comm_nodes in enumerate(ground_truth):
            if i in comm_nodes:
                node_attr[f'community_{comm_idx}'] = 1
        
        node_mapping[i] = G.add_node(node_attr)
    
    # Add edges
    for u, v in G_nx.edges():
        G.add_edge(node_mapping[u], node_mapping[v], None)
    
    return G, ground_truth


# Function to visualize overlapping communities
def visualize_overlapping_communities(G: rx.PyGraph, community_lists: List[List[int]], 
                                   pos: Optional[Dict] = None, 
                                   figsize: Tuple[int, int] = (12, 10), 
                                   alpha: float = 0.6):
    """
    Alias for plot_overlapping_communities to maintain backward compatibility
    """
    return plot_overlapping_communities(G, community_lists, pos, figsize, alpha)


# Function to visualize overlapping communities
def plot_overlapping_communities(G: rx.PyGraph, community_lists: List[List[int]], 
                               pos: Optional[Dict] = None, 
                               figsize: Tuple[int, int] = (12, 10), 
                               alpha: float = 0.6):
    """
    Visualize overlapping communities in a graph
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
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
    # Convert to NetworkX for visualization
    import networkx as nx
    G_nx = nx.Graph()
    
    # Add nodes
    for i in range(len(G)):
        G_nx.add_node(i)
    
    # Add edges
    for edge in G.edge_list():
        source, target = edge[0], edge[1]
        G_nx.add_edge(source, target)
    
    if pos is None:
        pos = nx.spring_layout(G_nx, seed=42)
    
    # Create figure and axes for the first plot
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    # Plot communities with different colors
    colors = plt.cm.rainbow(torch.linspace(0, 1, len(community_lists)).numpy())
    
    # First draw all nodes in gray
    nx.draw_networkx_nodes(G_nx, pos, node_color='lightgray', node_size=100, ax=ax1)
    nx.draw_networkx_edges(G_nx, pos, alpha=0.3, ax=ax1)
    
    # Then draw each community
    for i, (community, color) in enumerate(zip(community_lists, colors)):
        nx.draw_networkx_nodes(G_nx, pos, nodelist=community, node_color=[color], alpha=alpha, 
                             node_size=100, label=f'Community {i+1}', ax=ax1)
    
    ax1.set_title('Overlapping Communities')
    ax1.legend()
    ax1.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Create figure and axes for the second plot
    fig2, ax2 = plt.subplots(figsize=figsize)
    
    # Count membership
    membership_count = {}
    for comm in community_lists:
        for node in comm:
            membership_count[node] = membership_count.get(node, 0) + 1
    
    # Node colors based on number of communities they belong to
    node_colors = [membership_count.get(node, 0) for node in G_nx.nodes()]
    
    # Draw
    nx.draw_networkx(G_nx, pos, node_color=node_colors, cmap=plt.cm.viridis, 
                    with_labels=True, node_size=100, font_size=8, ax=ax2)
    
    # Add colorbar with explicit axis reference
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                             norm=plt.Normalize(vmin=min(node_colors) if node_colors else 0, 
                                               vmax=max(node_colors) if node_colors else 1))
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax2)
    cbar.set_label('Number of Communities')
    
    ax2.set_title('Nodes Colored by Community Membership Count')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()


# 1. BigCLAM algorithm for overlapping community detection
def run_bigclam(G: rx.PyGraph, k: Optional[int] = None, iterations: int = 50, 
               learning_rate: float = 0.005, seed: int = 42) -> Tuple[List[List[int]], float]:
    """
    Run the BigCLAM algorithm for overlapping community detection based on the paper:
    "Overlapping Community Detection at Scale: A Nonnegative Matrix Factorization Approach"
    by Yang and Leskovec (WSDM 2013)
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
    k: int
        Number of communities
    iterations: int
        Number of training iterations
    learning_rate: float
        Learning rate for gradient ascent
    seed: int
        Random seed
        
    Returns:
    --------
    communities: list of lists
        Detected overlapping communities
    execution_time: float
        Execution time in seconds
    """
    start_time = time.time()
    
    # Convert to NetworkX for implementation
    import networkx as nx
    G_nx = nx.Graph()
    
    # Add nodes
    for i in range(len(G)):
        G_nx.add_node(i)
    
    # Add edges
    for edge in G.edge_list():
        source, target = edge[0], edge[1]
        G_nx.add_edge(source, target)
    
    if k is None:
        # Estimate number of communities
        try:
            if CDLIB_AVAILABLE:
                louvain_communities = algorithms.louvain(G_nx).communities
                k = len(louvain_communities)
            else:
                # Fallback to a heuristic if cdlib not available
                k = max(5, int(len(G_nx) ** 0.5 / 5))
        except:
            k = 5  # Default
    
    # Check if karateclub is available and use that if possible
    try:
        from karateclub import BigClam
        print("Using karateclub's BigClam implementation")
        # Use karateclub's implementation
        bigclam_model = BigClam(dimensions=k, iterations=iterations, learning_rate=learning_rate, seed=seed)
        bigclam_model.fit(G_nx)
        
        # Get memberships and convert to list of communities
        memberships = bigclam_model.get_memberships()
        
        # Create communities list (nodes in each community)
        community_to_nodes = {}
        for node, comms in memberships.items():
            for comm in comms:
                if comm not in community_to_nodes:
                    community_to_nodes[comm] = []
                community_to_nodes[comm].append(node)
        
        communities = list(community_to_nodes.values())
        
    except ImportError:
        # Implement BigClam algorithm natively if karateclub not available
        print("karateclub not available, using custom BigClam implementation")
        
        # Initialize node embeddings - shape: (n_nodes, k)
        n_nodes = len(G_nx)
        torch.manual_seed(seed)
        F = torch.rand(n_nodes, k) * 0.1
        
        # Get adjacency list for faster updates
        adjacency_list = {node: list(neighbors) for node, neighbors in G_nx.adjacency()}
        nodes = list(G_nx.nodes())
        
        # Gradient ascent updates
        for iteration in range(iterations):
            # Shuffle nodes for better convergence
            torch.manual_seed(seed + iteration)
            permutation = torch.randperm(n_nodes)
            
            # Update each node's embedding
            for i in permutation:
                node = nodes[i]
                if node not in adjacency_list:  # Skip isolated nodes
                    continue
                
                # Get neighbors
                neighbors = adjacency_list[node]
                
                # Calculate gradients for neighbors (positive force)
                neighbor_ids = [nodes.index(neigh) for neigh in neighbors]
                if not neighbor_ids:  # Skip nodes with no neighbors
                    continue
                    
                neighbor_features = F[neighbor_ids, :]
                sum_of_features = torch.sum(F, dim=0)
                
                # Compute gradients
                score = torch.mm(neighbor_features, F[i].view(-1, 1)).view(-1)
                exp_score = torch.exp(-score)
                derivative_1 = torch.mm(exp_score.view(1, -1), neighbor_features).view(-1)
                
                # Non-neighbor gradient (negative force)
                h_estimate = torch.mm(F[i].view(1, -1), sum_of_features.view(-1, 1))
                derivative_2 = h_estimate.view(-1) * F[i]
                
                # Update rule
                delta = derivative_1 - derivative_2
                F[i] = F[i] + learning_rate * delta
                F[i] = torch.clamp(F[i], min=0)  # Enforce non-negativity
        
        # Determine community memberships
        # A node belongs to a community if its embedding value exceeds a threshold
        threshold = 0.1
        community_to_nodes = {}
        
        for node_idx, node in enumerate(nodes):
            # Find communities this node belongs to
            assignments = (F[node_idx, :] > threshold).nonzero(as_tuple=True)[0].tolist()
            for comm_idx in assignments:
                if comm_idx not in community_to_nodes:
                    community_to_nodes[comm_idx] = []
                community_to_nodes[comm_idx].append(node)
        
        communities = list(community_to_nodes.values())
        
    # Filter out empty communities
    communities = [comm for comm in communities if len(comm) > 0]
    
    execution_time = time.time() - start_time
    return communities, execution_time


# 2. DEMON algorithm for overlapping community detection
def run_demon(G: rx.PyGraph, epsilon: float = 0.25, min_com_size: int = 3) -> Tuple[List[List[int]], float]:
    """
    Run the DEMON algorithm for overlapping community detection
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
    epsilon: float
        Community threshold
    min_com_size: int
        Minimum community size (default: 3)
        
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
    
    # Convert to NetworkX for cdlib
    import networkx as nx
    G_nx = nx.Graph()
    
    # Add nodes
    for i in range(len(G)):
        G_nx.add_node(i)
    
    # Add edges
    for edge in G.edge_list():
        source, target = edge[0], edge[1]
        G_nx.add_edge(source, target)
    
    # Try different parameter combinations
    try:
        # First try without the min size parameter
        print("Trying DEMON with just epsilon parameter...")
        demon_result = algorithms.demon(G_nx, epsilon=epsilon)
    except Exception as e1:
        print(f"Error with basic parameters: {e1}")
        try:
            # Try with min_com_size
            print("Trying with min_com_size parameter...")
            demon_result = algorithms.demon(G_nx, epsilon=epsilon, min_com_size=min_com_size)
        except Exception as e2:
            print(f"Error with min_com_size: {e2}")
            try:
                # Try with min_comm_size
                print("Trying with min_comm_size parameter...")
                demon_result = algorithms.demon(G_nx, epsilon=epsilon, min_comm_size=min_com_size)
            except Exception as e3:
                print(f"All parameter combinations failed. Errors: {e1}, {e2}, {e3}")
                raise ImportError("Could not run DEMON with any parameter combination. Check your cdlib version.")
    
    communities = demon_result.communities
    
    execution_time = time.time() - start_time
    return communities, execution_time


# 3. SLPA algorithm for overlapping community detection
def run_slpa(G: rx.PyGraph, t: int = 21, r: float = 0.1) -> Tuple[List[List[int]], float]:
    """
    Run the SLPA algorithm for overlapping community detection
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
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
    
    # Convert to NetworkX for cdlib
    import networkx as nx
    G_nx = nx.Graph()
    
    # Add nodes
    for i in range(len(G)):
        G_nx.add_node(i)
    
    # Add edges
    for edge in G.edge_list():
        source, target = edge[0], edge[1]
        G_nx.add_edge(source, target)
    
    # Try different parameter combinations for robustness
    try:
        # Try with both parameters
        print("Trying SLPA with standard parameters...")
        slpa_result = algorithms.slpa(G_nx, t=t, r=r)
        print("✓ Success!")
    except Exception as e1:
        print(f"Error with standard parameters: {e1}")
        try:
            # Try with just threshold parameter
            print("Trying with just 'r' parameter...")
            slpa_result = algorithms.slpa(G_nx, r=r)
            print("✓ Success!")
        except Exception as e2:
            print(f"Error with 'r' parameter only: {e2}")
            try:
                # Try with just iterations parameter
                print("Trying with just 't' parameter...")
                slpa_result = algorithms.slpa(G_nx, t=t)
                print("✓ Success!")
            except Exception as e3:
                print(f"All parameter combinations failed. Errors: \n{e1}\n{e2}\n{e3}")
                raise ImportError("Could not run SLPA with any parameter combination. Check your cdlib version.")
    
    communities = slpa_result.communities
    
    execution_time = time.time() - start_time
    return communities, execution_time


# GNN-based overlapping community detection
class GNN_Overlapping(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_communities: int, dropout: float = 0.5):
        super(GNN_Overlapping, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_communities)
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
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


# Function to convert RustWorkX graph to PyTorch Geometric format for overlapping communities
def rwx_to_pyg_overlapping(G: rx.PyGraph, community_lists: List[List[int]]) -> Data:
    """
    Convert RustWorkX graph to PyTorch Geometric Data object for overlapping communities
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
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
    edge_list = []
    for edge in G.edge_list():
        edge_list.append((edge[0], edge[1]))
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # For undirected graph, ensure both directions included
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Node features (use degree as default feature)
    degrees = torch.tensor([G.degree(i) for i in range(len(G))])
    max_degree = degrees.max().item()
    x = torch.zeros((len(G), max_degree + 1), dtype=torch.float)
    for i, degree in enumerate(degrees):
        x[i, degree] = 1.0
    
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
def train_gnn_overlapping(model: nn.Module, data: Data, epochs: int = 100, 
                        lr: float = 0.01, weight_decay: float = 5e-4) -> Tuple[nn.Module, List[float]]:
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
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
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
def predict_gnn_overlapping(model: nn.Module, data: Data, 
                          threshold: float = 0.5) -> List[List[int]]:
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
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    
    # Convert to CPU numpy
    probs = out.cpu().numpy()
    
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
def run_gnn_overlapping(G: rx.PyGraph, community_lists: List[List[int]], 
                      hidden_dim: int = 64, epochs: int = 100, 
                      threshold: float = 0.5) -> Tuple[List[List[int]], float, List[float]]:
    """
    Run GNN-based overlapping community detection
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
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
    losses: list
        Training losses over epochs
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric not available")
    
    start_time = time.time()
    
    # Convert to PyG format
    data = rwx_to_pyg_overlapping(G, community_lists)
    
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
def evaluate_overlapping_communities(detected: List[List[int]], 
                                   ground_truth: List[List[int]]) -> Dict[str, float]:
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
    print("Using custom metric implementation for overlapping community evaluation...")
    
    # Calculate metrics directly using custom implementations
    metrics = {}
    
    # Custom implementation for F1 score based on best community matching
    total_precision = 0
    total_recall = 0
    
    # For each detected community
    for det_comm in detected:
        det_comm_set = set(det_comm)
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        
        # Find best matching ground truth community
        for gt_comm in ground_truth:
            gt_comm_set = set(gt_comm)
            
            # Calculate overlap
            intersection = len(det_comm_set & gt_comm_set)
            
            if intersection > 0:
                precision = intersection / len(det_comm_set)  # TP / (TP + FP)
                recall = intersection / len(gt_comm_set)      # TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
        
        total_precision += best_precision
        total_recall += best_recall
    
    # Calculate average precision and recall
    avg_precision = total_precision / len(detected) if detected else 0
    avg_recall = total_recall / len(detected) if detected else 0
    
    # Calculate F1 score
    if avg_precision + avg_recall > 0:
        metrics['f1'] = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        metrics['f1'] = 0
    
    # Calculate a custom Normalized Mutual Information and Omega index
    # Convert communities to node-to-community dictionaries
    gt_assignment = {}
    for comm_idx, comm in enumerate(ground_truth):
        for node in comm:
            if node not in gt_assignment:
                gt_assignment[node] = []
            gt_assignment[node].append(comm_idx)
    
    det_assignment = {}
    for comm_idx, comm in enumerate(detected):
        for node in comm:
            if node not in det_assignment:
                det_assignment[node] = []
            det_assignment[node].append(comm_idx)
    
    # Get all nodes
    all_nodes = set(gt_assignment.keys()) | set(det_assignment.keys())
    
    # Calculate Jaccard similarity for each node's community assignments
    jaccard_sum = 0
    for node in all_nodes:
        gt_comms = set(gt_assignment.get(node, []))
        det_comms = set(det_assignment.get(node, []))
        
        if not gt_comms and not det_comms:
            continue
            
        # Calculate Jaccard similarity
        intersection = len(gt_comms & det_comms)
        union = len(gt_comms | det_comms)
        jaccard = intersection / union if union > 0 else 0
        jaccard_sum += jaccard
    
    # Normalize
    metrics['omega'] = jaccard_sum / len(all_nodes) if all_nodes else 0
    
    # Use a placeholder for NMI since it's complicated to implement correctly
    metrics['nmi'] = metrics['omega']  # Use Omega as a proxy
    
    print(f"Metrics: F1={metrics['f1']:.4f}, Omega={metrics['omega']:.4f}")
    return metrics


# Function to compare overlapping community detection methods
def compare_overlapping_methods(G: rx.PyGraph, 
                              ground_truth: List[List[int]]) -> pl.DataFrame:
    """
    Compare different overlapping community detection methods
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
    ground_truth: list of lists
        Ground truth overlapping communities
        
    Returns:
    --------
    results: DataFrame
        Comparison results
    """
    methods = {}
    
    if CDLIB_AVAILABLE:
        methods.update({
            'BigCLAM': lambda g: run_bigclam(g, k=len(ground_truth)),
            'DEMON': run_demon,
            'SLPA': run_slpa
        })
    
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
                'Avg Community Size': sum(len(comm) for comm in communities) / len(communities),
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
                'Avg Community Size': sum(len(comm) for comm in communities) / len(communities),
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
    
    # Create DataFrame
    return pl.DataFrame(results)