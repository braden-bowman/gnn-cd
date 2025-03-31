# Data Preparation and Graph Construction for Community Detection
# ===============================================================

import torch
import polars as pl
import rustworkx as rx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import warnings
warnings.filterwarnings('ignore')

# For GPU acceleration
try:
    import cudf
    import cugraph
    CUGRAPH_AVAILABLE = True
except ImportError:
    CUGRAPH_AVAILABLE = False

# Optional imports for visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    
# For graph neural networks
try:
    import torch_geometric
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

# For community detection baselines
try:
    from cdlib import algorithms
    from cdlib.classes import NodeClustering
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False


# Helper functions for data loading
def load_data(filepath: str, filetype: str = 'parquet') -> Union[pl.DataFrame, Dict]:
    """
    Load data from various file formats
    
    Parameters:
    -----------
    filepath: str
        Path to the data file
    filetype: str
        Type of file (parquet, csv, json, pickle, etc.)
        
    Returns:
    --------
    data: DataFrame or dict
        Loaded data
    """
    if filetype.lower() == 'parquet':
        return pl.read_parquet(filepath)
    elif filetype.lower() == 'csv':
        # Use lazy evaluation with scan_csv
        return pl.scan_csv(filepath).collect()
    elif filetype.lower() == 'json':
        return pl.read_json(filepath)
    elif filetype.lower() == 'pickle':
        with open(filepath, 'rb') as f:
            import pickle
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {filetype}")


def create_graph_from_edgelist(edgelist: pl.DataFrame, directed: bool = False, weighted: bool = False) -> rx.PyGraph:
    """
    Create a RustWorkX graph from an edge list
    
    Parameters:
    -----------
    edgelist: pl.DataFrame
        Polars DataFrame with edge information (source, target, [weight])
    directed: bool
        Whether the graph is directed
    weighted: bool
        Whether the graph has edge weights
        
    Returns:
    --------
    G: RustWorkX PyGraph or PyDiGraph
        The constructed graph
    """
    # Create the appropriate graph type
    if directed:
        G = rx.PyDiGraph()
    else:
        G = rx.PyGraph()
    
    # Extract unique nodes using Polars
    if hasattr(edgelist, 'collect'):
        # Handle LazyFrame
        df = edgelist.collect()
    else:
        # Regular DataFrame
        df = edgelist
    
    # Get unique nodes using Polars operations
    source_nodes = df.select('source').unique().to_series().to_list()
    target_nodes = df.select('target').unique().to_series().to_list() 
    all_nodes = list(set(source_nodes + target_nodes))
    
    # Add nodes to the graph and create mapping
    node_mapping = {}
    for node in all_nodes:
        node_idx = G.add_node(node)
        node_mapping[node] = node_idx
    
    # Extract edge data as native Python lists
    source_list = df['source'].to_list()
    target_list = df['target'].to_list()
    
    # Add edges
    if weighted and 'weight' in df.columns:
        weight_list = df['weight'].to_list()
        for src, tgt, wt in zip(source_list, target_list, weight_list):
            try:
                G.add_edge(node_mapping[src], node_mapping[tgt], {'weight': wt})
            except KeyError as e:
                print(f"KeyError: {e}, src type: {type(src)}, value: {src}")
                print(f"node_mapping keys: {[type(k) for k in node_mapping.keys()][:5]}...")
                raise
    else:
        for src, tgt in zip(source_list, target_list):
            G.add_edge(node_mapping[src], node_mapping[tgt], None)
    
    return G


def create_graph_from_adjacency(adjacency: torch.Tensor, 
                              node_features: Optional[torch.Tensor] = None, 
                              node_labels: Optional[torch.Tensor] = None) -> rx.PyGraph:
    """
    Create a RustWorkX graph from an adjacency matrix
    
    Parameters:
    -----------
    adjacency: torch.Tensor
        Adjacency matrix
    node_features: torch.Tensor
        Matrix of node features
    node_labels: torch.Tensor
        Array of node labels (for ground truth communities)
        
    Returns:
    --------
    G: RustWorkX PyGraph
        The constructed graph
    """
    # Create an undirected graph
    G = rx.PyGraph()
    
    # Convert adjacency matrix to torch tensor if it's not already
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.tensor(adjacency)
    
    # Add nodes with features and labels if provided
    n_nodes = adjacency.shape[0]
    node_indices = []
    
    for i in range(n_nodes):
        node_data = {}
        
        # Add node features if provided
        if node_features is not None:
            node_data['features'] = node_features[i].detach().cpu().numpy()
        
        # Add node labels if provided
        if node_labels is not None:
            node_data['community'] = int(node_labels[i].item())
        
        # Add node to graph
        node_indices.append(G.add_node(node_data))
    
    # Efficiently get edge indices using torch operations
    if torch.cuda.is_available():
        adjacency = adjacency.cuda()
    
    # Get edge indices from adjacency matrix
    edge_indices = torch.nonzero(adjacency > 0).cpu().numpy()
    
    # Add edges based on adjacency matrix
    for i, j in edge_indices:
        if i < j:  # Upper triangular part for undirected graph
            weight = {'weight': float(adjacency[i, j].item())} if adjacency[i, j] != 1 else None
            G.add_edge(node_indices[i], node_indices[j], weight)
    
    return G


def convert_rustworkx_to_pytorch_geometric(G: rx.PyGraph, 
                                         node_features: Optional[torch.Tensor] = None, 
                                         node_labels: Optional[torch.Tensor] = None) -> Data:
    """
    Convert a RustWorkX graph to a PyTorch Geometric Data object
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to convert
    node_features: torch.Tensor
        Matrix of node features (optional if already in G)
    node_labels: torch.Tensor
        Array of node labels (optional if already in G)
        
    Returns:
    --------
    data: torch_geometric.data.Data
        PyTorch Geometric Data object
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric is not available")
    
    num_nodes = len(G)
    
    # Get edges as tensor
    edge_list = []
    for edge in G.edge_list():
        edge_list.append((edge[0], edge[1]))
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # For undirected graph, ensure both directions included
        if isinstance(G, rx.PyGraph):
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Get node features
    if node_features is not None:
        x = node_features
    else:
        # Try to get features from graph nodes
        node_feature_list = []
        for i in range(num_nodes):
            node_data = G.get_node_data(i)
            if node_data and 'features' in node_data:
                node_feature_list.append(torch.tensor(node_data['features'], dtype=torch.float))
        
        if node_feature_list:
            x = torch.stack(node_feature_list)
        else:
            # Use one-hot degree as default feature
            node_degrees = [G.degree(i) for i in range(num_nodes)]
            max_degree = max(node_degrees) if node_degrees else 0
            x = torch.zeros((num_nodes, max_degree + 1))
            for i, degree in enumerate(node_degrees):
                x[i, degree] = 1
    
    # Get labels
    if node_labels is not None:
        y = node_labels
    else:
        # Try to get labels from graph nodes
        node_label_list = []
        for i in range(num_nodes):
            node_data = G.get_node_data(i)
            if node_data and 'community' in node_data:
                node_label_list.append(node_data['community'])
        
        if node_label_list:
            y = torch.tensor(node_label_list, dtype=torch.long)
        else:
            y = None
    
    # Extract edge weights if available
    edge_attr = None
    if G.num_edges() > 0:
        edge_weights = []
        for i, edge in enumerate(G.edge_list()):
            edge_data = G.get_edge_data(edge[0], edge[1])
            if edge_data and 'weight' in edge_data:
                edge_weights.append(edge_data['weight'])
            else:
                edge_weights.append(1.0)  # Default weight
                
        # For undirected graph, duplicate weights for both directions
        if isinstance(G, rx.PyGraph):
            edge_weights = edge_weights + edge_weights
            
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data


def plot_graph(G: rx.PyGraph, community_attr: Optional[str] = None, 
             pos: Optional[Dict] = None, figsize: Tuple[int, int] = (10, 8), 
             title: str = "Graph Visualization"):
    """
    Plot a RustWorkX graph with community colors
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to plot
    community_attr: str
        Node attribute name for community labels
    pos: dict
        Node positions for visualization
    figsize: tuple
        Figure size
    title: str
        Plot title
        
    Returns:
    --------
    None
    """
    from typing import Dict
    plt.figure(figsize=figsize)
    
    # Convert to NetworkX for visualization
    import networkx as nx
    G_nx = nx.Graph()
    
    # First, add all nodes without attributes
    for i in range(len(G)):
        G_nx.add_node(i)
    
    # Then, add node attributes separately
    for i in range(len(G)):
        node_data = G.get_node_data(i)
        if node_data is None:
            continue
        
        # Handle different node data types
        if isinstance(node_data, dict):
            # It's already a dictionary, add each key-value pair as an attribute
            for key, value in node_data.items():
                G_nx.nodes[i][key] = value
        else:
            # It's a primitive type, store as 'value' attribute
            G_nx.nodes[i]['value'] = node_data
            # If this is the community attribute, make it accessible
            if community_attr == 'value':
                G_nx.nodes[i][community_attr] = node_data
    
    # Add edges without attributes first
    for edge in G.edge_list():
        source, target = edge[0], edge[1]
        G_nx.add_edge(source, target)
    
    # Then add edge attributes
    for edge in G.edge_list():
        source, target = edge[0], edge[1]
        edge_data = G.get_edge_data(source, target)
        
        if edge_data is None:
            continue
            
        # Handle different edge data types
        if isinstance(edge_data, dict):
            # It's a dictionary, add each key-value pair as an attribute
            for key, value in edge_data.items():
                G_nx.edges[source, target][key] = value
        else:
            # It's a primitive type, store as 'weight'
            G_nx.edges[source, target]['weight'] = edge_data
    
    if pos is None:
        pos = nx.spring_layout(G_nx, seed=42)
    
    if community_attr is not None:
        # Get community values, defaulting to 0 if not found
        communities = []
        for node in G_nx.nodes():
            if community_attr in G_nx.nodes[node]:
                communities.append(G_nx.nodes[node][community_attr])
            else:
                communities.append(0)
        
        cmap = plt.cm.rainbow
        nx.draw_networkx(G_nx, pos=pos, node_color=communities, cmap=cmap, 
                         with_labels=True, node_size=100, font_size=8)
    else:
        nx.draw_networkx(G_nx, pos=pos, with_labels=True, node_size=100, font_size=8)
    
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def generate_synthetic_graph(graph_type: str, n_nodes: int, 
                           n_communities: Optional[int] = None, **kwargs) -> Tuple[rx.PyGraph, List[int]]:
    """
    Generate a synthetic graph with ground truth communities
    
    Parameters:
    -----------
    graph_type: str
        Type of graph ('sbm', 'nws', 'ba', 'lfr')
    n_nodes: int
        Number of nodes
    n_communities: int
        Number of communities
    **kwargs: 
        Additional parameters for specific graph models
        
    Returns:
    --------
    G: RustWorkX PyGraph
        The synthetic graph
    ground_truth: list
        List of ground truth community assignments
    """
    # First create with NetworkX, then convert to RustWorkX
    # (Since RustWorkX doesn't have built-in graph generators yet)
    import networkx as nx
    
    if graph_type.lower() == 'sbm':  # Stochastic Block Model
        if n_communities is None:
            n_communities = 3
            
        # Example parameters
        sizes = [n_nodes // n_communities] * n_communities
        p_in = kwargs.get('p_in', 0.3)  # probability within community
        p_out = kwargs.get('p_out', 0.05)  # probability between communities
        
        # Create probability matrix
        p_matrix = torch.ones((n_communities, n_communities)) * p_out
        p_matrix.fill_diagonal_(p_in)
        p_matrix_np = p_matrix.numpy()
        
        G_nx = nx.stochastic_block_model(sizes, p_matrix_np, seed=42)
        
        # Add ground truth
        ground_truth = []
        for i, size in enumerate(sizes):
            ground_truth.extend([i] * size)
            
        for i, comm in enumerate(ground_truth):
            G_nx.nodes[i]['community'] = comm
            
    elif graph_type.lower() == 'nws':  # Newman-Watts-Strogatz
        k = kwargs.get('k', 5)  # Each node connected to k nearest neighbors
        p = kwargs.get('p', 0.1)  # Probability of rewiring
        
        G_nx = nx.newman_watts_strogatz_graph(n_nodes, k, p, seed=42)
        
        # For NWS, we don't have inherent communities, so we'll use Louvain
        if CDLIB_AVAILABLE:
            communities = algorithms.louvain(G_nx)
            ground_truth_comm = communities.communities
            
            # Convert to node-indexed list
            node_to_comm = {}
            for i, comm in enumerate(ground_truth_comm):
                for node in comm:
                    node_to_comm[node] = i
                    
            ground_truth = [node_to_comm[i] for i in range(n_nodes)]
            
            for i, comm in enumerate(ground_truth):
                G_nx.nodes[i]['community'] = comm
        else:
            # Fallback if cdlib not available
            ground_truth = [0] * n_nodes  # Default community
            
    elif graph_type.lower() == 'ba':  # Barabási-Albert
        m = kwargs.get('m', 3)  # Number of edges to attach from a new node
        
        G_nx = nx.barabasi_albert_graph(n_nodes, m, seed=42)
        
        # For BA, we don't have inherent communities, so we'll use Louvain
        if CDLIB_AVAILABLE:
            communities = algorithms.louvain(G_nx)
            ground_truth_comm = communities.communities
            
            # Convert to node-indexed list
            node_to_comm = {}
            for i, comm in enumerate(ground_truth_comm):
                for node in comm:
                    node_to_comm[node] = i
                    
            ground_truth = [node_to_comm[i] for i in range(n_nodes)]
            
            for i, comm in enumerate(ground_truth):
                G_nx.nodes[i]['community'] = comm
        else:
            # Fallback if cdlib not available
            ground_truth = [0] * n_nodes  # Default community
            
    elif graph_type.lower() == 'lfr':  # Lancichinetti-Fortunato-Radicchi benchmark
        if not CDLIB_AVAILABLE:
            raise ImportError("cdlib is required for LFR benchmark")
            
        tau1 = kwargs.get('tau1', 3)  # Power law exponent for degree distribution
        tau2 = kwargs.get('tau2', 1.5)  # Power law exponent for community size distribution
        mu = kwargs.get('mu', 0.1)  # Mixing parameter
        
        # Use cdlib to generate LFR benchmark
        G_nx, ground_truth_comm = algorithms.lfr_benchmark(
            n=n_nodes,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            average_degree=kwargs.get('avg_degree', 10),
            min_degree=kwargs.get('min_degree', 5),
            max_degree=kwargs.get('max_degree', 50),
            min_community=kwargs.get('min_community', 20),
            max_community=kwargs.get('max_community', 100),
            seed=42
        )
        
        # Convert list-of-lists communities to node list
        node_to_comm = {}
        for i, comm in enumerate(ground_truth_comm):
            for node in comm:
                node_to_comm[node] = i
                
        ground_truth = [node_to_comm.get(i, 0) for i in range(n_nodes)]
        
        # Add ground truth to nodes
        for i, comm in enumerate(ground_truth):
            G_nx.nodes[i]['community'] = comm
    else:
        raise ValueError(f"Unsupported graph type: {graph_type}")
    
    # Convert from NetworkX to RustWorkX
    G = rx.PyGraph()
    
    # Add nodes with attributes
    node_mapping = {}
    for node in G_nx.nodes():
        attrs = G_nx.nodes[node]
        node_mapping[node] = G.add_node(attrs)
    
    # Add edges with attributes
    for u, v, data in G_nx.edges(data=True):
        G.add_edge(node_mapping[u], node_mapping[v], data if data else None)
        
    return G, ground_truth


def compute_graph_statistics(G: rx.PyGraph) -> Dict[str, Any]:
    """
    Compute various statistics for a graph
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
        
    Returns:
    --------
    stats: dict
        Dictionary of graph statistics
    """
    stats = {}
    
    stats['n_nodes'] = len(G)
    stats['n_edges'] = G.num_edges()
    stats['density'] = 2 * G.num_edges() / (len(G) * (len(G) - 1)) if len(G) > 1 else 0
    
    # Calculate degree statistics
    degrees = torch.tensor([G.degree(i) for i in range(len(G))])
    
    stats['avg_degree'] = degrees.float().mean().item() if len(G) > 0 else 0
    stats['max_degree'] = degrees.max().item() if len(G) > 0 else 0
    stats['min_degree'] = degrees.min().item() if len(G) > 0 else 0
    
    # Calculate clustering (convert to NetworkX temporarily for this)
    import networkx as nx
    G_nx = nx.Graph()
    for i in range(len(G)):
        G_nx.add_node(i)
    for edge in G.edge_list():
        G_nx.add_edge(edge[0], edge[1])
    
    stats['avg_clustering'] = nx.average_clustering(G_nx)
    
    try:
        # These can be computationally expensive for large graphs
        if rx.is_connected(G):
            # RustWorkX implementations for performance
            stats['diameter'] = rx.diameter(G)
            stats['avg_shortest_path'] = rx.average_shortest_path_length(G)
        else:
            stats['diameter'] = 'N/A (Graph is disconnected)'
            stats['avg_shortest_path'] = 'N/A (Graph is disconnected)'
    except Exception as e:
        stats['diameter'] = f'N/A (Error: {str(e)})'
        stats['avg_shortest_path'] = f'N/A (Error: {str(e)})'
    
    # Degree distribution
    stats['degree_distribution'] = degrees.tolist()
    
    return stats


def display_graph_statistics(stats: Dict[str, Any]):
    """
    Print graph statistics in a readable format
    
    Parameters:
    -----------
    stats: dict
        Dictionary of graph statistics
        
    Returns:
    --------
    None
    """
    print("=" * 50)
    print("GRAPH STATISTICS")
    print("=" * 50)
    print(f"Number of nodes: {stats['n_nodes']}")
    print(f"Number of edges: {stats['n_edges']}")
    print(f"Graph density: {stats['density']:.4f}")
    print(f"Average degree: {stats['avg_degree']:.4f}")
    print(f"Minimum degree: {stats['min_degree']}")
    print(f"Maximum degree: {stats['max_degree']}")
    print(f"Average clustering coefficient: {stats['avg_clustering']:.4f}")
    print(f"Diameter: {stats['diameter']}")
    print(f"Average shortest path length: {stats['avg_shortest_path']}")
    print("=" * 50)
    
    # Plot degree distribution
    plt.figure(figsize=(10, 6))
    plt.hist(stats['degree_distribution'], bins=30)
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.show()


def convert_rustworkx_to_pytorch_geometric(G: rx.PyGraph, 
                                         node_features: Optional[torch.Tensor] = None, 
                                         node_labels: Optional[torch.Tensor] = None) -> Data:
    """
    Compatibility wrapper for rwx_to_pyg() in gnn_community_detection.py
    
    This function maintains backward compatibility with older code.
    See gnn_community_detection.rwx_to_pyg() for proper documentation.
    """
    from .gnn_community_detection import rwx_to_pyg
    return rwx_to_pyg(G, node_features, node_labels)