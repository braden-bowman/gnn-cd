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
def load_data(filepath: str, filetype: str = 'parquet', 
            chunk_size: Optional[int] = None, 
            use_streaming: bool = True) -> Union[pl.DataFrame, Dict]:
    """
    Load data from various file formats with optimized memory usage
    
    Parameters:
    -----------
    filepath: str
        Path to the data file
    filetype: str
        Type of file (parquet, csv, json, pickle, etc.)
    chunk_size: Optional[int]
        Size of chunks for streaming large files (None means autodetect)
    use_streaming: bool
        Whether to use streaming for large files
        
    Returns:
    --------
    data: DataFrame or dict
        Loaded data
    """
    # Check file size to determine if streaming is needed
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    large_file = file_size_mb > 100  # Consider files > 100 MB as large
    
    if filetype.lower() == 'parquet':
        if large_file and use_streaming:
            # Use lazy evaluation for large parquet files
            return pl.scan_parquet(filepath)
        else:
            return pl.read_parquet(filepath)
    elif filetype.lower() == 'csv':
        # Try multiple encodings for CSV files
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
        
        for encoding in encodings:
            try:
                if large_file and use_streaming:
                    # Use lazy evaluation with streaming
                    if chunk_size:
                        return pl.scan_csv(filepath, encoding=encoding).collect(streaming=True, chunk_size=chunk_size)
                    else:
                        return pl.scan_csv(filepath, encoding=encoding).collect(streaming=True)
                else:
                    # For smaller files, read directly
                    return pl.read_csv(filepath, encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with ignore errors
        return pl.read_csv(filepath, encoding_errors='ignore')
    elif filetype.lower() == 'json':
        if large_file and use_streaming:
            # Stream JSON data for large files
            return pl.scan_ndjson(filepath).collect(streaming=True)
        else:
            return pl.read_json(filepath)
    elif filetype.lower() == 'pickle':
        with open(filepath, 'rb') as f:
            import pickle
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {filetype}")


def create_graph_from_edgelist(edgelist: Union[pl.DataFrame, pl.LazyFrame], 
                              directed: bool = False, 
                              weighted: bool = False,
                              chunk_size: Optional[int] = None,
                              max_nodes: Optional[int] = None) -> rx.PyGraph:
    """
    Create a RustWorkX graph from an edge list with memory-optimized processing
    
    Parameters:
    -----------
    edgelist: pl.DataFrame or pl.LazyFrame
        Polars DataFrame with edge information (source, target, [weight])
    directed: bool
        Whether the graph is directed
    weighted: bool
        Whether the graph has edge weights
    chunk_size: Optional[int]
        Process edges in chunks to save memory (None means process all at once)
    max_nodes: Optional[int]
        Limit the number of nodes for very large graphs
        
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
    
    # Determine if we need to process in chunks
    is_lazy = isinstance(edgelist, pl.LazyFrame)
    large_graph = False
    
    # Get DataFrame from LazyFrame if needed
    if is_lazy:
        # For large graphs, check the size first
        sample = edgelist.limit(1000).collect()
        estimated_rows = edgelist.select(pl.count()).collect().item()
        # Estimate if this is a large graph (>1M edges)
        large_graph = estimated_rows > 1_000_000
        
        if large_graph and chunk_size is not None:
            # Process in chunks for memory efficiency
            node_mapping = {}
            
            # First pass: collect unique nodes
            unique_sources = set()
            unique_targets = set()
            
            for chunk in edgelist.collect(streaming=True, chunk_size=chunk_size):
                unique_sources.update(chunk['source'].unique().to_list())
                unique_targets.update(chunk['target'].unique().to_list())
            
            all_nodes = list(unique_sources.union(unique_targets))
            
            # Sample nodes if max_nodes is specified
            if max_nodes is not None and len(all_nodes) > max_nodes:
                import random
                random.seed(42)  # For reproducibility
                all_nodes = random.sample(all_nodes, max_nodes)
            
            # Add nodes to the graph
            for node in all_nodes:
                node_idx = G.add_node(node)
                node_mapping[node] = node_idx
            
            # Second pass: add edges in chunks
            for chunk in edgelist.collect(streaming=True, chunk_size=chunk_size):
                source_list = chunk['source'].to_list()
                target_list = chunk['target'].to_list()
                
                if weighted and 'weight' in chunk.columns:
                    weight_list = chunk['weight'].to_list()
                    for src, tgt, wt in zip(source_list, target_list, weight_list):
                        # Skip if nodes were sampled out
                        if src in node_mapping and tgt in node_mapping:
                            try:
                                G.add_edge(node_mapping[src], node_mapping[tgt], {'weight': wt})
                            except KeyError:
                                # Silently skip edges with missing nodes (due to sampling)
                                pass
                else:
                    for src, tgt in zip(source_list, target_list):
                        # Skip if nodes were sampled out
                        if src in node_mapping and tgt in node_mapping:
                            try:
                                G.add_edge(node_mapping[src], node_mapping[tgt], None)
                            except KeyError:
                                # Silently skip edges with missing nodes (due to sampling)
                                pass
                
            return G  # Return early with chunked processing complete
        else:
            # For smaller LazyFrames, collect all at once
            df = edgelist.collect()
    else:
        # Regular DataFrame
        df = edgelist
        large_graph = len(df) > 1_000_000
    
    # For non-chunked processing:
    # Get unique nodes using Polars operations
    source_nodes = df.select('source').unique().to_series().to_list()
    target_nodes = df.select('target').unique().to_series().to_list() 
    all_nodes = list(set(source_nodes + target_nodes))
    
    # Sample nodes if max_nodes is specified and the graph is large
    if max_nodes is not None and len(all_nodes) > max_nodes:
        import random
        random.seed(42)  # For reproducibility
        all_nodes = random.sample(all_nodes, max_nodes)
        
        # Create a set for faster lookups
        nodes_set = set(all_nodes)
        
        # Filter edges to only include sampled nodes
        mask = (df['source'].is_in(nodes_set)) & (df['target'].is_in(nodes_set))
        df = df.filter(mask)
    
    # Add nodes to the graph and create mapping
    node_mapping = {}
    for node in all_nodes:
        node_idx = G.add_node(node)
        node_mapping[node] = node_idx
    
    # Extract edge data as native Python lists - use Polars' optimized methods
    source_list = df['source'].to_list()
    target_list = df['target'].to_list()
    
    # Add edges in a memory-efficient way
    if weighted and 'weight' in df.columns:
        # Use Polars' vectorized operations for efficiency
        weight_list = df['weight'].to_list()
        
        # Process in batches if it's a large graph
        if large_graph and chunk_size is not None:
            total_edges = len(source_list)
            for i in range(0, total_edges, chunk_size):
                end_idx = min(i + chunk_size, total_edges)
                for j in range(i, end_idx):
                    src, tgt, wt = source_list[j], target_list[j], weight_list[j]
                    try:
                        G.add_edge(node_mapping[src], node_mapping[tgt], {'weight': wt})
                    except KeyError:
                        # Skip edges with missing nodes due to sampling
                        pass
        else:
            # Process all at once for smaller graphs
            for src, tgt, wt in zip(source_list, target_list, weight_list):
                try:
                    G.add_edge(node_mapping[src], node_mapping[tgt], {'weight': wt})
                except KeyError as e:
                    print(f"KeyError: {e}, src type: {type(src)}, value: {src}")
                    print(f"node_mapping keys: {[type(k) for k in node_mapping.keys()][:5]}...")
                    raise
    else:
        # Process in batches if it's a large graph
        if large_graph and chunk_size is not None:
            total_edges = len(source_list)
            for i in range(0, total_edges, chunk_size):
                end_idx = min(i + chunk_size, total_edges)
                for j in range(i, end_idx):
                    src, tgt = source_list[j], target_list[j]
                    try:
                        G.add_edge(node_mapping[src], node_mapping[tgt], None)
                    except KeyError:
                        # Skip edges with missing nodes due to sampling
                        pass
        else:
            # Process all at once for smaller graphs
            for src, tgt in zip(source_list, target_list):
                try:
                    G.add_edge(node_mapping[src], node_mapping[tgt], None)
                except KeyError:
                    # Skip edges with missing nodes due to sampling
                    pass
    
    return G


def create_graph_from_adjacency(adjacency: torch.Tensor, 
                              node_features: Optional[torch.Tensor] = None, 
                              node_labels: Optional[torch.Tensor] = None,
                              batch_size: int = 10000,
                              use_gpu: bool = True) -> rx.PyGraph:
    """
    Create a RustWorkX graph from an adjacency matrix with GPU acceleration when available
    
    Parameters:
    -----------
    adjacency: torch.Tensor
        Adjacency matrix
    node_features: torch.Tensor
        Matrix of node features
    node_labels: torch.Tensor
        Array of node labels (for ground truth communities)
    batch_size: int
        Batch size for processing large adjacency matrices
    use_gpu: bool
        Whether to use GPU acceleration if available
        
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
    
    # Process node features if needed
    if node_features is not None and torch.is_tensor(node_features) and use_gpu and torch.cuda.is_available():
        # Move features to CPU for extraction
        node_features = node_features.detach().cpu()
    
    # Add all nodes first (optimized operation)
    for i in range(n_nodes):
        node_data = {}
        
        # Add node features if provided
        if node_features is not None:
            node_data['features'] = node_features[i].numpy() if torch.is_tensor(node_features[i]) else node_features[i]
        
        # Add node labels if provided
        if node_labels is not None:
            if torch.is_tensor(node_labels):
                node_data['community'] = int(node_labels[i].item())
            else:
                node_data['community'] = int(node_labels[i])
        
        # Add node to graph
        node_indices.append(G.add_node(node_data))
    
    # Efficiently get edge indices using GPU if available
    gpu_available = torch.cuda.is_available() and use_gpu
    device = torch.device('cuda' if gpu_available else 'cpu')
    
    # For large matrices, process in batches
    large_matrix = n_nodes > 1000
    if large_matrix:
        # Process in batches to save memory
        for i in range(0, n_nodes, batch_size):
            # Get a batch of rows
            end_idx = min(i + batch_size, n_nodes)
            batch = adjacency[i:end_idx]
            
            if gpu_available:
                batch = batch.to(device)
            
            # Find non-zero entries in this batch
            batch_indices = torch.nonzero(batch > 0)
            
            if gpu_available:
                batch_indices = batch_indices.cpu()
            
            # Add edges for this batch
            for row_idx, col_idx in batch_indices:
                # Adjust row index based on batch offset
                global_row = i + row_idx
                global_col = col_idx
                
                # Only add edges for upper triangular part to avoid duplicates
                if global_row < global_col:
                    # Get weight if not 1
                    weight_val = float(adjacency[global_row, global_col].item())
                    weight = {'weight': weight_val} if weight_val != 1 else None
                    
                    # Add edge
                    G.add_edge(node_indices[global_row], node_indices[global_col], weight)
    else:
        # For smaller matrices, process all at once
        if gpu_available:
            adjacency = adjacency.to(device)
        
        # Get all edge indices at once
        edge_indices = torch.nonzero(adjacency > 0)
        
        if gpu_available:
            edge_indices = edge_indices.cpu()
        
        # Process edges in batches for better performance
        total_edges = len(edge_indices)
        for idx in range(0, total_edges, batch_size):
            end_idx = min(idx + batch_size, total_edges)
            for k in range(idx, end_idx):
                i, j = edge_indices[k]
                if i < j:  # Upper triangular part for undirected graph
                    weight_val = float(adjacency[i, j].item())
                    weight = {'weight': weight_val} if weight_val != 1 else None
                    G.add_edge(node_indices[i], node_indices[j], weight)
    
    return G


def convert_rustworkx_to_pytorch_geometric(G: rx.PyGraph, 
                                         node_features: Optional[torch.Tensor] = None, 
                                         node_labels: Optional[torch.Tensor] = None,
                                         batch_size: Optional[int] = None,
                                         use_gpu: bool = True) -> Data:
    """
    Convert a RustWorkX graph to a PyTorch Geometric Data object with optimized memory usage
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to convert
    node_features: torch.Tensor
        Matrix of node features (optional if already in G)
    node_labels: torch.Tensor
        Array of node labels (optional if already in G)
    batch_size: Optional[int]
        Size of batches when processing large graphs
    use_gpu: bool
        Whether to use GPU acceleration if available
        
    Returns:
    --------
    data: torch_geometric.data.Data
        PyTorch Geometric Data object
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric is not available")
    
    num_nodes = len(G)
    num_edges = G.num_edges()
    
    # Determine if this is a large graph
    large_graph = num_nodes > 10000 or num_edges > 100000
    
    # Set batch size if not provided
    if batch_size is None:
        batch_size = 10000 if large_graph else num_edges
    
    # Get edges as tensor
    if large_graph:
        # Process edges in batches for large graphs
        edge_list = G.edge_list()
        
        # Pre-allocate edge_index for better memory efficiency
        edge_index = torch.zeros((2, len(edge_list) * (2 if isinstance(G, rx.PyGraph) else 1)), 
                                dtype=torch.long)
        
        # Fill in batches
        for i in range(0, len(edge_list), batch_size):
            end_idx = min(i + batch_size, len(edge_list))
            batch_edges = edge_list[i:end_idx]
            
            # Convert batch to tensor
            batch_edge_index = torch.tensor(batch_edges, dtype=torch.long).t()
            
            # Add to edge_index
            edge_index[:, i*2:(i+len(batch_edges))*2] = torch.cat(
                [batch_edge_index, batch_edge_index.flip(0)], dim=1) if isinstance(G, rx.PyGraph) else batch_edge_index
    else:
        # For smaller graphs, process all at once
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
    
    # Get node features with GPU acceleration if available
    gpu_available = torch.cuda.is_available() and use_gpu
    device = torch.device('cuda' if gpu_available else 'cpu')
    
    if node_features is not None:
        x = node_features
    else:
        # Try to get features from graph nodes
        if large_graph:
            # For large graphs, process in batches
            features_shape = None
            all_features = []
            
            # First determine feature shape from a sample
            for i in range(num_nodes):
                node_data = G.get_node_data(i)
                if node_data and 'features' in node_data:
                    features_shape = len(node_data['features'])
                    break
            
            if features_shape is not None:
                # Now collect all features efficiently
                for i in range(0, num_nodes, batch_size):
                    end_idx = min(i + batch_size, num_nodes)
                    batch_features = []
                    
                    for j in range(i, end_idx):
                        node_data = G.get_node_data(j)
                        if node_data and 'features' in node_data:
                            batch_features.append(node_data['features'])
                        else:
                            # Use zeros if features missing
                            batch_features.append(torch.zeros(features_shape))
                    
                    all_features.append(torch.tensor(batch_features, dtype=torch.float))
                
                x = torch.cat(all_features)
            else:
                # Use one-hot degree as default feature
                node_degrees = torch.tensor([G.degree(i) for i in range(num_nodes)])
                max_degree = node_degrees.max().item() if len(node_degrees) > 0 else 0
                x = torch.zeros((num_nodes, max_degree + 1))
                
                # Fill in batches
                for i in range(0, num_nodes, batch_size):
                    end_idx = min(i + batch_size, num_nodes)
                    degrees_batch = node_degrees[i:end_idx]
                    
                    for j, degree in enumerate(degrees_batch):
                        x[i+j, degree] = 1
        else:
            # For smaller graphs, collect all features at once
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
        if large_graph:
            # Check if we have labels first
            has_labels = False
            for i in range(min(100, num_nodes)):  # Check a sample
                node_data = G.get_node_data(i)
                if node_data and 'community' in node_data:
                    has_labels = True
                    break
                    
            if has_labels:
                # Allocate tensor
                y = torch.zeros(num_nodes, dtype=torch.long)
                
                # Fill in batches
                for i in range(0, num_nodes, batch_size):
                    end_idx = min(i + batch_size, num_nodes)
                    
                    for j in range(i, end_idx):
                        node_data = G.get_node_data(j)
                        if node_data and 'community' in node_data:
                            y[j] = node_data['community']
            else:
                y = None
        else:
            # For smaller graphs, collect all labels at once
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
    if num_edges > 0:
        if large_graph:
            # Process in batches for large graphs
            edge_list = G.edge_list()
            edge_weights = torch.ones(len(edge_list) * (2 if isinstance(G, rx.PyGraph) else 1))
            
            idx = 0
            for i in range(0, len(edge_list), batch_size):
                end_idx = min(i + batch_size, len(edge_list))
                batch_edges = edge_list[i:end_idx]
                
                for j, edge in enumerate(batch_edges):
                    edge_data = G.get_edge_data(edge[0], edge[1])
                    weight = edge_data.get('weight', 1.0) if edge_data else 1.0
                    
                    # Set weight
                    edge_weights[idx] = weight
                    idx += 1
                    
                    # For undirected graphs, duplicate weight
                    if isinstance(G, rx.PyGraph):
                        edge_weights[idx] = weight
                        idx += 1
            
            edge_attr = edge_weights.view(-1, 1)
        else:
            # For smaller graphs, process all at once
            edge_weights = []
            for edge in G.edge_list():
                edge_data = G.get_edge_data(edge[0], edge[1])
                if edge_data and 'weight' in edge_data:
                    edge_weights.append(edge_data['weight'])
                else:
                    edge_weights.append(1.0)  # Default weight
                    
            # For undirected graph, duplicate weights for both directions
            if isinstance(G, rx.PyGraph):
                edge_weights = edge_weights + edge_weights
                
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    
    # Move to GPU if available and requested
    if gpu_available:
        if x is not None:
            x = x.to(device)
        edge_index = edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        if y is not None:
            y = y.to(device)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data


def plot_graph(G: rx.PyGraph, community_attr: Optional[str] = None, 
             pos: Optional[Dict] = None, figsize: Tuple[int, int] = (10, 8), 
             title: str = "Graph Visualization",
             max_nodes: int = 1000,
             use_plotly: bool = False,
             edge_alpha: float = 0.3,
             node_size_factor: float = 1.0):
    """
    Plot a RustWorkX graph with community colors, optimized for large graphs
    
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
    max_nodes: int
        Maximum number of nodes to display for large graphs
    use_plotly: bool
        Whether to use Plotly (interactive) over Matplotlib (static)
    edge_alpha: float
        Transparency of edges
    node_size_factor: float
        Scale factor for node sizes
        
    Returns:
    --------
    None (displays plot) or pos (node positions dictionary)
    """
    num_nodes = len(G)
    num_edges = G.num_edges()
    
    # Sample nodes if graph is too large
    if num_nodes > max_nodes:
        print(f"Graph has {num_nodes} nodes, sampling {max_nodes} for visualization")
        
        # Try to select nodes intelligently
        # If community attribute is provided, preserve community structure
        if community_attr is not None:
            # Gather nodes by community
            communities_dict = {}
            for i in range(num_nodes):
                node_data = G.get_node_data(i)
                if node_data and community_attr in node_data:
                    comm = node_data[community_attr]
                    if comm not in communities_dict:
                        communities_dict[comm] = []
                    communities_dict[comm].append(i)
                    
            # Sample proportionally from each community
            import random
            random.seed(42)
            sampled_nodes = []
            target_per_community = max(1, max_nodes // (len(communities_dict) or 1))
            
            for comm, nodes in communities_dict.items():
                # Take all nodes if fewer than target, otherwise sample
                if len(nodes) <= target_per_community:
                    sampled_nodes.extend(nodes)
                else:
                    sampled_nodes.extend(random.sample(nodes, target_per_community))
                    
            # If we need more nodes, sample randomly from the rest
            if len(sampled_nodes) < max_nodes:
                remaining = set(range(num_nodes)) - set(sampled_nodes)
                if remaining:
                    sampled_nodes.extend(random.sample(remaining, min(max_nodes - len(sampled_nodes), len(remaining))))
        else:
            # Sample based on degree for better visualization
            # High-degree nodes are more important for visualization
            degrees = [(i, G.degree(i)) for i in range(num_nodes)]
            degrees.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 20% by degree plus random sample of the rest
            top_count = max_nodes // 5
            random_count = max_nodes - top_count
            
            top_nodes = [node for node, _ in degrees[:top_count]]
            
            import random
            random.seed(42)
            random_nodes = random.sample([node for node, _ in degrees[top_count:]], min(random_count, len(degrees) - top_count))
            
            sampled_nodes = top_nodes + random_nodes
    else:
        # Use all nodes if under the limit
        sampled_nodes = list(range(num_nodes))
    
    # Create a subgraph with only the sampled nodes
    node_id_map = {original_id: i for i, original_id in enumerate(sampled_nodes)}
    
    # Use Plotly for interactive visualization if requested and available
    if use_plotly and PLOTLY_AVAILABLE:
        import plotly.graph_objects as go
        import numpy as np
        
        # For Plotly, we'll convert directly without going through NetworkX
        
        # Get node positions
        if pos is None:
            # We need NetworkX to compute layout
            import networkx as nx
            G_nx = nx.Graph()
            
            # Add sampled nodes
            for original_id in sampled_nodes:
                G_nx.add_node(original_id)
            
            # Add edges between sampled nodes
            edge_list = []
            for edge in G.edge_list():
                source, target = edge[0], edge[1]
                if source in sampled_nodes and target in sampled_nodes:
                    G_nx.add_edge(source, target)
                    edge_list.append((node_id_map[source], node_id_map[target]))
            
            # Compute layout
            nx_pos = nx.spring_layout(G_nx, seed=42)
            # Convert to format needed for plotly
            pos = {node_id_map[i]: [coord[0], coord[1]] for i, coord in nx_pos.items()}
        
        # Prepare node data
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_size = []
        
        for i, original_id in enumerate(sampled_nodes):
            node_data = G.get_node_data(original_id)
            
            # Add position
            node_pos = pos.get(i, [0, 0])
            node_x.append(node_pos[0])
            node_y.append(node_pos[1])
            
            # Add color based on community
            if node_data and community_attr and community_attr in node_data:
                node_colors.append(node_data[community_attr])
            else:
                node_colors.append(0)
            
            # Add tooltip text
            node_text.append(f"Node ID: {original_id}<br>Degree: {G.degree(original_id)}")
            
            # Node size based on degree
            node_size.append(G.degree(original_id) * node_size_factor + 5)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        
        for edge in G.edge_list():
            source, target = edge[0], edge[1]
            if source in sampled_nodes and target in sampled_nodes:
                source_idx = node_id_map[source]
                target_idx = node_id_map[target]
                
                source_pos = pos.get(source_idx, [0, 0])
                target_pos = pos.get(target_idx, [0, 0])
                
                edge_x.extend([source_pos[0], target_pos[0], None])
                edge_y.extend([source_pos[1], target_pos[1], None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color=f'rgba(150,150,150,{edge_alpha})'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Rainbow',
                reversescale=False,
                color=node_colors,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title=community_attr or 'Group',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=0.5, color='rgb(255,255,255)')
            )
        ))
        
        # Set layout
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=figsize[0]*100,
            height=figsize[1]*100
        )
        
        fig.show()
        
        # Return node positions for potential reuse
        return pos
    
    else:
        # Fall back to Matplotlib for static visualization
        plt.figure(figsize=figsize)
        
        # Convert to NetworkX for visualization
        import networkx as nx
        G_nx = nx.Graph()
        
        # Add sampled nodes
        for i, original_id in enumerate(sampled_nodes):
            G_nx.add_node(i)  # Add with new index
            
            # Add node attributes
            node_data = G.get_node_data(original_id)
            if node_data:
                if isinstance(node_data, dict):
                    for key, value in node_data.items():
                        G_nx.nodes[i][key] = value
                else:
                    G_nx.nodes[i]['value'] = node_data
                    if community_attr == 'value':
                        G_nx.nodes[i][community_attr] = node_data
        
        # Add edges between sampled nodes
        for edge in G.edge_list():
            source, target = edge[0], edge[1]
            if source in sampled_nodes and target in sampled_nodes:
                # Map to new indices
                new_source = node_id_map[source]
                new_target = node_id_map[target]
                G_nx.add_edge(new_source, new_target)
                
                # Add edge attributes
                edge_data = G.get_edge_data(source, target)
                if edge_data:
                    if isinstance(edge_data, dict):
                        for key, value in edge_data.items():
                            G_nx.edges[new_source, new_target][key] = value
                    else:
                        G_nx.edges[new_source, new_target]['weight'] = edge_data
        
        # Compute positions if not provided
        if pos is None:
            pos = nx.spring_layout(G_nx, seed=42)
        
        # Visualize with colors based on community attribute
        if community_attr is not None:
            communities = []
            for node in G_nx.nodes():
                if community_attr in G_nx.nodes[node]:
                    communities.append(G_nx.nodes[node][community_attr])
                else:
                    communities.append(0)
            
            cmap = plt.cm.rainbow
            nx.draw_networkx(G_nx, pos=pos, node_color=communities, cmap=cmap, 
                            with_labels=len(G_nx) < 100,  # Only show labels for small graphs
                            node_size=100 * node_size_factor, 
                            font_size=8,
                            alpha=1.0,
                            edge_color=f'rgba(150,150,150,{edge_alpha})')
        else:
            nx.draw_networkx(G_nx, pos=pos, 
                            with_labels=len(G_nx) < 100,  # Only show labels for small graphs
                            node_size=100 * node_size_factor, 
                            font_size=8,
                            alpha=1.0,
                            edge_color=f'rgba(150,150,150,{edge_alpha})')
        
        plt.title(f"{title} (showing {len(sampled_nodes)} of {num_nodes} nodes)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        # Return node positions for potential reuse
        return pos


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


def compute_graph_statistics(G: rx.PyGraph, 
                         compute_expensive: bool = True, 
                         sample_clustering: bool = True, 
                         max_sample_size: int = 1000,
                         use_gpu: bool = True) -> Dict[str, Any]:
    """
    Compute various statistics for a graph with memory-efficient processing
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
    compute_expensive: bool
        Whether to compute expensive metrics like diameter for large graphs
    sample_clustering: bool
        Whether to sample nodes for clustering coefficient (for large graphs)
    max_sample_size: int
        Maximum number of nodes to sample for expensive calculations
    use_gpu: bool
        Whether to use GPU acceleration for calculations when available
        
    Returns:
    --------
    stats: dict
        Dictionary of graph statistics
    """
    stats = {}
    
    num_nodes = len(G)
    num_edges = G.num_edges()
    large_graph = num_nodes > 10000 or num_edges > 100000
    
    stats['n_nodes'] = num_nodes
    stats['n_edges'] = num_edges
    stats['density'] = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    
    # Determine device for tensor operations
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Calculate degree statistics efficiently
    if large_graph:
        # Process in batches to avoid memory issues
        batch_size = 10000
        all_degrees = []
        
        for i in range(0, num_nodes, batch_size):
            end_idx = min(i + batch_size, num_nodes)
            degrees_batch = [G.degree(j) for j in range(i, end_idx)]
            all_degrees.extend(degrees_batch)
            
        # Convert to tensor for statistics
        degrees = torch.tensor(all_degrees, dtype=torch.float, device=device)
    else:
        # For smaller graphs, process all at once
        degrees = torch.tensor([G.degree(i) for i in range(num_nodes)], 
                              dtype=torch.float, device=device)
    
    # Calculate basic statistics
    stats['avg_degree'] = degrees.mean().item() if num_nodes > 0 else 0
    stats['max_degree'] = degrees.max().item() if num_nodes > 0 else 0
    stats['min_degree'] = degrees.min().item() if num_nodes > 0 else 0
    
    # Calculate degree distribution histogram rather than storing all degrees
    if large_graph:
        # Create histogram instead of storing all degrees
        hist_values, hist_edges = torch.histogram(degrees, bins=100)
        stats['degree_histogram'] = {
            'values': hist_values.tolist(),
            'bin_edges': hist_edges.tolist()
        }
    else:
        # For smaller graphs, we can store the full distribution
        stats['degree_distribution'] = degrees.tolist()
    
    # Calculate clustering coefficient
    if large_graph and sample_clustering:
        # Sample nodes for clustering coefficient calculation
        import random
        random.seed(42)
        sample_size = min(max_sample_size, num_nodes)
        sampled_nodes = random.sample(range(num_nodes), sample_size)
        
        # Calculate clustering only for sampled nodes
        import networkx as nx
        G_nx = nx.Graph()
        
        # Add sampled nodes
        for node in sampled_nodes:
            G_nx.add_node(node)
        
        # Add edges between sampled nodes
        for edge in G.edge_list():
            source, target = edge[0], edge[1]
            if source in sampled_nodes and target in sampled_nodes:
                G_nx.add_edge(source, target)
        
        # Calculate clustering coefficient on sampled subgraph
        stats['avg_clustering'] = nx.average_clustering(G_nx)
        stats['clustering_sample_size'] = sample_size
    else:
        # For smaller graphs, calculate exact clustering
        import networkx as nx
        G_nx = nx.Graph()
        
        # Add all nodes
        for i in range(num_nodes):
            G_nx.add_node(i)
        
        # Add all edges
        for edge in G.edge_list():
            source, target = edge[0], edge[1]
            G_nx.add_edge(source, target)
        
        # Calculate clustering coefficient
        stats['avg_clustering'] = nx.average_clustering(G_nx)
    
    # Calculate expensive metrics only if requested and graph is not too large
    if compute_expensive and not large_graph:
        try:
            # Check connectivity first (much faster than computing diameter)
            if rx.is_connected(G):
                # RustworkX implementations are faster than NetworkX
                stats['diameter'] = rx.diameter(G)
                stats['avg_shortest_path'] = rx.average_shortest_path_length(G)
            else:
                # For disconnected graphs, report N/A
                stats['diameter'] = 'N/A (Graph is disconnected)'
                stats['avg_shortest_path'] = 'N/A (Graph is disconnected)'
        except Exception as e:
            stats['diameter'] = f'N/A (Error: {str(e)})'
            stats['avg_shortest_path'] = f'N/A (Error: {str(e)})'
    else:
        # Skip expensive calculations for large graphs
        stats['diameter'] = 'N/A (Large graph, skipped calculation)'
        stats['avg_shortest_path'] = 'N/A (Large graph, skipped calculation)'
    
    # Calculate largest connected component size
    try:
        components = list(rx.connected_components(G))
        largest_component = max(components, key=len)
        stats['largest_component_size'] = len(largest_component)
        stats['largest_component_ratio'] = len(largest_component) / num_nodes
        stats['num_components'] = len(components)
    except Exception as e:
        stats['largest_component_size'] = f'N/A (Error: {str(e)})'
        stats['largest_component_ratio'] = 'N/A'
        stats['num_components'] = 'N/A'
    
    return stats


def display_graph_statistics(stats: Dict[str, Any], 
                           figsize: Tuple[int, int] = (10, 6),
                           use_plotly: bool = False):
    """
    Print graph statistics in a readable format with optimized visualization
    
    Parameters:
    -----------
    stats: dict
        Dictionary of graph statistics
    figsize: tuple
        Figure size for plots
    use_plotly: bool
        Whether to use Plotly (interactive) over Matplotlib (static)
        
    Returns:
    --------
    None
    """
    print("=" * 50)
    print("GRAPH STATISTICS")
    print("=" * 50)
    print(f"Number of nodes: {stats['n_nodes']}")
    print(f"Number of edges: {stats['n_edges']}")
    print(f"Graph density: {stats['density']:.6f}")
    print(f"Average degree: {stats['avg_degree']:.4f}")
    print(f"Minimum degree: {stats['min_degree']}")
    print(f"Maximum degree: {stats['max_degree']}")
    
    # Print clustering info (may have been sampled)
    if 'clustering_sample_size' in stats:
        print(f"Average clustering coefficient: {stats['avg_clustering']:.4f} (sampled from {stats['clustering_sample_size']} nodes)")
    else:
        print(f"Average clustering coefficient: {stats['avg_clustering']:.4f}")
    
    # Print component info if available
    if 'num_components' in stats and not isinstance(stats['num_components'], str):
        print(f"Number of connected components: {stats['num_components']}")
        print(f"Largest component size: {stats['largest_component_size']} nodes ({stats['largest_component_ratio']:.2%} of graph)")
    
    print(f"Diameter: {stats['diameter']}")
    print(f"Average shortest path length: {stats['avg_shortest_path']}")
    print("=" * 50)
    
    # Plot degree distribution
    if use_plotly and PLOTLY_AVAILABLE:
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Check if we have histogram or full distribution
        if 'degree_histogram' in stats:
            # For large graphs, we use pre-computed histogram
            hist_data = stats['degree_histogram']
            bin_values = hist_data['values']
            bin_edges = hist_data['bin_edges']
            
            # Convert bin edges to centers for bar plot
            bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
            
            fig.add_trace(go.Bar(
                x=bin_centers,
                y=bin_values,
                marker_color='royalblue',
                opacity=0.7
            ))
            
            # Use log scale for clearer visualization of power-law distributions
            fig.update_layout(
                title='Degree Distribution (Histogram)',
                xaxis_title='Degree',
                yaxis_title='Frequency',
                yaxis_type='log',  # Log scale for better power-law visualization
                width=figsize[0]*100,
                height=figsize[1]*100
            )
            
        elif 'degree_distribution' in stats:
            # For smaller graphs, we have the full distribution
            degree_dist = stats['degree_distribution']
            
            # Create histogram
            fig.add_trace(go.Histogram(
                x=degree_dist,
                marker=dict(
                    color='royalblue',
                    opacity=0.7
                ),
                nbinsx=30
            ))
            
            fig.update_layout(
                title='Degree Distribution',
                xaxis_title='Degree',
                yaxis_title='Frequency',
                width=figsize[0]*100,
                height=figsize[1]*100
            )
            
        # Add grid lines
        fig.update_layout(
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(200,200,200,0.2)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(200,200,200,0.2)'
            ),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.show()
    else:
        # Use matplotlib for static plots
        plt.figure(figsize=figsize)
        
        # Check if we have histogram or full distribution
        if 'degree_histogram' in stats:
            # For large graphs, we use pre-computed histogram
            hist_data = stats['degree_histogram']
            bin_values = hist_data['values']
            bin_edges = hist_data['bin_edges']
            
            # Get bin centers for bar plot
            bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
            
            plt.bar(bin_centers, bin_values, width=bin_edges[1]-bin_edges[0], alpha=0.7)
            plt.yscale('log')  # Log scale for better power-law visualization
            plt.title('Degree Distribution (Histogram)')
            
        elif 'degree_distribution' in stats:
            # For smaller graphs, we have the full distribution
            plt.hist(stats['degree_distribution'], bins=30, alpha=0.7)
            plt.title('Degree Distribution')
            
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def convert_rustworkx_to_pytorch_geometric(G: rx.PyGraph, 
                                         node_features: Optional[torch.Tensor] = None, 
                                         node_labels: Optional[torch.Tensor] = None,
                                         batch_size: Optional[int] = None,
                                         use_gpu: bool = True) -> Data:
    """
    Compatibility wrapper for rwx_to_pyg() in gnn_community_detection.py
    
    This function maintains backward compatibility with older code.
    See gnn_community_detection.rwx_to_pyg() for proper documentation.
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to convert
    node_features: torch.Tensor
        Matrix of node features (optional if already in G)
    node_labels: torch.Tensor
        Array of node labels (optional if already in G)
    batch_size: Optional[int]
        Size of batches when processing large graphs
    use_gpu: bool
        Whether to use GPU acceleration if available
        
    Returns:
    --------
    data: torch_geometric.data.Data
        PyTorch Geometric Data object
    """
    from .gnn_community_detection import rwx_to_pyg
    
    # Forward to implementation with optimized parameters
    if hasattr(rwx_to_pyg, '__code__') and 'batch_size' in rwx_to_pyg.__code__.co_varnames:
        # New implementation accepts batch_size and use_gpu
        return rwx_to_pyg(G, node_features, node_labels, batch_size, use_gpu)
    else:
        # Fall back to old implementation if needed
        return rwx_to_pyg(G, node_features, node_labels)