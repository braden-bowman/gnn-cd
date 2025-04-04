#!/usr/bin/env python3
# Process UNSW-NB15 dataset and construct a graph for community detection

import os
import polars as pl
import numpy as np
import networkx as nx
import rustworkx as rx
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import torch
import glob
from pathlib import Path
import time

# Define paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
UNSW_FEATURES_PATH = os.path.join(DATA_DIR, "UNSW-NB15_features.csv")
# Use glob pattern to find all data files
UNSW_DATA_PATHS = glob.glob(os.path.join(DATA_DIR, "UNSW-NB15_*.csv"))
UNSW_LABELS_PATH = os.path.join(DATA_DIR, "UNSW-NB15_GT.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
Path(PROCESSED_DIR).mkdir(exist_ok=True)

# Cache files
FEATURES_CACHE = os.path.join(PROCESSED_DIR, "selected_features.parquet")
GRAPH_CACHE = os.path.join(PROCESSED_DIR, "unsw_graph.pt")

def load_dataset(data_paths, features_path):
    """
    Load the UNSW-NB15 dataset and its features using Polars for efficiency
    
    Parameters:
    -----------
    data_paths: list
        List of paths to CSV data files
    features_path: str
        Path to features CSV file
        
    Returns:
    --------
    data: polars.DataFrame
        Combined dataset
    features_info: polars.DataFrame
        Feature information
    """
    print(f"Loading feature information from {features_path}")
    try:
        # Load feature information with Polars
        features_info = pl.read_csv(features_path)
        print(f"Loaded {features_info.height} features")
    except FileNotFoundError:
        print(f"Features file not found: {features_path}")
        features_info = None
    
    # Filter data files (excluding feature file)
    data_paths = [p for p in data_paths if os.path.basename(p).startswith("UNSW-NB15_") 
                  and not "features" in p.lower() and not "gt" in p.lower()]
    
    if not data_paths:
        print("No data files found. Please download the UNSW-NB15 dataset first.")
        print("Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        return None, features_info
    
    # Check for processed cache
    all_data_cache = os.path.join(PROCESSED_DIR, "all_data.parquet")
    data_files_mtime = max([os.path.getmtime(p) for p in data_paths if os.path.exists(p)])
    
    # Use cached data if available and up-to-date
    if os.path.exists(all_data_cache) and os.path.getmtime(all_data_cache) > data_files_mtime:
        print(f"Loading data from cache: {all_data_cache}")
        start_time = time.time()
        data = pl.read_parquet(all_data_cache)
        print(f"Loaded {data.height} records in {time.time() - start_time:.2f}s")
        return data, features_info
    
    # Otherwise, load and combine data files
    print("Loading and processing data files with Polars...")
    start_time = time.time()
    
    # Read all files and concatenate - much faster with Polars
    try:
        data_frames = []
        for path in data_paths:
            if os.path.exists(path):
                print(f"Loading {path}...")
                df = pl.read_csv(path, low_memory=True)
                print(f"  Loaded {df.height} records")
                data_frames.append(df)
        
        if not data_frames:
            print("No valid data files found")
            return None, features_info
        
        # Combine all data frames
        data = pl.concat(data_frames)
        print(f"Combined {len(data_frames)} files: {data.height} records in {time.time() - start_time:.2f}s")
        
        # Save to parquet for faster loading next time
        print(f"Saving combined data to cache: {all_data_cache}")
        data.write_parquet(all_data_cache)
        
        return data, features_info
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, features_info

def feature_selection(data, k=20):
    """
    Perform feature selection on the dataset with caching for efficiency
    
    Parameters:
    -----------
    data: polars.DataFrame
        Dataset with features and labels
    k: int
        Number of features to select
        
    Returns:
    --------
    selected_features: list
        List of selected feature names
    X_selected: numpy.ndarray
        Feature matrix with selected features
    y: numpy.ndarray
        Labels array
    """
    # Check if cached features exist and are up-to-date
    if os.path.exists(FEATURES_CACHE):
        print(f"Loading selected features from cache: {FEATURES_CACHE}")
        start_time = time.time()
        features_df = pl.read_parquet(FEATURES_CACHE)
        selected_features = features_df["feature_name"].to_list()
        
        # Extract features and labels from data
        exclude_cols = ['label', 'attack_cat', 'srcip', 'dstip']
        feature_cols = [col for col in selected_features if col in data.columns]
        
        # Convert to numpy arrays for scikit-learn compatibility
        X_pd = data.select(feature_cols).to_pandas()
        y_pd = data.select(pl.col("label")).to_pandas() if "label" in data.columns else pd.DataFrame(np.zeros(data.height))
        
        X_selected = X_pd.values
        y = y_pd.values.ravel()
        
        print(f"Loaded {len(selected_features)} cached features in {time.time() - start_time:.2f}s")
        return selected_features, X_selected, y
    
    print("Performing feature selection...")
    start_time = time.time()
    
    # Exclude label columns and IP addresses
    exclude_cols = ['label', 'attack_cat', 'srcip', 'dstip']
    X_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Convert to pandas for scikit-learn compatibility
    X_pd = data.select(X_cols).to_pandas()
    y_pd = data.select(pl.col("label")).to_pandas() if "label" in data.columns else pd.DataFrame(np.zeros(data.height))
    
    # Handle categorical features
    categorical_cols = X_pd.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X_pd = pd.get_dummies(X_pd, columns=categorical_cols, drop_first=True)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pd)
    
    # Apply feature selection
    selector = SelectKBest(f_classif, k=min(k, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y_pd.values.ravel())
    
    # Get selected feature names
    selected_features = X_pd.columns[selector.get_support()].tolist()
    print(f"Selected {len(selected_features)} features in {time.time() - start_time:.2f}s")
    
    # Cache the selected features
    print(f"Caching selected features to {FEATURES_CACHE}")
    feature_scores = selector.scores_[selector.get_support()].tolist()
    
    # Create dataframe with feature names and scores
    features_df = pl.DataFrame({
        "feature_name": selected_features,
        "score": feature_scores
    }).sort("score", descending=True)
    
    # Save to parquet
    features_df.write_parquet(FEATURES_CACHE)
    
    y = y_pd.values.ravel()
    return selected_features, X_selected, y

def construct_graph(data, feature_cols, target_col='label'):
    """
    Construct a graph from the dataset where devices are nodes using RustworkX for efficiency
    
    Parameters:
    -----------
    data: polars.DataFrame
        Dataset with features and labels
    feature_cols: list
        List of feature columns to use
    target_col: str
        Target column name
        
    Returns:
    --------
    G: rustworkx.PyGraph
        Constructed graph
    node_mapping: dict
        Mapping from node IDs to device IPs
    """
    # Check if cached graph exists
    if os.path.exists(GRAPH_CACHE):
        print(f"Loading cached graph from {GRAPH_CACHE}")
        start_time = time.time()
        G = load_graph(GRAPH_CACHE)
        # Create node mapping from loaded graph
        node_mapping = {}
        for i in range(len(G)):
            node_data = G.get_node_data(i)
            if 'ip' in node_data:
                node_mapping[node_data['ip']] = i
        print(f"Loaded graph with {len(G)} nodes and {G.num_edges()} edges in {time.time() - start_time:.2f}s")
        return G, node_mapping
    
    print("Constructing network graph...")
    start_time = time.time()
    
    # Extract unique source and destination IPs
    if 'srcip' in data.columns and 'dstip' in data.columns:
        # Use Polars for faster unique value extraction
        src_ips = set(data.select('srcip').unique().to_series().to_list())
        dst_ips = set(data.select('dstip').unique().to_series().to_list())
        all_ips = src_ips.union(dst_ips)
        print(f"Found {len(all_ips)} unique IP addresses")
    else:
        # If IP columns not found, use a different approach
        print("IP columns not found. Using synthetic node IDs based on row indices.")
        all_ips = set(range(data.height))
        # Add srcip and dstip columns
        data = data.with_columns([
            pl.Series(name="srcip", values=list(range(data.height))),
            pl.Series(name="dstip", values=[(i + 1) % data.height for i in range(data.height)])
        ])
    
    # Create mapping from IPs to node indices
    ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
    
    # Create graph
    G = rx.PyGraph()
    
    # Add nodes with feature vectors and labels
    node_mapping = {}
    
    # Process in chunks for large datasets
    chunk_size = 1000  # Adjust based on available memory
    ips_list = list(all_ips)
    
    for chunk_start in range(0, len(ips_list), chunk_size):
        chunk_ips = ips_list[chunk_start:chunk_start + chunk_size]
        
        # Process nodes in chunks
        for ip in chunk_ips:
            # Get rows where this IP appears as source or destination using Polars
            ip_data = data.filter((pl.col('srcip') == ip) | (pl.col('dstip') == ip))
            
            if ip_data.height == 0:
                # If no data for this IP, use zeros for features
                features = np.zeros(len(feature_cols))
                label = 0
            else:
                # Use available feature columns
                available_features = [col for col in feature_cols if col in ip_data.columns]
                
                if available_features:
                    # Aggregate features for this IP (using mean)
                    features = ip_data.select(available_features).mean().row(0)
                    
                    # If fewer features than expected, pad with zeros
                    if len(features) < len(feature_cols):
                        features_dict = dict(zip(available_features, features))
                        features = [features_dict.get(col, 0.0) for col in feature_cols]
                else:
                    # If no features available, use zeros
                    features = np.zeros(len(feature_cols))
                
                # Determine label (1 if any traffic involving this IP is malicious)
                if target_col in ip_data.columns:
                    label = 1 if ip_data.select(pl.col(target_col) == 1).any().row(0)[0] else 0
                else:
                    label = 0
            
            # Create node data with features and label
            node_data = {
                'features': np.array(features, dtype=np.float32),
                'label': int(label),
                'ip': ip
            }
            
            # Add node to graph
            node_idx = G.add_node(node_data)
            node_mapping[ip] = node_idx
    
    # Process edges in chunks for large datasets
    print("Adding edges to graph...")
    edge_time = time.time()
    
    # Convert to pandas for more efficient iteration over rows
    edge_data = data.select(['srcip', 'dstip']).to_pandas()
    
    # Track edges as (src_idx, dst_idx) -> weight
    edge_counts = {}
    
    # Process in batches
    batch_size = 10000
    for i in range(0, len(edge_data), batch_size):
        batch = edge_data.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            src = row['srcip']
            dst = row['dstip']
            
            if src != dst and src in node_mapping and dst in node_mapping:  # Avoid self-loops
                src_idx = node_mapping[src]
                dst_idx = node_mapping[dst]
                
                # Count occurrences of this edge for weight
                edge_key = (src_idx, dst_idx)
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
    
    # Add all edges at once - more efficient in RustworkX
    for (src_idx, dst_idx), weight in edge_counts.items():
        G.add_edge(src_idx, dst_idx, weight)
    
    print(f"Added {G.num_edges()} edges in {time.time() - edge_time:.2f}s")
    print(f"Created graph with {len(G)} nodes and {G.num_edges()} edges in {time.time() - start_time:.2f}s")
    
    # Save graph for future use
    print(f"Saving graph to {GRAPH_CACHE}")
    save_graph(G, GRAPH_CACHE)
    
    return G, node_mapping

def visualize_graph(G, node_mapping, max_nodes=100, save_path=None):
    """
    Visualize the graph with optimized rendering for large graphs
    
    Parameters:
    -----------
    G: rustworkx.PyGraph
        Graph to visualize
    node_mapping: dict
        Mapping from device IPs to node IDs
    max_nodes: int
        Maximum number of nodes to show
    save_path: str
        Path to save the visualization (default: processed/graph_viz.png)
    """
    # Default save path
    if save_path is None:
        save_path = os.path.join(PROCESSED_DIR, 'graph_viz.png')
    
    # Check if visualization already exists
    if os.path.exists(save_path) and os.path.getmtime(save_path) > os.path.getmtime(GRAPH_CACHE):
        print(f"Using existing graph visualization: {save_path}")
        return
        
    print("Generating graph visualization...")
    start_time = time.time()
    
    # For very large graphs, we might want to sample nodes based on importance
    if len(G) > max_nodes * 10:
        print(f"Graph is very large ({len(G)} nodes). Sampling important nodes...")
        
        # Compute node degrees
        node_degrees = {}
        for idx in range(len(G)):
            node_degrees[idx] = len(list(G.neighbors(idx)))
        
        # Select nodes with highest degrees and nodes with attacks
        high_degree_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)[:max_nodes//2]
        
        # Find nodes with attack labels
        attack_nodes = []
        for idx in range(len(G)):
            if len(attack_nodes) >= max_nodes//2:
                break
            node_data = G.get_node_data(idx)
            if node_data.get('label', 0) == 1 and idx not in high_degree_nodes:
                attack_nodes.append(idx)
        
        # Combine selected nodes
        selected_nodes = list(high_degree_nodes) + attack_nodes
        selected_nodes = selected_nodes[:max_nodes]
    else:
        # Just take the first max_nodes
        selected_nodes = list(range(min(len(G), max_nodes)))
    
    # Convert to NetworkX for visualization
    G_nx = nx.Graph()
    
    # Add nodes with attributes
    for i in selected_nodes:
        node_data = G.get_node_data(i)
        label = node_data.get('label', 0)
        G_nx.add_node(i, label=label)
    
    # Add edges between selected nodes
    edge_count = 0
    for edge in G.edge_list():
        source, target = edge[0], edge[1]
        if source in selected_nodes and target in selected_nodes:
            weight = G.get_edge_data(source, target)
            G_nx.add_edge(source, target, weight=weight)
            edge_count += 1
    
    print(f"Visualizing {len(G_nx)} nodes and {edge_count} edges")
    
    # Create figure
    plt.figure(figsize=(16, 12), dpi=100)
    
    # Use a faster layout algorithm for large graphs
    if len(G_nx) > 500:
        print("Using sfdp layout algorithm for large graph...")
        try:
            # Try using graphviz if available (much faster for large graphs)
            import pygraphviz as pgv
            A = pgv.AGraph()
            for n in G_nx.nodes():
                A.add_node(n)
            for u, v in G_nx.edges():
                A.add_edge(u, v)
            A.layout(prog='sfdp')
            
            # Convert positions
            pos = {}
            for n in G_nx.nodes():
                node = A.get_node(n)
                pos[n] = (float(node.attr['pos'].split(',')[0]), 
                         float(node.attr['pos'].split(',')[1]))
        except ImportError:
            print("pygraphviz not available, using spring layout...")
            pos = nx.spring_layout(G_nx, seed=42)
    else:
        pos = nx.spring_layout(G_nx, seed=42)
    
    # Get node colors based on labels - red for attacks, blue for normal
    node_colors = [G_nx.nodes[n].get('label', 0) for n in G_nx.nodes()]
    
    # Draw network
    plt.figure(figsize=(14, 12))
    
    # Draw nodes with attack nodes larger
    node_sizes = [300 if G_nx.nodes[n].get('label', 0) == 1 else 100 for n in G_nx.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G_nx, pos, 
                          node_color=node_colors, 
                          cmap=plt.cm.coolwarm, 
                          alpha=0.8,
                          node_size=node_sizes)
    
    # Get edge weights, with maximum value for normalization
    edge_weights = [G_nx[u][v].get('weight', 1) for u, v in G_nx.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    
    # Draw edges with width based on weight
    edge_widths = [min(G_nx[u][v].get('weight', 1) / max_weight * 3, 3) for u, v in G_nx.edges()]
    nx.draw_networkx_edges(G_nx, pos, width=edge_widths, alpha=0.3, edge_color='gray')
    
    # Show labels for high-degree nodes if not too many
    if len(G_nx) <= 50:
        # Find top 10 nodes by degree
        top_nodes = sorted(G_nx.degree, key=lambda x: x[1], reverse=True)[:10]
        labels = {n: f"Node {n}" for n, _ in top_nodes}
        nx.draw_networkx_labels(G_nx, pos, labels=labels, font_size=8)
    
    plt.title(f"Network Graph from UNSW-NB15 (showing {len(G_nx)} nodes out of {len(G)})")
    plt.axis('off')
    
    # Add custom legend for attack vs. normal nodes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal Traffic', 
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Attack Traffic', 
               markerfacecolor='red', markersize=15)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add custom colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, label='Attack Label (1=Attack)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Graph visualization saved to {save_path} in {time.time() - start_time:.2f}s")
    
    # Keep memory usage down
    plt.close()

def save_graph(G, filepath):
    """
    Save the graph to a file with optimized storage for GPU compatibility
    
    Parameters:
    -----------
    G: rustworkx.PyGraph
        Graph to save
    filepath: str
        Path to save the graph
    """
    start_time = time.time()
    print(f"Saving graph with {len(G)} nodes and {G.num_edges()} edges...")
    
    # Prepare node data for GPU-compatible format
    node_data_list = []
    for i in range(len(G)):
        node_data = G.get_node_data(i)
        if node_data:
            # Convert numpy arrays to torch tensors for GPU compatibility
            if 'features' in node_data and isinstance(node_data['features'], np.ndarray):
                node_data = node_data.copy()  # Create copy to avoid modifying original
                node_data['features'] = node_data['features'].tolist()
            node_data_list.append(node_data)
        else:
            node_data_list.append({})
    
    # Prepare edge list and data in efficient format
    edge_list = G.edge_list()
    edge_data = [G.get_edge_data(e[0], e[1]) for e in edge_list]
    
    # Prepare adjacency structure for faster graph algorithms
    # Store as sparse format: {node_id: [neighbor_ids]}
    adjacency = {}
    for i in range(len(G)):
        adjacency[i] = list(G.neighbors(i))
    
    # Compile data dictionary
    data = {
        'num_nodes': len(G),
        'edge_list': edge_list,
        'node_data': node_data_list,
        'edge_data': edge_data,
        'adjacency': adjacency,
        'metadata': {
            'creation_time': time.time(),
            'description': 'UNSW-NB15 Network Graph',
            'node_count': len(G),
            'edge_count': G.num_edges()
        }
    }
    
    # Save with torch - compatible with both CPU and GPU
    torch.save(data, filepath)
    print(f"Graph saved to {filepath} in {time.time() - start_time:.2f}s")
    
    # Also save a compressed copy for backup
    compressed_path = f"{filepath}.gz"
    try:
        import gzip
        import pickle
        with gzip.open(compressed_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Compressed backup saved to {compressed_path}")
    except Exception as e:
        print(f"Note: Couldn't create compressed backup: {e}")

def load_graph(filepath, device=None):
    # If file ends with .pt but is actually pickle, try pickle first
    if filepath.endswith(".pt"):
        try:
            import pickle
            print(f"Trying pickle for {filepath}...")
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            print(f"Successfully loaded with pickle: {len(data["node_data"])} nodes")
            
            # Create new graph
            G = rx.PyGraph()
            
            # Add nodes with data
            for node_data in data["node_data"]:
                G.add_node(node_data)
            
            # Add edges
            for (src, dst), edge_data in zip(data["edge_list"], data["edge_data"]):
                G.add_edge(src, dst, edge_data)
            
            print(f"Loaded graph with {len(G)} nodes and {G.num_edges()} edges")
            return G
        except Exception as pickle_e:
            print(f"Pickle loading failed: {pickle_e}, trying torch...")

    """
    Load a graph from a file with GPU compatibility
    
    Parameters:
    -----------
    filepath: str
        Path to the saved graph
    device: torch.device
        Device to load tensors to (None for CPU, 'cuda' for GPU)
        
    Returns:
    --------
    G: rustworkx.PyGraph
        Loaded graph
    """
    start_time = time.time()
    
    # Try to load with torch first (faster)
    try:
        # Use torch.load with map_location to control device placement
        if device:
            data = torch.load(filepath, map_location=device, weights_only=False)
        else:
            data = torch.load(filepath)
    except Exception as e:
        print(f"Error loading with torch: {e}")
        # Fallback to compressed backup if available
        compressed_path = f"{filepath}.gz"
        if os.path.exists(compressed_path):
            print(f"Trying compressed backup: {compressed_path}")
            import gzip
            import pickle
            with gzip.open(compressed_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise e
    
    # Create new graph
    G = rx.PyGraph()
    
    # Add nodes with proper data type conversion
    for node_data in data['node_data']:
        # Convert features back to numpy array if needed
        node_data_copy = node_data.copy()  # Create copy to avoid modifying original
        if 'features' in node_data_copy and isinstance(node_data_copy['features'], list):
            node_data_copy['features'] = np.array(node_data_copy['features'], dtype=np.float32)
        
        # Convert any torch tensors to numpy arrays
        for key, value in node_data_copy.items():
            if isinstance(value, torch.Tensor):
                node_data_copy[key] = value.cpu().numpy()
        
        G.add_node(node_data_copy)
    
    # Add edges
    for (src, dst), edge_data in zip(data['edge_list'], data['edge_data']):
        G.add_edge(src, dst, edge_data)
    
    # Check if adjacency information is available to validate
    if 'adjacency' in data:
        # Verify graph integrity
        for node_id, neighbors in data['adjacency'].items():
            # Skip if node_id is out of range
            if node_id >= len(G):
                continue
                
            graph_neighbors = set(G.neighbors(node_id))
            stored_neighbors = set(neighbors)
            
            if graph_neighbors != stored_neighbors:
                print(f"Warning: Adjacency mismatch for node {node_id}")
                print(f"  Graph has {len(graph_neighbors)} neighbors, stored has {len(stored_neighbors)}")
    
    print(f"Loaded graph with {len(G)} nodes and {G.num_edges()} edges in {time.time() - start_time:.2f}s")
    return G

def create_gpu_ready_data(G, output_dir=None):
    """
    Create GPU-ready data for faster GNN training
    
    Parameters:
    -----------
    G: rustworkx.PyGraph
        Graph to convert
    output_dir: str
        Directory to save the data (default: processed/gpu_data)
        
    Returns:
    --------
    data: dict
        Dictionary with GPU-ready data
    """
    if output_dir is None:
        output_dir = os.path.join(PROCESSED_DIR, "gpu_data")
    
    Path(output_dir).mkdir(exist_ok=True)
    gpu_data_path = os.path.join(output_dir, "gpu_ready_data.pt")
    
    # Check if GPU data already exists
    if os.path.exists(gpu_data_path):
        print(f"Loading existing GPU-ready data from {gpu_data_path}")
        return torch.load(gpu_data_path, map_location='cpu')
    
    print("Creating GPU-ready data for GNN training...")
    start_time = time.time()
    
    # Extract features and labels
    features = []
    labels = []
    for i in range(len(G)):
        node_data = G.get_node_data(i)
        if 'features' in node_data:
            features.append(node_data['features'])
        else:
            features.append(np.zeros(10))  # Default features
        
        labels.append(node_data.get('label', 0))
    
    # Convert to torch tensors
    x = torch.tensor(np.array(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    # Create edge index for PyTorch Geometric format
    edge_index = []
    edge_attr = []
    
    for src, dst in G.edge_list():
        weight = G.get_edge_data(src, dst)
        edge_index.append([src, dst])
        edge_attr.append(weight)
    
    # Convert to PyTorch Geometric format
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    # Create data dictionary compatible with PyTorch Geometric
    data = {
        'x': x,  # Node features [num_nodes, num_features]
        'y': y,  # Node labels [num_nodes]
        'edge_index': edge_index,  # Graph connectivity [2, num_edges]
        'edge_attr': edge_attr,  # Edge features [num_edges]
        'num_nodes': len(G)
    }
    
    # Save the data
    torch.save(data, gpu_data_path)
    print(f"GPU-ready data created and saved to {gpu_data_path} in {time.time() - start_time:.2f}s")
    
    return data

def main():
    """Main function to process the UNSW-NB15 dataset and create a graph"""
    # First, check for previously processed data
    if os.path.exists(GRAPH_CACHE):
        print(f"Found existing processed graph at {GRAPH_CACHE}")
        G = load_graph(GRAPH_CACHE)
        
        # Determine node mapping from graph
        node_mapping = {}
        for i in range(len(G)):
            node_data = G.get_node_data(i)
            if 'ip' in node_data:
                node_mapping[node_data['ip']] = i
        
        # Create GPU-ready data
        create_gpu_ready_data(G)
        
        # Visualize the graph if no visualization exists
        viz_path = os.path.join(PROCESSED_DIR, 'graph_viz.png')
        if not os.path.exists(viz_path):
            visualize_graph(G, node_mapping, save_path=viz_path)
        
        print("Processing completed. Use the graph for community detection.")
        return G, node_mapping
    
    # Otherwise, process the dataset from scratch
    print("No processed data found. Processing UNSW-NB15 dataset from scratch...")
    
    # Load dataset
    data, features_info = load_dataset(UNSW_DATA_PATHS, UNSW_FEATURES_PATH)
    
    if data is None:
        print("No data loaded. Please download the dataset first.")
        print("Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        return None, None
    
    # Perform feature selection
    selected_features, X_selected, y = feature_selection(data, k=20)
    
    # Construct graph
    G, node_mapping = construct_graph(data, selected_features)
    
    # Create GPU-ready data
    create_gpu_ready_data(G)
    
    # Visualize graph
    visualize_graph(G, node_mapping, save_path=os.path.join(PROCESSED_DIR, 'graph_viz.png'))
    
    print("Processing completed. Use the graph for community detection.")
    return G, node_mapping

if __name__ == "__main__":
    G, node_mapping = main()