# Traditional Community Detection Methods
# ======================================

import torch
import polars as pl
import rustworkx as rx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
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

# For community detection
try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

try:
    from cdlib import algorithms, evaluation
    from cdlib.classes import NodeClustering
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False


def _rwx_to_nx(G: rx.PyGraph):
    """
    Convert RustWorkX graph to NetworkX for algorithms that require it
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to convert
        
    Returns:
    --------
    G_nx: NetworkX Graph
        Converted graph
    """
    import networkx as nx
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
    
    return G_nx


def _nx_to_rwx(G_nx):
    """
    Convert NetworkX graph to RustWorkX
    
    Parameters:
    -----------
    G_nx: NetworkX Graph
        The graph to convert
        
    Returns:
    --------
    G: RustWorkX PyGraph
        Converted graph
    """
    G = rx.PyGraph()
    
    # Add nodes with attributes
    node_mapping = {}
    for node in G_nx.nodes():
        attrs = G_nx.nodes[node]
        node_mapping[node] = G.add_node(attrs if attrs else None)
    
    # Add edges with attributes
    for u, v, data in G_nx.edges(data=True):
        G.add_edge(node_mapping[u], node_mapping[v], data if data else None)
    
    return G


def run_louvain(G: rx.PyGraph) -> Tuple[Dict[int, int], float]:
    """
    Run the Louvain community detection algorithm
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
        
    Returns:
    --------
    communities: dict
        Dictionary mapping node to community
    execution_time: float
        Execution time in seconds
    """
    start_time = time.time()
    
    # For large graphs, try cugraph if available
    if CUGRAPH_AVAILABLE and len(G) > 10000:
        try:
            # Convert to NetworkX first
            G_nx = _rwx_to_nx(G)
            
            # Convert to cuGraph
            import cudf
            import pandas as pd
            src_list = []
            dst_list = []
            weight_list = []
            
            for u, v, data in G_nx.edges(data=True):
                src_list.append(u)
                dst_list.append(v)
                weight_list.append(data.get('weight', 1.0) if data else 1.0)
            
            df = pd.DataFrame({'source': src_list, 'destination': dst_list, 'weight': weight_list})
            cudf_df = cudf.DataFrame(df)
            
            G_cu = cugraph.Graph()
            G_cu.from_cudf_edgelist(cudf_df, source='source', destination='destination', edge_attr='weight')
            
            # Run Louvain on cuGraph
            parts = cugraph.louvain(G_cu)
            
            # Convert to dictionary format
            # Convert to dictionary format
            parts_df = parts.to_pandas()
            communities = {row['vertex']: int(row['partition']) for _, row in parts_df.iterrows()}
            
            execution_time = time.time() - start_time
            return communities, execution_time
        except Exception as e:
            print(f"Error using cugraph Louvain: {e}. Falling back to standard implementation.")
    
    # Fall back to CPU-based implementation
    if LOUVAIN_AVAILABLE:
        # Using python-louvain
        G_nx = _rwx_to_nx(G)
        communities = community_louvain.best_partition(G_nx)
    elif CDLIB_AVAILABLE:
        # Using cdlib
        G_nx = _rwx_to_nx(G)
        result = algorithms.louvain(G_nx)
        communities = {}
        for i, comm in enumerate(result.communities):
            for node in comm:
                communities[node] = i
    else:
        # Fallback to NetworkX
        try:
            import networkx as nx
            from networkx.algorithms import community
            G_nx = _rwx_to_nx(G)
            communities_nx = community.louvain_communities(G_nx)
            # Convert to dict format
            communities = {}
            for i, comm in enumerate(communities_nx):
                for node in comm:
                    communities[node] = i
        except:
            raise ImportError("No community detection library available. Please install python-louvain or cdlib.")
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_leiden(G: rx.PyGraph) -> Tuple[Dict[int, int], float]:
    """
    Run the Leiden community detection algorithm
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
        
    Returns:
    --------
    communities: dict
        Dictionary mapping node to community
    execution_time: float
        Execution time in seconds
    """
    if not CDLIB_AVAILABLE:
        raise ImportError("cdlib is required for Leiden algorithm")
    
    start_time = time.time()
    G_nx = _rwx_to_nx(G)
    result = algorithms.leiden(G_nx)
    
    communities = {}
    for i, comm in enumerate(result.communities):
        for node in comm:
            communities[node] = i
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_infomap(G: rx.PyGraph) -> Tuple[Dict[int, int], float]:
    """
    Run the Infomap community detection algorithm
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
        
    Returns:
    --------
    communities: dict
        Dictionary mapping node to community
    execution_time: float
        Execution time in seconds
    """
    if not CDLIB_AVAILABLE:
        raise ImportError("cdlib is required for Infomap algorithm")
    
    start_time = time.time()
    G_nx = _rwx_to_nx(G)
    result = algorithms.infomap(G_nx)
    
    communities = {}
    for i, comm in enumerate(result.communities):
        for node in comm:
            communities[node] = i
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_label_propagation(G: rx.PyGraph) -> Tuple[Dict[int, int], float]:
    """
    Run the Label Propagation community detection algorithm
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
        
    Returns:
    --------
    communities: dict
        Dictionary mapping node to community
    execution_time: float
        Execution time in seconds
    """
    start_time = time.time()
    
    if CDLIB_AVAILABLE:
        # Using cdlib
        G_nx = _rwx_to_nx(G)
        result = algorithms.label_propagation(G_nx)
        communities = {}
        for i, comm in enumerate(result.communities):
            for node in comm:
                communities[node] = i
    else:
        # Using NetworkX
        import networkx as nx
        from networkx.algorithms import community
        G_nx = _rwx_to_nx(G)
        result = community.label_propagation_communities(G_nx)
        communities = {}
        for i, comm in enumerate(result):
            for node in comm:
                communities[node] = i
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_girvan_newman(G: rx.PyGraph, k: Optional[int] = None) -> Tuple[Dict[int, int], float]:
    """
    Run the Girvan-Newman community detection algorithm
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
    k: int
        Number of communities to detect
        
    Returns:
    --------
    communities: dict
        Dictionary mapping node to community
    execution_time: float
        Execution time in seconds
    """
    start_time = time.time()
    
    if CDLIB_AVAILABLE:
        # Using cdlib
        G_nx = _rwx_to_nx(G)
        result = algorithms.girvan_newman(G_nx, level=k)
        communities = {}
        for i, comm in enumerate(result.communities):
            for node in comm:
                communities[node] = i
    else:
        # Using NetworkX
        import networkx as nx
        from networkx.algorithms import community
        G_nx = _rwx_to_nx(G)
        comp = community.girvan_newman(G_nx)
        
        # Limit to k iterations if specified
        if k is not None:
            for _ in range(k-1):
                next(comp)
            communities_list = next(comp)
        else:
            # Default: get the first partition
            communities_list = next(comp)
        
        communities = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                communities[node] = i
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_spectral_clustering(G: rx.PyGraph, n_clusters: int) -> Tuple[Dict[int, int], float]:
    """
    Run Spectral Clustering for community detection
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
    n_clusters: int
        Number of communities to detect
        
    Returns:
    --------
    communities: dict
        Dictionary mapping node to community
    execution_time: float
        Execution time in seconds
    """
    start_time = time.time()
    
    # Get adjacency matrix using torch tensors
    n_nodes = len(G)
    adj = torch.zeros((n_nodes, n_nodes))
    
    for edge in G.edge_list():
        source, target = edge[0], edge[1]
        edge_data = G.get_edge_data(source, target)
        weight = edge_data.get('weight', 1.0) if edge_data else 1.0
        adj[source, target] = weight
        adj[target, source] = weight  # For undirected graph
    
    # Convert to numpy for sklearn
    adj_np = adj.numpy()
    
    # Run spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, 
                                  affinity='precomputed', 
                                  assign_labels='kmeans',
                                  random_state=42)
    labels = spectral.fit_predict(adj_np)
    
    # Convert to dictionary format
    communities = {i: int(label) for i, label in enumerate(labels)}
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_walktrap(G: rx.PyGraph) -> Tuple[Dict[int, int], float]:
    """
    Run the Walktrap community detection algorithm
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
        
    Returns:
    --------
    communities: dict
        Dictionary mapping node to community
    execution_time: float
        Execution time in seconds
    """
    if not CDLIB_AVAILABLE:
        raise ImportError("cdlib is required for Walktrap algorithm")
    
    start_time = time.time()
    G_nx = _rwx_to_nx(G)
    result = algorithms.walktrap(G_nx)
    
    communities = {}
    for i, comm in enumerate(result.communities):
        for node in comm:
            communities[node] = i
    
    execution_time = time.time() - start_time
    return communities, execution_time


def evaluate_against_ground_truth(G: rx.PyGraph, detected_communities: Dict[int, int], 
                                ground_truth_attr: str = 'community') -> Dict[str, float]:
    """
    Evaluate detected communities against ground truth
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph with community information
    detected_communities: dict
        Dictionary mapping node to detected community
    ground_truth_attr: str
        Node attribute name for ground truth community
        
    Returns:
    --------
    metrics: dict
        Dictionary of evaluation metrics
    """
    # Get ground truth communities
    ground_truth = {}
    for i in range(len(G)):
        node_data = G.get_node_data(i)
        if node_data and ground_truth_attr in node_data:
            ground_truth[i] = node_data[ground_truth_attr]
        else:
            ground_truth[i] = -1  # Default for missing labels
    
    # Ensure both dictionaries have the same keys
    nodes = list(range(len(G)))
    true_labels = torch.tensor([ground_truth.get(n, -1) for n in nodes])
    pred_labels = torch.tensor([detected_communities.get(n, -1) for n in nodes])
    
    # Calculate metrics
    metrics = {}
    metrics['nmi'] = normalized_mutual_info_score(true_labels.numpy(), pred_labels.numpy())
    metrics['ari'] = adjusted_rand_score(true_labels.numpy(), pred_labels.numpy())
    
    # Calculate modularity
    if LOUVAIN_AVAILABLE:
        G_nx = _rwx_to_nx(G)
        metrics['modularity'] = community_louvain.modularity(detected_communities, G_nx)
    elif CDLIB_AVAILABLE:
        # Convert to cdlib format
        G_nx = _rwx_to_nx(G)
        communities_list = []
        for comm_id in set(detected_communities.values()):
            comm = [n for n, c in detected_communities.items() if c == comm_id]
            communities_list.append(comm)
        
        communities = NodeClustering(communities_list, G_nx)
        metrics['modularity'] = evaluation.newman_girvan_modularity(communities).score
    else:
        metrics['modularity'] = "Not available (install python-louvain or cdlib)"
    
    return metrics


def plot_communities(G: rx.PyGraph, communities: Dict[int, int], pos: Optional[Dict] = None, 
                    figsize: Tuple[int, int] = (12, 10), title: str = "Community Detection Results"):
    """
    Visualize detected communities
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to visualize
    communities: dict
        Dictionary mapping node to community
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
    plt.figure(figsize=figsize)
    
    # Convert to NetworkX for visualization
    import networkx as nx
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
    
    if pos is None:
        pos = nx.spring_layout(G_nx, seed=42)
    
    # Set node colors based on community
    cmap = plt.cm.rainbow
    node_colors = [communities.get(n, 0) for n in G_nx.nodes()]
    
    # Draw the graph
    nx.draw_networkx(G_nx, pos=pos, node_color=node_colors, cmap=cmap, 
                     with_labels=True, node_size=100, font_size=8, alpha=0.8)
    
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def add_communities_to_graph(G: rx.PyGraph, communities: Dict[int, int], 
                           attr_name: str = 'detected_community') -> rx.PyGraph:
    """
    Add community assignments as node attributes to a graph
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to modify
    communities: dict
        Dictionary mapping node to community
    attr_name: str
        Attribute name to use for community assignment
        
    Returns:
    --------
    G: RustWorkX PyGraph
        The modified graph
    """
    # Create a new graph with the same structure but updated node data
    if isinstance(G, rx.PyDiGraph):
        new_G = rx.PyDiGraph()
    else:
        new_G = rx.PyGraph()
    
    # Add nodes with updated attributes
    node_mapping = {}
    for i in range(len(G)):
        # Get existing node data
        node_data = G.get_node_data(i)
        
        # Convert None to empty dict
        if node_data is None:
            node_data = {}
        elif not isinstance(node_data, dict):
            # If node_data is not a dict, create a dict with the value
            node_data = {'value': node_data}
        else:
            # Make a copy to avoid modifying the original
            node_data = dict(node_data)
        
        # Add community info if this node is in communities dict
        if i in communities:
            node_data[attr_name] = communities[i]
        
        # Add node with updated data
        node_mapping[i] = new_G.add_node(node_data)
    
    # Add edges
    for edge in G.edge_list():
        source, target = edge[0], edge[1]
        edge_data = G.get_edge_data(source, target)
        new_G.add_edge(node_mapping[source], node_mapping[target], edge_data)
    
    return new_G


def compare_methods(G: rx.PyGraph, ground_truth_attr: str = 'community', 
                  n_clusters: Optional[int] = None) -> pl.DataFrame:
    """
    Compare different community detection methods
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to analyze
    ground_truth_attr: str
        Node attribute name for ground truth community
    n_clusters: int
        Number of clusters for methods that require it
        
    Returns:
    --------
    results: DataFrame
        DataFrame with comparison results
    """
    methods = {
        'Louvain': run_louvain,
        'Label Propagation': run_label_propagation
    }
    
    # Add methods that require cdlib
    if CDLIB_AVAILABLE:
        methods.update({
            'Leiden': run_leiden,
            'Infomap': run_infomap,
            'Walktrap': run_walktrap
        })
    
    # Add Girvan-Newman (can be slow for large graphs)
    if len(G) < 1000:
        methods['Girvan-Newman'] = lambda g: run_girvan_newman(g, k=n_clusters)
    
    # Add spectral clustering if n_clusters is provided
    if n_clusters is not None:
        methods['Spectral Clustering'] = lambda g: run_spectral_clustering(g, n_clusters)
    
    # Run all methods and collect results
    results = []
    
    for method_name, method_func in methods.items():
        print(f"Running {method_name}...")
        try:
            communities, execution_time = method_func(G)
            
            # Add communities to graph
            add_communities_to_graph(G, communities)
            
            # Count number of detected communities
            n_communities = len(set(communities.values()))
            
            # Evaluate against ground truth
            metrics = evaluate_against_ground_truth(G, communities, ground_truth_attr)
            
            # Store results
            results.append({
                'Method': method_name,
                'Num Communities': n_communities,
                'NMI': metrics['nmi'],
                'ARI': metrics['ari'],
                'Modularity': metrics['modularity'] if isinstance(metrics['modularity'], float) else None,
                'Execution Time (s)': execution_time
            })
            
            # Visualize communities
            plot_communities(G, communities, title=f"{method_name} - {n_communities} communities")
            
        except Exception as e:
            print(f"Error running {method_name}: {e}")
    
    # Convert to polars DataFrame
    return pl.DataFrame(results)


def save_communities(G: rx.PyGraph, communities: Dict[int, int], 
                    output_path: str, save_graph: bool = False):
    """
    Save detected communities to file
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph analyzed
    communities: dict
        Dictionary mapping node to community
    output_path: str
        Path to save the results
    save_graph: bool
        Whether to also save the graph
        
    Returns:
    --------
    None
    """
    # Save communities to parquet (more efficient than CSV)
    df = pl.DataFrame({
        'node': list(communities.keys()),
        'community': list(communities.values())
    })
    
    # Save using parquet format
    df.write_parquet(f"{output_path}_communities.parquet", compression="zstd")
    
    # Save graph if requested
    if save_graph:
        # Add communities as node attributes
        for node, comm in communities.items():
            node_data = G.get_node_data(node) or {}
            node_data['detected_community'] = comm
            G.set_node_data(node, node_data)
        
        # Save as pickle file since rustworkx doesn't have GraphML support
        import pickle
        with open(f"{output_path}_graph.pkl", 'wb') as f:
            # Convert to NetworkX first
            G_nx = _rwx_to_nx(G)
            pickle.dump(G_nx, f)