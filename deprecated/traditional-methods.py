# Traditional Community Detection Methods
# ======================================

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import time
import warnings
warnings.filterwarnings('ignore')

# For community detection
try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

try:
    from cdlib import algorithms, evaluation
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False

# Import utility functions from the data preparation notebook
# In a real scenario, these would be imported from a module
from data_prep import (load_data, create_graph_from_edgelist, create_graph_from_adjacency,
                       generate_synthetic_graph, compute_graph_statistics, plot_graph)


def run_louvain(G):
    """
    Run the Louvain community detection algorithm
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph to analyze
        
    Returns:
    --------
    communities: dict
        Dictionary mapping node to community
    execution_time: float
        Execution time in seconds
    """
    start_time = time.time()
    
    if LOUVAIN_AVAILABLE:
        # Using python-louvain
        communities = community_louvain.best_partition(G)
    elif CDLIB_AVAILABLE:
        # Using cdlib
        result = algorithms.louvain(G)
        communities = {}
        for i, comm in enumerate(result.communities):
            for node in comm:
                communities[node] = i
    else:
        # Fallback to NetworkX
        try:
            from networkx.algorithms import community
            communities = community.louvain_communities(G)
            # Convert to dict format
            comm_dict = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    comm_dict[node] = i
            communities = comm_dict
        except:
            raise ImportError("No community detection library available. Please install python-louvain or cdlib.")
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_leiden(G):
    """
    Run the Leiden community detection algorithm
    
    Parameters:
    -----------
    G: NetworkX Graph
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
    result = algorithms.leiden(G)
    
    communities = {}
    for i, comm in enumerate(result.communities):
        for node in comm:
            communities[node] = i
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_infomap(G):
    """
    Run the Infomap community detection algorithm
    
    Parameters:
    -----------
    G: NetworkX Graph
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
    result = algorithms.infomap(G)
    
    communities = {}
    for i, comm in enumerate(result.communities):
        for node in comm:
            communities[node] = i
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_label_propagation(G):
    """
    Run the Label Propagation community detection algorithm
    
    Parameters:
    -----------
    G: NetworkX Graph
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
        result = algorithms.label_propagation(G)
        communities = {}
        for i, comm in enumerate(result.communities):
            for node in comm:
                communities[node] = i
    else:
        # Using NetworkX
        from networkx.algorithms import community
        result = community.label_propagation_communities(G)
        communities = {}
        for i, comm in enumerate(result):
            for node in comm:
                communities[node] = i
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_girvan_newman(G, k=None):
    """
    Run the Girvan-Newman community detection algorithm
    
    Parameters:
    -----------
    G: NetworkX Graph
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
        result = algorithms.girvan_newman(G, level=k)
        communities = {}
        for i, comm in enumerate(result.communities):
            for node in comm:
                communities[node] = i
    else:
        # Using NetworkX
        from networkx.algorithms import community
        comp = community.girvan_newman(G)
        
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


def run_spectral_clustering(G, n_clusters):
    """
    Run Spectral Clustering for community detection
    
    Parameters:
    -----------
    G: NetworkX Graph
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
    
    # Get adjacency matrix
    A = nx.to_numpy_array(G)
    
    # Run spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, 
                                  affinity='precomputed', 
                                  assign_labels='kmeans',
                                  random_state=42)
    labels = spectral.fit_predict(A)
    
    # Convert to dictionary format
    communities = {node: labels[i] for i, node in enumerate(G.nodes())}
    
    execution_time = time.time() - start_time
    return communities, execution_time


def run_walktrap(G):
    """
    Run the Walktrap community detection algorithm
    
    Parameters:
    -----------
    G: NetworkX Graph
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
    result = algorithms.walktrap(G)
    
    communities = {}
    for i, comm in enumerate(result.communities):
        for node in comm:
            communities[node] = i
    
    execution_time = time.time() - start_time
    return communities, execution_time


def evaluate_against_ground_truth(G, detected_communities, ground_truth_attr='community'):
    """
    Evaluate detected communities against ground truth
    
    Parameters:
    -----------
    G: NetworkX Graph
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
    ground_truth = nx.get_node_attributes(G, ground_truth_attr)
    
    # Ensure both dictionaries have the same keys
    nodes = list(G.nodes())
    true_labels = [ground_truth.get(n, -1) for n in nodes]
    pred_labels = [detected_communities.get(n, -1) for n in nodes]
    
    # Calculate metrics
    metrics = {}
    metrics['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
    metrics['ari'] = adjusted_rand_score(true_labels, pred_labels)
    
    # Calculate modularity
    if LOUVAIN_AVAILABLE:
        metrics['modularity'] = community_louvain.modularity(detected_communities, G)
    elif CDLIB_AVAILABLE:
        # Convert to cdlib format
        communities_list = []
        for comm_id in set(detected_communities.values()):
            comm = [n for n, c in detected_communities.items() if c == comm_id]
            communities_list.append(comm)
        
        communities = evaluation.NodeClustering(communities_list, G)
        metrics['modularity'] = evaluation.newman_girvan_modularity(communities).score
    else:
        metrics['modularity'] = "Not available (install python-louvain or cdlib)"
    
    return metrics


def plot_communities(G, communities, pos=None, figsize=(12, 10), title="Community Detection Results"):
    """
    Visualize detected communities
    
    Parameters:
    -----------
    G: NetworkX Graph
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
    
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    # Set node colors based on community
    cmap = plt.cm.rainbow
    node_colors = [communities[n] for n in G.nodes()]
    
    # Draw the graph
    nx.draw_networkx(G, pos=pos, node_color=node_colors, cmap=cmap, 
                     with_labels=True, node_size=100, font_size=8, alpha=0.8)
    
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def compare_methods(G, ground_truth_attr='community', n_clusters=None):
    """
    Compare different community detection methods
    
    Parameters:
    -----------
    G: NetworkX Graph
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
    if G.number_of_nodes() < 1000:
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
    
    return pd.DataFrame(results)


def save_communities(G, communities, output_path, save_graph=False):
    """
    Save detected communities to file
    
    Parameters:
    -----------
    G: NetworkX Graph
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
    # Save communities
    df = pd.DataFrame({
        'node': list(communities.keys()),
        'community': list(communities.values())
    })
    df.to_csv(f"{output_path}_communities.csv", index=False)
    
    # Save graph if requested
    if save_graph:
        # Add communities as node attributes
        for node, comm in communities.items():
            G.nodes[node]['detected_community'] = comm
        
        # Save as GraphML
        nx.write_graphml(G, f"{output_path}_graph.graphml")


# Main execution example
if __name__ == "__main__":
    print("Traditional Community Detection Methods")
    print("="*50)
    
    # Generate a synthetic graph with ground truth communities
    print("\n1. Generating a Stochastic Block Model graph...")
    n_communities = 5
    G, ground_truth = generate_synthetic_graph('sbm', n_nodes=100, n_communities=n_communities,
                                              p_in=0.3, p_out=0.05)
    
    # Visualize original graph with ground truth
    print("\n2. Visualizing ground truth communities...")
    plot_graph(G, community_attr='community', title="Ground Truth Communities")
    
    # Compare different community detection methods
    print("\n3. Comparing different community detection methods...")
    results = compare_methods(G, ground_truth_attr='community', n_clusters=n_communities)
    
    # Display comparison results
    print("\n4. Summary of results:")
    print(results)
    
    # Visualize comparison results
    plt.figure(figsize=(12, 6))
    
    # Plot NMI and ARI
    plt.subplot(1, 2, 1)
    results.plot(x='Method', y=['NMI', 'ARI'], kind='bar', ax=plt.gca())
    plt.title('Quality Metrics by Method')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Plot execution time
    plt.subplot(1, 2, 2)
    results.plot(x='Method', y='Execution Time (s)', kind='bar', ax=plt.gca(), color='green')
    plt.title('Execution Time by Method')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Run on a real-world network if available
    try:
        # Replace with your actual data loading code
        # G_real = load_data("path/to/your/network/data")
        # compare_methods(G_real)
        pass
    except:
        print("\nNo real-world network data available.")
    
    print("\nAnalysis complete!")
