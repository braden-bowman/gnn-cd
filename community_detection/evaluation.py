# Comprehensive Evaluation of Community Detection Methods
# ======================================================

import os
import torch
import polars as pl
import rustworkx as rx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import time
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

# For community evaluation
try:
    from cdlib import evaluation
    from cdlib.classes import NodeClustering
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False
    print("cdlib not available. Some evaluation metrics will be limited.")

# For interactive/dynamic visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Constants for method categorization
METHOD_CATEGORIES = {
    'traditional': ['Louvain', 'Leiden', 'Infomap', 'Label Propagation', 'Spectral Clustering'],
    'embedding': ['Node2Vec+KMeans', 'DeepWalk+KMeans'],
    'gnn': ['GCN', 'GraphSAGE', 'GAT', 'VGAE'],
    'dynamic': ['EvolveGCN', 'DySAT'],
    'overlapping': ['BigCLAM', 'DEMON', 'SLPA', 'GNN-Overlapping']
}


# Function to load results from previous notebook runs
def load_results(results_dir: str) -> Dict[str, Any]:
    """
    Load results from previous notebook runs
    
    Parameters:
    -----------
    results_dir: str
        Directory containing saved results
        
    Returns:
    --------
    all_results: dict
        Dictionary containing results from all methods
    """
    all_results = {}
    
    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found. Creating it.")
        os.makedirs(results_dir)
        return all_results
    
    # Load all parquet and pickle files in directory
    for filename in os.listdir(results_dir):
        filepath = os.path.join(results_dir, filename)
        try:
            if filename.endswith('.parquet'):
                # Load parquet file
                method_name = filename.split('.')[0]
                result = pl.read_parquet(filepath).to_dict(as_series=False)
                all_results[method_name] = result
                print(f"Loaded results for {method_name} from parquet")
            elif filename.endswith('.pkl'):
                # Load pickle file
                method_name = filename.split('.')[0]
                with open(filepath, 'rb') as f:
                    result = pickle.load(f)
                all_results[method_name] = result
                print(f"Loaded results for {method_name} from pickle")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return all_results


# Function to save results for a specific method
def save_result(result: Dict[str, Any], method_name: str, results_dir: str = 'results'):
    """
    Save results for a specific method
    
    Parameters:
    -----------
    result: dict
        Results dictionary
    method_name: str
        Name of the method
    results_dir: str
        Directory to save results
    
    Returns:
    --------
    None
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save as parquet if possible (for numeric/tabular data)
    try:
        # Try to convert to DataFrame
        result_df = pl.DataFrame(result)
        filepath = os.path.join(results_dir, f"{method_name}.parquet")
        result_df.write_parquet(filepath, compression="zstd")
        print(f"Saved results for {method_name} to {filepath}")
    except:
        # Fall back to pickle for complex data structures
        filepath = os.path.join(results_dir, f"{method_name}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved results for {method_name} to {filepath}")


# Function to compile results into a single DataFrame
def compile_results(all_results: Dict[str, Dict]) -> pl.DataFrame:
    """
    Compile results from all methods into a single DataFrame
    
    Parameters:
    -----------
    all_results: dict
        Dictionary containing results from all methods
        
    Returns:
    --------
    df: DataFrame
        DataFrame containing compiled results
    """
    rows = []
    
    for method_name, result in all_results.items():
        # Skip if result doesn't have 'metrics' key
        if 'metrics' not in result:
            continue
        
        # Determine method category
        category = 'other'
        for cat, methods in METHOD_CATEGORIES.items():
            if any(m.lower() in method_name.lower() for m in methods):
                category = cat
                break
        
        # Create row for method
        row = {
            'Method': method_name,
            'Category': category,
            'Training Time (s)': result.get('training_time', 0),
            'Execution Time (s)': result.get('execution_time', 0),
            'Num Communities': result.get('num_communities', 0)
        }
        
        # Add metrics
        metrics = result['metrics']
        for metric_name, value in metrics.items():
            row[metric_name] = value
        
        rows.append(row)
    
    return pl.DataFrame(rows)


# Function to generate a heatmap of method performance
def plot_heatmap(df: pl.DataFrame, metrics: List[str] = ['nmi', 'ari', 'modularity'], 
               figsize: Tuple[int, int] = (12, 8)):
    """
    Generate a heatmap of method performance
    
    Parameters:
    -----------
    df: DataFrame
        DataFrame containing compiled results
    metrics: list
        List of metrics to include in heatmap
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    # Convert to pandas for seaborn
    pd_df = df.to_pandas()
    
    # Filter to only include specified metrics
    metrics_df = pd_df[['Method', 'Category'] + metrics].set_index('Method')
    
    # Sort by category and method name
    metrics_df = metrics_df.sort_values(['Category', 'Method'])
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(metrics_df[metrics], annot=True, cmap='viridis', fmt='.3f',
               linewidths=0.5, cbar_kws={'label': 'Score'})
    plt.title('Performance Comparison of Community Detection Methods')
    plt.tight_layout()
    plt.show()


# Function to generate a radar chart of method performance
def plot_radar_chart(df: pl.DataFrame, metrics: List[str] = ['nmi', 'ari', 'modularity'], 
                    n_methods: int = 5, figsize: Tuple[int, int] = (14, 10)):
    """
    Generate a radar chart of method performance
    
    Parameters:
    -----------
    df: DataFrame
        DataFrame containing compiled results
    metrics: list
        List of metrics to include in radar chart
    n_methods: int
        Number of top methods to include
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Cannot generate radar chart.")
        return
    
    # Convert to pandas for easier filtering
    pd_df = df.to_pandas()
    
    # Select top N methods based on average of metrics
    available_metrics = [m for m in metrics if m in pd_df.columns]
    if len(available_metrics) == 0:
        print("No valid metrics found for radar chart.")
        return
    
    pd_df['avg_score'] = pd_df[available_metrics].mean(axis=1)
    top_methods = pd_df.sort_values('avg_score', ascending=False).head(n_methods)
    
    # Prepare data for radar chart
    categories = available_metrics
    fig = go.Figure()
    
    for _, row in top_methods.iterrows():
        method_name = row['Method']
        values = [row[m] for m in available_metrics]
        # Close the polygon
        values.append(values[0])
        categories_closed = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill='toself',
            name=method_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Radar Chart of Method Performance",
        showlegend=True,
        width=figsize[0]*100,
        height=figsize[1]*100
    )
    
    fig.show()


# Function to plot performance vs. execution time
def plot_performance_vs_time(df: pl.DataFrame, metric: str = 'nmi', 
                           figsize: Tuple[int, int] = (12, 8)):
    """
    Plot performance vs. execution time for all methods
    
    Parameters:
    -----------
    df: DataFrame
        DataFrame containing compiled results
    metric: str
        Metric to use for performance
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    # Convert to pandas for easier plotting
    pd_df = df.to_pandas()
    
    if metric not in pd_df.columns:
        print(f"Metric {metric} not found in results.")
        return
    
    plt.figure(figsize=figsize)
    
    # Create scatterplot
    sns.scatterplot(x='Execution Time (s)', y=metric, 
                   hue='Category', size='Num Communities',
                   data=pd_df, alpha=0.7, sizes=(50, 200))
    
    # Add method names as annotations
    for _, row in pd_df.iterrows():
        plt.annotate(row['Method'], 
                    (row['Execution Time (s)'], row[metric]),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.title(f'{metric.upper()} vs. Execution Time')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel(metric.upper())
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Function to plot performance by category
def plot_performance_by_category(df: pl.DataFrame, 
                               metrics: List[str] = ['nmi', 'ari', 'modularity'], 
                               figsize: Tuple[int, int] = (14, 8)):
    """
    Plot performance by method category
    
    Parameters:
    -----------
    df: DataFrame
        DataFrame containing compiled results
    metrics: list
        List of metrics to include
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    # Convert to pandas for easier plotting
    pd_df = df.to_pandas()
    
    # Check which metrics are available
    available_metrics = [m for m in metrics if m in pd_df.columns]
    if len(available_metrics) == 0:
        print("No specified metrics found in results.")
        return
    
    # Prepare data for plotting
    melted_df = pd.melt(pd_df, id_vars=['Method', 'Category'], 
                       value_vars=available_metrics,
                       var_name='Metric', value_name='Score')
    
    plt.figure(figsize=figsize)
    sns.boxplot(x='Category', y='Score', hue='Metric', data=melted_df)
    plt.title('Performance by Method Category')
    plt.xlabel('Method Category')
    plt.ylabel('Score')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Function to plot top methods comparison
def plot_top_methods_comparison(df: pl.DataFrame, metric: str = 'nmi', 
                              n_methods: int = 5, figsize: Tuple[int, int] = (12, 6)):
    """
    Generate a bar chart comparing top methods for a specific metric
    
    Parameters:
    -----------
    df: DataFrame
        DataFrame containing compiled results
    metric: str
        Metric to use for comparison
    n_methods: int
        Number of top methods to include
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    # Convert to pandas for easier plotting
    pd_df = df.to_pandas()
    
    if metric not in pd_df.columns:
        print(f"Metric {metric} not found in results.")
        return
    
    # Select top methods
    top_methods = pd_df.sort_values(metric, ascending=False).head(n_methods)
    
    plt.figure(figsize=figsize)
    sns.barplot(x='Method', y=metric, hue='Category', data=top_methods)
    plt.title(f'Top {n_methods} Methods by {metric.upper()}')
    plt.xlabel('Method')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Function to generate a summary table of results
def generate_summary_table(df: pl.DataFrame, 
                         metrics: List[str] = ['nmi', 'ari', 'modularity']):
    """
    Generate a summary table of results
    
    Parameters:
    -----------
    df: DataFrame
        DataFrame containing compiled results
    metrics: list
        List of metrics to include
        
    Returns:
    --------
    summary: DataFrame
        Summary table
    """
    # Convert to pandas for easier pivoting
    pd_df = df.to_pandas()
    
    # Check which metrics are available
    available_metrics = [m for m in metrics if m in pd_df.columns]
    if len(available_metrics) == 0:
        print("No specified metrics found in results.")
        return None
    
    # Create summary for each category
    categories = pd_df['Category'].unique()
    rows = []
    
    for category in categories:
        category_df = pd_df[pd_df['Category'] == category]
        
        # Find best method for each metric
        for metric in available_metrics:
            if metric in category_df.columns:
                best_method = category_df.loc[category_df[metric].idxmax()]
                rows.append({
                    'Category': category,
                    'Metric': metric,
                    'Best Method': best_method['Method'],
                    'Score': best_method[metric],
                    'Execution Time (s)': best_method['Execution Time (s)']
                })
    
    summary_df = pd.DataFrame(rows)
    
    # Create pivot table
    pivot = summary_df.pivot_table(
        index='Category', 
        columns='Metric', 
        values=['Best Method', 'Score'],
        aggfunc='first'
    )
    
    # Convert back to polars
    return pl.from_pandas(pivot.reset_index())


# Function to generate evaluation report
def generate_evaluation_report(df: pl.DataFrame, output_dir: str = 'report'):
    """
    Generate comprehensive evaluation report
    
    Parameters:
    -----------
    df: DataFrame
        DataFrame containing compiled results
    output_dir: str
        Directory to save report
        
    Returns:
    --------
    None
    """
    # Convert to pandas for easier aggregation
    pd_df = df.to_pandas()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Generate summary statistics
    if 'nmi' in pd_df.columns and 'ari' in pd_df.columns:
        summary = pd_df.groupby('Category').agg({
            'Method': 'count',
            'nmi': ['mean', 'std', 'min', 'max'],
            'ari': ['mean', 'std', 'min', 'max'],
            'Execution Time (s)': ['mean', 'min', 'max']
        }).reset_index()
        
        # Save as parquet
        summary_pl = pl.from_pandas(summary)
        summary_pl.write_parquet(os.path.join(output_dir, 'summary_statistics.parquet'), 
                             compression="zstd")
    
    # 2. Generate top performers for each metric
    metrics = ['nmi', 'ari', 'modularity']
    available_metrics = [m for m in metrics if m in pd_df.columns]
    
    for metric in available_metrics:
        top = pd_df.sort_values(metric, ascending=False).head(5)
        # Save as parquet
        top_pl = pl.from_pandas(top)
        top_pl.write_parquet(os.path.join(output_dir, f'top_{metric}.parquet'), 
                         compression="zstd")
    
    # 3. Generate visualizations
    plt.figure(figsize=(12, 8))
    plot_heatmap(df, metrics=available_metrics)
    plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'))
    
    plt.figure(figsize=(12, 8))
    plot_performance_by_category(df, metrics=available_metrics)
    plt.savefig(os.path.join(output_dir, 'performance_by_category.png'))
    
    for metric in available_metrics:
        plt.figure(figsize=(12, 8))
        plot_performance_vs_time(df, metric=metric)
        plt.savefig(os.path.join(output_dir, f'{metric}_vs_time.png'))
    
    # 4. Generate overall best performers
    if len(available_metrics) > 0:
        pd_df['avg_score'] = pd_df[available_metrics].mean(axis=1)
        best_overall = pd_df.sort_values('avg_score', ascending=False).head(5)
        # Save as parquet
        best_overall_pl = pl.from_pandas(best_overall)
        best_overall_pl.write_parquet(os.path.join(output_dir, 'best_overall.parquet'), 
                                  compression="zstd")
    
    print(f"Evaluation report generated in {output_dir}")


# Function to evaluate communities against ground truth
def evaluate_communities(G: rx.PyGraph, communities: Dict[int, int], 
                       ground_truth_attr: str = 'community') -> Dict[str, float]:
    """
    Evaluate detected communities against ground truth
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph with community information
    communities: dict or list
        Dictionary mapping node to detected community or list of community lists
    ground_truth_attr: str
        Node attribute for ground truth community
        
    Returns:
    --------
    metrics: dict
        Evaluation metrics
    """
    # Convert list-based communities to dict if necessary
    if isinstance(communities, list):
        comm_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_dict[node] = i
        communities = comm_dict
    
    # Get ground truth
    ground_truth = {}
    for i in range(len(G)):
        node_data = G.get_node_data(i)
        if node_data and ground_truth_attr in node_data:
            ground_truth[i] = node_data[ground_truth_attr]
    
    # Ensure both dictionaries have the same keys
    nodes = list(range(len(G)))
    true_labels = torch.tensor([ground_truth.get(n, -1) for n in nodes])
    pred_labels = torch.tensor([communities.get(n, -1) for n in nodes])
    
    # Calculate metrics
# Calculate metrics
    metrics = {}
    metrics['nmi'] = normalized_mutual_info_score(true_labels.numpy(), pred_labels.numpy())
    metrics['ari'] = adjusted_rand_score(true_labels.numpy(), pred_labels.numpy())
    
    # Calculate modularity
    if CDLIB_AVAILABLE:
        # Convert to NetworkX and cdlib format
        import networkx as nx
        G_nx = nx.Graph()
        
        # Add nodes
        for i in range(len(G)):
            G_nx.add_node(i)
        
        # Add edges
        for edge in G.edge_list():
            source, target = edge[0], edge[1]
            G_nx.add_edge(source, target)
        
        # Create community lists
        communities_list = []
        for comm_id in set(communities.values()):
            comm = [n for n, c in communities.items() if c == comm_id]
            communities_list.append(comm)
            
        communities_obj = NodeClustering(communities_list, G_nx)
        metrics['modularity'] = evaluation.newman_girvan_modularity(communities_obj).score
    else:
        # Fallback calculation if cdlib not available
        try:
            import networkx as nx
            import community as community_louvain
            
            G_nx = nx.Graph()
            
            # Add nodes
            for i in range(len(G)):
                G_nx.add_node(i)
            
            # Add edges
            for edge in G.edge_list():
                source, target = edge[0], edge[1]
                G_nx.add_edge(source, target)
                
            metrics['modularity'] = community_louvain.modularity(communities, G_nx)
        except Exception as e:
            metrics['modularity'] = float('nan')
            print(f"Modularity calculation failed: {e}")
    
    return metrics


# Function to visualize community stability across different methods
def visualize_community_stability(all_results: Dict[str, Dict], 
                                reference_method: str, n_methods: int = 5, 
                                figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize community stability across different methods
    
    Parameters:
    -----------
    all_results: dict
        Dictionary containing results from all methods
    reference_method: str
        Method to use as reference
    n_methods: int
        Number of methods to compare
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    if reference_method not in all_results:
        print(f"Reference method {reference_method} not found in results.")
        return
    
    # Get communities from reference method
    reference_communities = all_results[reference_method]['communities']
    
    # Calculate ARI between reference and all other methods
    stability_scores = {}
    
    for method_name, result in all_results.items():
        if 'communities' in result and method_name != reference_method:
            try:
                # Convert communities to torch tensors if they're dictionaries
                if isinstance(reference_communities, dict) and isinstance(result['communities'], dict):
                    nodes = sorted(reference_communities.keys())
                    ref_labels = torch.tensor([reference_communities[n] for n in nodes])
                    method_labels = torch.tensor([result['communities'][n] for n in nodes])
                    
                    ari = adjusted_rand_score(ref_labels.numpy(), method_labels.numpy())
                    stability_scores[method_name] = ari
            except Exception as e:
                print(f"Error calculating stability for {method_name}: {e}")
    
    # Sort by stability score and select top N
    sorted_methods = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
    top_methods = sorted_methods[:n_methods]
    
    # Visualize
    plt.figure(figsize=figsize)
    methods, scores = zip(*top_methods)
    
    # Convert to pandas for easier plotting
    import pandas as pd
    plot_df = pd.DataFrame({'Method': methods, 'ARI': scores})
    
    sns.barplot(x='Method', y='ARI', data=plot_df)
    plt.title(f'Community Stability Compared to {reference_method}')
    plt.xlabel('Method')
    plt.ylabel('Adjusted Rand Index')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()