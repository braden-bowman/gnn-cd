# Comprehensive Evaluation of Community Detection Methods
# ======================================================

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# For community evaluation
try:
    from cdlib import evaluation
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False
    print("cdlib not available. Please install it for evaluation metrics.")

# For interactive/dynamic visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import utility functions from other notebooks
# In a real scenario, these would be imported from a module
from data_prep import (load_data, generate_synthetic_graph, plot_graph)


# Constants for method categorization
METHOD_CATEGORIES = {
    'traditional': ['Louvain', 'Leiden', 'Infomap', 'Label Propagation', 'Spectral Clustering'],
    'embedding': ['Node2Vec+KMeans', 'DeepWalk+KMeans'],
    'gnn': ['GCN', 'GraphSAGE', 'GAT', 'VGAE'],
    'dynamic': ['EvolveGCN', 'DySAT'],
    'overlapping': ['BigCLAM', 'DEMON', 'SLPA', 'GNN-Overlapping']
}


# Function to load results from previous notebooks
def load_results(results_dir):
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
    
    # Load all pickle files in directory
    for filename in os.listdir(results_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    result = pickle.load(f)
                method_name = filename.split('.')[0]
                all_results[method_name] = result
                print(f"Loaded results for {method_name}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return all_results


# Function to save results for a specific method
def save_result(result, method_name, results_dir):
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
    
    filepath = os.path.join(results_dir, f"{method_name}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved results for {method_name} to {filepath}")


# Function to compile results into a single DataFrame
def compile_results(all_results):
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
    
    return pd.DataFrame(rows)


# Function to generate a heatmap of method performance
def plot_heatmap(df, metrics=['nmi', 'ari', 'modularity'], figsize=(12, 8)):
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
    # Filter to only include specified metrics
    metrics_df = df[['Method', 'Category'] + metrics].set_index('Method')
    
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
def plot_radar_chart(df, metrics=['nmi', 'ari', 'modularity'], 
                    n_methods=5, figsize=(14, 10)):
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
    
    # Select top N methods based on average of metrics
    for m in metrics:
        if m not in df.columns:
            metrics.remove(m)
    
    if len(metrics) == 0:
        print("No valid metrics found for radar chart.")
        return
    
    df['avg_score'] = df[metrics].mean(axis=1)
    top_methods = df.sort_values('avg_score', ascending=False).head(n_methods)
    
    # Prepare data for radar chart
    categories = metrics
    fig = go.Figure()
    
    for _, row in top_methods.iterrows():
        method_name = row['Method']
        values = [row[m] for m in metrics]
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
def plot_performance_vs_time(df, metric='nmi', figsize=(12, 8)):
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
    if metric not in df.columns:
        print(f"Metric {metric} not found in results.")
        return
    
    plt.figure(figsize=figsize)
    
    # Create scatterplot
    sns.scatterplot(x='Execution Time (s)', y=metric, 
                   hue='Category', size='Num Communities',
                   data=df, alpha=0.7, sizes=(50, 200))
    
    # Add method names as annotations
    for _, row in df.iterrows():
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
def plot_performance_by_category(df, metrics=['nmi', 'ari', 'modularity'], figsize=(14, 8)):
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
    # Check which metrics are available
    available_metrics = [m for m in metrics if m in df.columns]
    if len(available_metrics) == 0:
        print("No specified metrics found in results.")
        return
    
    # Prepare data for plotting
    melted_df = pd.melt(df, id_vars=['Method', 'Category'], 
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


# Function to evaluate methods on a specific dataset
def evaluate_on_dataset(G, ground_truth_attr='community', results_dir='results'):
    """
    Evaluate all available methods on a specific dataset
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph to analyze
    ground_truth_attr: str
        Node attribute for ground truth communities
    results_dir: str
        Directory to save results
        
    Returns:
    --------
    results: DataFrame
        DataFrame containing evaluation results
    """
    # First check if we have ground truth
    if ground_truth_attr not in G.nodes[0]:
        print(f"Ground truth attribute '{ground_truth_attr}' not found in graph.")
        return None
    
    # Load available methods from modules
    # In a real implementation, would import and run these methods
    # For now, we'll just simulate the evaluation
    
    # Create a placeholder for results
    all_results = {}
    
    # Example: Evaluate traditional methods
    try:
        from traditional_methods import (run_louvain, run_leiden, run_infomap,
                                       run_label_propagation, run_spectral_clustering)
        
        # Run Louvain
        print("Running Louvain...")
        communities, execution_time = run_louvain(G)
        # Evaluate
        metrics = evaluate_communities(G, communities, ground_truth_attr)
        all_results['Louvain'] = {
            'communities': communities,
            'execution_time': execution_time,
            'metrics': metrics,
            'num_communities': len(set(communities.values()))
        }
        
        # Save result
        save_result(all_results['Louvain'], 'Louvain', results_dir)
        
        # Similar for other methods...
        
    except ImportError:
        print("Traditional methods module not available.")
    
    # Example: Evaluate GNN methods
    try:
        from gnn_community_detection import run_gnn_community_detection
        
        # Run GCN
        print("Running GCN...")
        gcn_results = run_gnn_community_detection(G, model_type='gcn', 
                                                ground_truth_attr=ground_truth_attr)
        all_results['GCN'] = {
            'communities': gcn_results['communities'],
            'training_time': gcn_results['training_time'],
            'metrics': gcn_results['metrics'],
            'num_communities': len(set(gcn_results['communities']))
        }
        
        # Save result
        save_result(all_results['GCN'], 'GCN', results_dir)
        
        # Similar for other GNN methods...
        
    except ImportError:
        print("GNN community detection module not available.")
    
    # Compile results
    compiled_results = compile_results(all_results)
    
    return compiled_results


# Function to evaluate communities
def evaluate_communities(G, communities, ground_truth_attr='community'):
    """
    Evaluate detected communities against ground truth
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph with community information
    communities: dict
        Dictionary mapping node to detected community
    ground_truth_attr: str
        Node attribute for ground truth community
        
    Returns:
    --------
    metrics: dict
        Evaluation metrics
    """
    # Get ground truth
    ground_truth = nx.get_node_attributes(G, ground_truth_attr)
    
    # Ensure both dictionaries have the same keys
    nodes = sorted(G.nodes())
    true_labels = [ground_truth.get(n, -1) for n in nodes]
    pred_labels = [communities.get(n, -1) for n in nodes]
    
    # Calculate metrics
    metrics = {}
    metrics['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
    metrics['ari'] = adjusted_rand_score(true_labels, pred_labels)
    
    # Calculate modularity if available
    try:
        if CDLIB_AVAILABLE:
            # Convert to cdlib format
            communities_list = []
            for comm_id in set(communities.values()):
                comm = [n for n, c in communities.items() if c == comm_id]
                communities_list.append(comm)
            
            communities_obj = evaluation.NodeClustering(communities_list, G)
            metrics['modularity'] = evaluation.newman_girvan_modularity(communities_obj).score
        else:
            try:
                import community as community_louvain
                metrics['modularity'] = community_louvain.modularity(communities, G)
            except ImportError:
                metrics['modularity'] = None
    except:
        metrics['modularity'] = None
    
    return metrics


# Function to generate a bar chart comparing top methods
def plot_top_methods_comparison(df, metric='nmi', n_methods=5, figsize=(12, 6)):
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
    if metric not in df.columns:
        print(f"Metric {metric} not found in results.")
        return
    
    # Select top methods
    top_methods = df.sort_values(metric, ascending=False).head(n_methods)
    
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
def generate_summary_table(df, metrics=['nmi', 'ari', 'modularity']):
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
    # Check which metrics are available
    available_metrics = [m for m in metrics if m in df.columns]
    if len(available_metrics) == 0:
        print("No specified metrics found in results.")
        return None
    
    # Create summary for each category
    categories = df['Category'].unique()
    rows = []
    
    for category in categories:
        category_df = df[df['Category'] == category]
        
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
    
    summary = pd.DataFrame(rows)
    
    # Create pivot table
    pivot = summary.pivot_table(
        index='Category', 
        columns='Metric', 
        values=['Best Method', 'Score'],
        aggfunc='first'
    )
    
    return pivot


# Function to visualize community stability across different methods
def visualize_community_stability(all_results, reference_method, n_methods=5, figsize=(12, 8)):
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
                # Convert communities to lists if they're dictionaries
                if isinstance(reference_communities, dict) and isinstance(result['communities'], dict):
                    nodes = sorted(reference_communities.keys())
                    ref_labels = [reference_communities[n] for n in nodes]
                    method_labels = [result['communities'][n] for n in nodes]
                    
                    ari = adjusted_rand_score(ref_labels, method_labels)
                    stability_scores[method_name] = ari
            except Exception as e:
                print(f"Error calculating stability for {method_name}: {e}")
    
    # Sort by stability score and select top N
    sorted_methods = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
    top_methods = sorted_methods[:n_methods]
    
    # Visualize
    plt.figure(figsize=figsize)
    methods, scores = zip(*top_methods)
    sns.barplot(x=list(methods), y=list(scores))
    plt.title(f'Community Stability Compared to {reference_method}')
    plt.xlabel('Method')
    plt.ylabel('Adjusted Rand Index')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Function to generate evaluation report
def generate_evaluation_report(df, output_dir='report'):
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Generate summary statistics
    summary = df.groupby('Category').agg({
        'Method': 'count',
        'nmi': ['mean', 'std', 'min', 'max'],
        'ari': ['mean', 'std', 'min', 'max'],
        'Execution Time (s)': ['mean', 'min', 'max']
    }).reset_index()
    
    summary.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    
    # 2. Generate top performers for each metric
    metrics = ['nmi', 'ari', 'modularity']
    available_metrics = [m for m in metrics if m in df.columns]
    
    for metric in available_metrics:
        top = df.sort_values(metric, ascending=False).head(5)
        top.to_csv(os.path.join(output_dir, f'top_{metric}.csv'))
    
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
    df['avg_score'] = df[available_metrics].mean(axis=1)
    best_overall = df.sort_values('avg_score', ascending=False).head(5)
    best_overall.to_csv(os.path.join(output_dir, 'best_overall.csv'))
    
    print(f"Evaluation report generated in {output_dir}")


# Main execution
if __name__ == "__main__":
    print("Comprehensive Evaluation of Community Detection Methods")
    print("="*60)
    
    # 1. Load results from previous notebooks
    print("\n1. Loading results from previous notebooks...")
    results_dir = 'results'
    all_results = load_results(results_dir)
    
    if not all_results:
        print("No results found. Running synthetic evaluation...")
        
        # Generate a synthetic graph with communities
        print("\n1.1 Generating a synthetic graph with communities...")
        n_communities = 5
        G, ground_truth = generate_synthetic_graph('sbm', n_nodes=100, 
                                                 n_communities=n_communities,
                                                 p_in=0.3, p_out=0.05)
        
        # Run evaluation on synthetic graph
        print("\n1.2 Running evaluation on synthetic graph...")
        compiled_results = evaluate_on_dataset(G, ground_truth_attr='community', 
                                             results_dir=results_dir)
    else:
        # Compile results
        print("\n2. Compiling results from all methods...")
        compiled_results = compile_results(all_results)
    
    if compiled_results is not None and not compiled_results.empty:
        # 3. Generate visualizations
        print("\n3. Generating visualizations...")
        
        # 3.1 Heatmap of performance metrics
        print("\n3.1 Generating heatmap of performance metrics...")
        metrics = ['nmi', 'ari', 'modularity']
        available_metrics = [m for m in metrics if m in compiled_results.columns]
        plot_heatmap(compiled_results, metrics=available_metrics)
        
        # 3.2 Performance vs. execution time
        print("\n3.2 Generating performance vs. execution time plot...")
        if 'nmi' in compiled_results.columns:
            plot_performance_vs_time(compiled_results, metric='nmi')
        
        # 3.3 Performance by category
        print("\n3.3 Generating performance by category plot...")
        plot_performance_by_category(compiled_results, metrics=available_metrics)
        
        # 3.4 Radar chart of top methods
        print("\n3.4 Generating radar chart of top methods...")
        if PLOTLY_AVAILABLE:
            plot_radar_chart(compiled_results, metrics=available_metrics, n_methods=5)
        
        # 3.5 Top methods comparison
        print("\n3.5 Generating top methods comparison...")
        if 'nmi' in compiled_results.columns:
            plot_top_methods_comparison(compiled_results, metric='nmi', n_methods=5)
        
        # 4. Generate summary table
        print("\n4. Generating summary table...")
        summary_table = generate_summary_table(compiled_results, metrics=available_metrics)
        if summary_table is not None:
            print("\nSummary Table:")
            print(summary_table)
        
        # 5. Community stability visualization
        print("\n5. Generating community stability visualization...")
        if len(all_results) > 1:
            # Choose first method as reference
            reference_method = list(all_results.keys())[0]
            visualize_community_stability(all_results, reference_method)
        
        # 6. Generate evaluation report
        print("\n6. Generating evaluation report...")
        generate_evaluation_report(compiled_results, output_dir='report')
    
    print("\nEvaluation complete!")
