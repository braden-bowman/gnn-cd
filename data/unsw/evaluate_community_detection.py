#!/usr/bin/env python3
# Evaluate community detection methods on UNSW-NB15 dataset with GPU acceleration

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import rustworkx as rx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import pickle
from pathlib import Path
from process_unsw import load_graph, PROCESSED_DIR

# Import community detection methods
from community_detection.traditional_methods import (
    run_louvain, run_leiden, run_infomap, run_label_propagation, run_spectral_clustering
)
from community_detection.gnn_community_detection import (
    run_gcn, run_graphsage, run_gat, run_vgae
)
from community_detection.overlapping_community_detection import (
    run_bigclam, run_demon, run_slpa, run_gnn_overlapping
)

# Define paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.path.join(PROCESSED_DIR, "unsw_graph.pt")
RESULTS_DIR = os.path.join(PROCESSED_DIR, "results")
Path(RESULTS_DIR).mkdir(exist_ok=True)
GPU_DATA_DIR = os.path.join(PROCESSED_DIR, "gpu_data")
GPU_DATA_PATH = os.path.join(GPU_DATA_DIR, "gpu_ready_data.pt")

# Check if GPU is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def evaluate_communities(G, communities, community_type='non-overlapping'):
    """
    Evaluate community detection results for cybersecurity
    
    Parameters:
    -----------
    G: rustworkx.PyGraph
        Graph with ground truth labels
    communities: dict or list
        Detected communities (dict for non-overlapping, list of lists for overlapping)
    community_type: str
        'non-overlapping' or 'overlapping'
        
    Returns:
    --------
    metrics: dict
        Evaluation metrics
    """
    if community_type == 'non-overlapping':
        # Convert community dict to community assignments
        community_assignments = {}
        for node, community in communities.items():
            if community not in community_assignments:
                community_assignments[community] = []
            community_assignments[community].append(node)
        
        communities_list = list(community_assignments.values())
    else:
        # Already in list of lists format
        communities_list = communities
    
    # Calculate homogeneity of communities regarding attack labels
    y_true = np.array([G.get_node_data(i)['label'] for i in range(len(G))])
    
    # Assign predicted label to each node based on majority class in its community
    y_pred = np.zeros_like(y_true)
    
    for comm_idx, community in enumerate(communities_list):
        # Get labels of nodes in this community
        comm_labels = [G.get_node_data(node)['label'] for node in community]
        
        # Determine majority class
        if len(comm_labels) > 0:
            majority_label = 1 if sum(comm_labels) / len(comm_labels) >= 0.5 else 0
            
            # Assign majority label to all nodes in this community
            for node in community:
                y_pred[node] = majority_label
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate AUC if possible
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    except:
        metrics['auc'] = 0.5  # Default for random classifier
    
    # Calculate cluster purity
    purities = []
    for community in communities_list:
        if len(community) > 0:
            comm_labels = [G.get_node_data(node)['label'] for node in community]
            majority_count = max(sum(comm_labels), len(comm_labels) - sum(comm_labels))
            purity = majority_count / len(comm_labels)
            purities.append(purity)
    
    metrics['avg_purity'] = np.mean(purities) if purities else 0
    metrics['min_purity'] = np.min(purities) if purities else 0
    metrics['max_purity'] = np.max(purities) if purities else 0
    
    # Number of communities and sizes
    metrics['num_communities'] = len(communities_list)
    
    community_sizes = [len(comm) for comm in communities_list]
    metrics['avg_community_size'] = np.mean(community_sizes) if community_sizes else 0
    metrics['min_community_size'] = np.min(community_sizes) if community_sizes else 0
    metrics['max_community_size'] = np.max(community_sizes) if community_sizes else 0
    
    # Calculate attack concentration
    attack_concentration = {}
    for comm_idx, community in enumerate(communities_list):
        attack_count = sum(G.get_node_data(node)['label'] for node in community)
        attack_concentration[comm_idx] = attack_count / len(community) if community else 0
    
    # Identify communities with high attack concentration
    high_attack_comms = [idx for idx, conc in attack_concentration.items() if conc >= 0.8]
    metrics['num_attack_communities'] = len(high_attack_comms)
    metrics['attack_communities_ratio'] = len(high_attack_comms) / len(communities_list) if communities_list else 0
    
    return metrics

def load_or_run_method(method_name, method_func, G, method_args=None, cache_dir=None):
    """
    Load cached results for a method or run it if no cache exists
    
    Parameters:
    -----------
    method_name: str
        Name of the method
    method_func: callable
        Function to run the method
    G: rustworkx.PyGraph
        Graph to apply the method to
    method_args: dict
        Arguments to pass to the method function
    cache_dir: str
        Directory to store cache files
        
    Returns:
    --------
    result_dict: dict
        Dictionary with method results
    """
    if cache_dir is None:
        cache_dir = RESULTS_DIR
        
    cache_file = os.path.join(cache_dir, f"{method_name}_results.pkl")
    
    # Check if cached results exist
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                result_dict = pickle.load(f)
            print(f"Loaded cached results for {method_name}")
            # Report metrics from cache
            metrics = result_dict['metrics']
            print(f"{method_name}: {metrics['num_communities']} communities, "
                  f"F1={metrics['f1']:.4f}, Time={result_dict['execution_time']:.2f}s (cached)")
            return result_dict
        except Exception as e:
            print(f"Error loading cache for {method_name}: {e}")
            # Continue to run the method
    
    # Run the method
    print(f"Running {method_name}...")
    start_time = time.time()
    
    # Apply method with arguments if provided
    if method_args:
        try:
            if method_name in ['gcn', 'graphsage', 'gat', 'vgae']:
                # GNN methods have different return values
                communities, _, _ = method_func(G, **method_args)
            elif method_name in ['bigclam', 'demon', 'slpa']:
                # Overlapping methods return exec time
                communities, exec_time = method_func(G, **method_args)
                # Use the reported execution time
                execution_time = exec_time
            elif method_name == 'gnn_overlapping':
                # GNN overlapping has a different signature
                communities, exec_time, _ = method_func(G, **method_args)
                execution_time = exec_time
            else:
                # Traditional methods
                communities, _ = method_func(G, **method_args)
            
            # Use measured time if not returned by the method
            if 'exec_time' not in locals():
                execution_time = time.time() - start_time
                
            # Evaluate communities
            if method_name in ['bigclam', 'demon', 'slpa', 'gnn_overlapping']:
                metrics = evaluate_communities(G, communities, community_type='overlapping')
            else:
                metrics = evaluate_communities(G, communities)
                
            # Create result dictionary
            result_dict = {
                'communities': communities,
                'execution_time': execution_time,
                'metrics': metrics
            }
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(result_dict, f)
                
            print(f"{method_name}: {metrics['num_communities']} communities, "
                  f"F1={metrics['f1']:.4f}, Time={execution_time:.2f}s")
                  
            return result_dict
        except Exception as e:
            print(f"Error running {method_name}: {e}")
            return None
    else:
        print(f"Missing arguments for {method_name}")
        return None

def evaluate_all_methods(G):
    """
    Apply and evaluate all community detection methods with caching and GPU acceleration
    
    Parameters:
    -----------
    G: rustworkx.PyGraph
        Graph with features and labels
        
    Returns:
    --------
    results: dict
        Dictionary of results for all methods
    """
    # Create dictionary to store all results
    results = {}
    
    # Check if GPU-ready data is available
    gpu_data = None
    if os.path.exists(GPU_DATA_PATH):
        try:
            gpu_data = torch.load(GPU_DATA_PATH, map_location=DEVICE)
            print(f"Loaded GPU-ready data from {GPU_DATA_PATH}")
        except Exception as e:
            print(f"Error loading GPU data: {e}")
    
    # Extract node features and labels if GPU data not available
    if gpu_data is None:
        print("Extracting features and labels from graph...")
        features = []
        labels = []
        for i in range(len(G)):
            node_data = G.get_node_data(i)
            if 'features' in node_data:
                features.append(node_data['features'])
            else:
                features.append(np.zeros(10))
            labels.append(node_data.get('label', 0))
        
        X = torch.tensor(np.array(features), dtype=torch.float32).to(DEVICE)
        y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    else:
        # Use GPU-ready data
        X = gpu_data['x'].to(DEVICE)
        y = gpu_data['y'].to(DEVICE)
    
    # Define all methods we want to evaluate
    methods = {
        # Traditional methods
        'louvain': {
            'func': run_louvain,
            'args': {}
        },
        'leiden': {
            'func': run_leiden,
            'args': {}
        },
        'infomap': {
            'func': run_infomap,
            'args': {}
        },
        'label_propagation': {
            'func': run_label_propagation,
            'args': {}
        },
        'spectral_clustering': {
            'func': run_spectral_clustering,
            'args': {'k': 10}
        },
        
        # GNN methods
        'gcn': {
            'func': run_gcn,
            'args': {
                'X': X,
                'y': y,
                'hidden_dim': 64,
                'epochs': 50,
                'device': DEVICE
            }
        },
        'graphsage': {
            'func': run_graphsage,
            'args': {
                'X': X,
                'y': y,
                'hidden_dim': 64,
                'epochs': 50,
                'device': DEVICE
            }
        },
        'gat': {
            'func': run_gat,
            'args': {
                'X': X,
                'y': y,
                'hidden_dim': 64,
                'epochs': 50,
                'device': DEVICE
            }
        },
        'vgae': {
            'func': run_vgae,
            'args': {
                'X': X,
                'hidden_dim': 64,
                'epochs': 50,
                'device': DEVICE
            }
        },
        
        # Overlapping community detection methods
        'bigclam': {
            'func': run_bigclam,
            'args': {'k': 10}
        },
        'demon': {
            'func': run_demon,
            'args': {}
        },
        'slpa': {
            'func': run_slpa,
            'args': {}
        }
    }
    
    # Group methods for easier printing
    method_groups = {
        'Traditional Methods': ['louvain', 'leiden', 'infomap', 'label_propagation', 'spectral_clustering'],
        'GNN Methods': ['gcn', 'graphsage', 'gat', 'vgae'],
        'Overlapping Methods': ['bigclam', 'demon', 'slpa']
    }
    
    # Run all methods with caching
    for group_name, group_methods in method_groups.items():
        print(f"\nEvaluating {group_name}...")
        
        for method_name in group_methods:
            if method_name in methods:
                method_info = methods[method_name]
                result = load_or_run_method(
                    method_name, 
                    method_info['func'], 
                    G, 
                    method_info['args']
                )
                if result:
                    results[method_name] = result
    
    # For GNN overlapping, we need ground truth communities
    # Here we'll use the results from Louvain as "ground truth" for demonstration
    if 'louvain' in results:
        try:
            print("\nRunning GNN Overlapping...")
            louvain_comms = results['louvain']['communities']
            # Convert dict to list of lists
            gt_communities = []
            for comm_id in set(louvain_comms.values()):
                comm = [n for n, c in louvain_comms.items() if c == comm_id]
                gt_communities.append(comm)
            
            # Run GNN overlapping with GT communities
            gnn_ovl_result = load_or_run_method(
                'gnn_overlapping',
                run_gnn_overlapping,
                G,
                {
                    'gt_communities': gt_communities,
                    'hidden_dim': 64,
                    'epochs': 50,
                    'device': DEVICE
                }
            )
            
            if gnn_ovl_result:
                results['gnn_overlapping'] = gnn_ovl_result
        except Exception as e:
            print(f"Error running GNN Overlapping: {e}")
    
    return results

def visualize_results(results):
    """
    Visualize comparison results
    
    Parameters:
    -----------
    results: dict
        Dictionary of results for all methods
    """
    # Create DataFrame for comparison
    rows = []
    for method_name, result in results.items():
        metrics = result['metrics']
        row = {
            'Method': method_name,
            'Num Communities': metrics['num_communities'],
            'Avg Community Size': metrics['avg_community_size'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'AUC': metrics['auc'],
            'Avg Purity': metrics['avg_purity'],
            'Execution Time (s)': result['execution_time']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save results to CSV
    df.to_csv(os.path.join(RESULTS_DIR, 'community_detection_comparison.csv'), index=False)
    
    # Create visualizations
    plt.figure(figsize=(14, 10))
    
    # Plot metrics
    plt.subplot(2, 2, 1)
    metrics_df = df.melt(id_vars=['Method'], 
                        value_vars=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Avg Purity'],
                        var_name='Metric', value_name='Value')
    sns.barplot(x='Method', y='Value', hue='Metric', data=metrics_df)
    plt.title('Quality Metrics by Method')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot execution time
    plt.subplot(2, 2, 2)
    sns.barplot(x='Method', y='Execution Time (s)', data=df, palette='viridis')
    plt.title('Execution Time by Method')
    plt.xticks(rotation=45)
    plt.yscale('log')  # Log scale for better visualization
    
    # Plot number of communities
    plt.subplot(2, 2, 3)
    sns.barplot(x='Method', y='Num Communities', data=df, palette='mako')
    plt.title('Number of Communities')
    plt.xticks(rotation=45)
    
    # Plot average community size
    plt.subplot(2, 2, 4)
    sns.barplot(x='Method', y='Avg Community Size', data=df, palette='flare')
    plt.title('Average Community Size')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'community_detection_comparison.png'))
    plt.show()
    
    # Create a heatmap of methods vs metrics
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Avg Purity']
    plt.figure(figsize=(12, 8))
    heatmap_df = df.set_index('Method')[metrics_cols]
    sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt='.3f', linewidths=0.5)
    plt.title('Performance Comparison of Community Detection Methods')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'community_detection_heatmap.png'))
    plt.show()
    
    # Plot F1 score vs execution time
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Execution Time (s)', y='F1', 
                   size='Num Communities', hue='Method', 
                   data=df, sizes=(50, 400), alpha=0.8)
    plt.xscale('log')
    plt.title('F1 Score vs Execution Time')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'f1_vs_time.png'))
    plt.show()

def analyze_attack_communities(G, methods_results, threshold=0.7, output_dir=None):
    """
    Analyze communities with high concentration of attack traffic
    
    Parameters:
    -----------
    G: rustworkx.PyGraph
        Graph with node labels
    methods_results: dict
        Dictionary with results for different methods
    threshold: float
        Threshold for attack concentration
    output_dir: str
        Directory to save visualizations
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
        
    print("\n=== Analysis of Attack-Related Communities ===")
    
    # For each method, identify communities with high attack concentration
    attack_communities = {}
    
    for method_name, result in methods_results.items():
        metrics = result['metrics']
        communities = result['communities']
        
        # Skip methods with no attack communities
        if metrics.get('num_attack_communities', 0) == 0:
            continue
            
        # Analyze communities by attack concentration
        if method_name in ['bigclam', 'demon', 'slpa', 'gnn_overlapping']:
            # Overlapping communities are already in list format
            comm_list = communities
            
            # Calculate attack concentration for each community
            attack_conc = {}
            for i, comm in enumerate(comm_list):
                if len(comm) > 0:
                    attack_count = sum(G.get_node_data(node)['label'] for node in comm)
                    conc = attack_count / len(comm)
                    attack_conc[i] = conc
        else:
            # Non-overlapping communities need conversion to lists
            comm_dict = {}
            for node, comm_id in communities.items():
                if comm_id not in comm_dict:
                    comm_dict[comm_id] = []
                comm_dict[comm_id].append(node)
                
            comm_list = list(comm_dict.values())
            
            # Calculate attack concentration for each community
            attack_conc = {}
            for i, (comm_id, nodes) in enumerate(comm_dict.items()):
                if len(nodes) > 0:
                    attack_count = sum(G.get_node_data(node)['label'] for node in nodes)
                    conc = attack_count / len(nodes)
                    attack_conc[comm_id] = conc
        
        # Find communities above threshold
        high_attack_comms = {
            comm_id: (conc, len(comm_list[comm_id]) if isinstance(comm_id, int) and comm_id < len(comm_list) else len(comm_dict[comm_id]))
            for comm_id, conc in attack_conc.items() 
            if conc >= threshold
        }
        
        if high_attack_comms:
            attack_communities[method_name] = high_attack_comms
            print(f"\n{method_name} identified {len(high_attack_comms)} attack-related communities:")
            
            for comm_id, (conc, size) in sorted(high_attack_comms.items(), key=lambda x: x[1][0], reverse=True):
                print(f"  Community {comm_id}: {conc:.2f} attack concentration, {size} nodes")
    
    # Compare methods by attack detection ability
    if attack_communities:
        print("\nMethod comparison for attack detection:")
        
        # Create DataFrame for visualization
        comparison_data = []
        for method, comms in attack_communities.items():
            # Get metrics
            metrics = methods_results[method]['metrics']
            
            # Total nodes in attack communities
            attack_nodes = sum(size for _, (_, size) in comms.items())
            
            comparison_data.append({
                'Method': method,
                'Attack Communities': len(comms),
                'Attack Nodes': attack_nodes,
                'Avg Concentration': np.mean([conc for conc, _ in comms.values()]),
                'Max Concentration': max([conc for conc, _ in comms.values()]),
                'F1 Score': metrics['f1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            })
        
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        comparison_df.to_csv(os.path.join(output_dir, 'attack_detection_comparison.csv'), index=False)
        
        # Plot comparison
        plt.figure(figsize=(14, 8))
        
        # Create bars for number of attack communities
        ax1 = plt.subplot(2, 2, 1)
        sns.barplot(x='Method', y='Attack Communities', data=comparison_df, palette='viridis', ax=ax1)
        plt.title('Number of Attack Communities')
        plt.xticks(rotation=45)
        
        # Plot average concentration
        ax2 = plt.subplot(2, 2, 2)
        sns.barplot(x='Method', y='Avg Concentration', data=comparison_df, palette='flare', ax=ax2)
        plt.title('Average Attack Concentration')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Plot F1 score
        ax3 = plt.subplot(2, 2, 3)
        sns.barplot(x='Method', y='F1 Score', data=comparison_df, palette='mako', ax=ax3)
        plt.title('F1 Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Scatter plot of attack communities vs F1
        ax4 = plt.subplot(2, 2, 4)
        sns.scatterplot(
            x='Attack Communities', 
            y='F1 Score', 
            size='Attack Nodes',
            hue='Method',
            data=comparison_df, 
            ax=ax4,
            sizes=(50, 500)
        )
        plt.title('Attack Communities vs. F1 Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attack_detection_comparison.png'))
        
        return comparison_df
    else:
        print("No methods identified attack communities above the threshold.")
        return None

def main():
    """Main function to evaluate community detection methods"""
    # Check if graph exists
    if not os.path.exists(GRAPH_PATH):
        print(f"Graph file not found: {GRAPH_PATH}")
        print("Please run process_unsw.py first to create the graph.")
        return
    
    # Load graph with GPU compatibility if available
    print(f"Loading graph from {GRAPH_PATH}...")
    G = load_graph(GRAPH_PATH, device=DEVICE)
    
    # Check for existing results
    results_file = os.path.join(RESULTS_DIR, 'community_detection_results.pkl')
    if os.path.exists(results_file):
        try:
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            print(f"Loaded existing results for {len(results)} methods")
            
            # Ask if user wants to rerun analysis
            rerun = input("Do you want to rerun the analysis? (y/n): ").lower() == 'y'
            if not rerun:
                # Just visualize existing results
                visualize_results(results)
                # Analyze attack communities
                analyze_attack_communities(G, results)
                return
        except Exception as e:
            print(f"Error loading existing results: {e}")
            # Continue to run evaluation
    
    # Evaluate all methods
    results = evaluate_all_methods(G)
    
    # Save results
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {results_file}")
    
    # Visualize results
    visualize_results(results)
    
    # Analyze attack communities
    analyze_attack_communities(G, results)
    
    print("\nEvaluation complete. Results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main()