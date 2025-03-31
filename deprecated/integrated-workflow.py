# Integrated Workflow for Community Detection Comparison
# =====================================================

import os
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

# For PyTorch and PyTorch Geometric
try:
    import torch
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
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. GNN methods will be skipped.")

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

# Import utility functions from other notebooks
# In a real scenario, these would be imported from modules
# For this demonstration, assume these imports are available
from data_prep import (load_data, create_graph_from_edgelist, create_graph_from_adjacency,
                      generate_synthetic_graph, compute_graph_statistics, plot_graph,
                      convert_nx_to_pytorch_geometric)

from traditional_methods import (run_louvain, run_leiden, run_infomap,
                               run_label_propagation, run_spectral_clustering,
                               evaluate_against_ground_truth, compare_methods as compare_traditional)

from gnn_community_detection import (run_gnn_community_detection, compare_gnn_models,
                                   GCN, GraphSAGE, GAT, extract_embeddings,
                                   detect_communities_from_embeddings, evaluate_communities)

from dynamic_gnn_notebook import (generate_dynamic_graphs, visualize_dynamic_communities,
                                run_dynamic_community_detection, compare_dynamic_gnn_models)

from overlapping_community_detection import (generate_synthetic_overlapping_graph,
                                           plot_overlapping_communities, run_bigclam,
                                           run_demon, run_slpa, run_gnn_overlapping,
                                           evaluate_overlapping_communities)

from comprehensive_evaluation import (compile_results, plot_heatmap, plot_performance_vs_time,
                                    plot_performance_by_category, plot_top_methods_comparison,
                                    generate_summary_table, generate_evaluation_report)

from visualization_notebook import (visualize_communities, visualize_community_evolution,
                                  community_membership_heatmap, alluvial_diagram,
                                  visualize_communities_3d, interactive_3d_visualization,
                                  vehlow_visualization)


# Function to save results
def save_results(results, method_name, results_dir='results'):
    """
    Save results for a specific method
    
    Parameters:
    -----------
    results: dict
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
        pickle.dump(results, f)
    print(f"Saved results for {method_name} to {filepath}")


# Function to load results
def load_results(method_name, results_dir='results'):
    """
    Load results for a specific method
    
    Parameters:
    -----------
    method_name: str
        Name of the method
    results_dir: str
        Directory containing saved results
        
    Returns:
    --------
    results: dict
        Results dictionary
    """
    filepath = os.path.join(results_dir, f"{method_name}.pkl")
    if not os.path.exists(filepath):
        print(f"No saved results for {method_name}")
        return None
    
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    return results


# Function to run all traditional methods
def run_all_traditional_methods(G, ground_truth_attr='community', results_dir='results'):
    """
    Run all traditional community detection methods
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to analyze
    ground_truth_attr: str
        Node attribute for ground truth communities
    results_dir: str
        Directory to save results
        
    Returns:
    --------
    all_results: dict
        Dictionary containing results from all methods
    """
    print("\n=== Running Traditional Community Detection Methods ===")
    
    all_results = {}
    
    # Get number of clusters from ground truth if available
    if ground_truth_attr in G.nodes[0]:
        ground_truth = [G.nodes[i][ground_truth_attr] for i in range(len(G))]
        n_clusters = len(set(ground_truth))
    else:
        n_clusters = None
    
    # Run Louvain
    print("Running Louvain method...")
    try:
        start_time = time.time()
        communities, _ = run_louvain(G)
        execution_time = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_against_ground_truth(G, communities, ground_truth_attr)
        
        # Store results
        all_results['Louvain'] = {
            'communities': communities,
            'execution_time': execution_time,
            'metrics': metrics,
            'num_communities': len(set(communities.values()))
        }
        
        # Save results
        save_results(all_results['Louvain'], 'Louvain', results_dir)
        
        # Visualize
        print("Visualizing Louvain communities...")
        plot_graph(G, community_attr='detected_community', 
                 title="Louvain Community Detection")
    except Exception as e:
        print(f"Error running Louvain: {e}")
    
    # Run Leiden
    if CDLIB_AVAILABLE:
        print("Running Leiden method...")
        try:
            start_time = time.time()
            communities, _ = run_leiden(G)
            execution_time = time.time() - start_time
            
            # Evaluate
            metrics = evaluate_against_ground_truth(G, communities, ground_truth_attr)
            
            # Store results
            all_results['Leiden'] = {
                'communities': communities,
                'execution_time': execution_time,
                'metrics': metrics,
                'num_communities': len(set(communities.values()))
            }
            
            # Save results
            save_results(all_results['Leiden'], 'Leiden', results_dir)
            
            # Visualize
            print("Visualizing Leiden communities...")
            plot_graph(G, community_attr='detected_community', 
                     title="Leiden Community Detection")
        except Exception as e:
            print(f"Error running Leiden: {e}")
    
    # Run Infomap
    if CDLIB_AVAILABLE:
        print("Running Infomap method...")
        try:
            start_time = time.time()
            communities, _ = run_infomap(G)
            execution_time = time.time() - start_time
            
            # Evaluate
            metrics = evaluate_against_ground_truth(G, communities, ground_truth_attr)
            
            # Store results
            all_results['Infomap'] = {
                'communities': communities,
                'execution_time': execution_time,
                'metrics': metrics,
                'num_communities': len(set(communities.values()))
            }
            
            # Save results
            save_results(all_results['Infomap'], 'Infomap', results_dir)
            
            # Visualize
            print("Visualizing Infomap communities...")
            plot_graph(G, community_attr='detected_community', 
                     title="Infomap Community Detection")
        except Exception as e:
            print(f"Error running Infomap: {e}")
    
    # Run Label Propagation
    print("Running Label Propagation method...")
    try:
        start_time = time.time()
        communities, _ = run_label_propagation(G)
        execution_time = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_against_ground_truth(G, communities, ground_truth_attr)
        
        # Store results
        all_results['LabelPropagation'] = {
            'communities': communities,
            'execution_time': execution_time,
            'metrics': metrics,
            'num_communities': len(set(communities.values()))
        }
        
        # Save results
        save_results(all_results['LabelPropagation'], 'LabelPropagation', results_dir)
        
        # Visualize
        print("Visualizing Label Propagation communities...")
        plot_graph(G, community_attr='detected_community', 
                 title="Label Propagation Community Detection")
    except Exception as e:
        print(f"Error running Label Propagation: {e}")
    
    # Run Spectral Clustering
    if n_clusters is not None:
        print(f"Running Spectral Clustering with {n_clusters} clusters...")
        try:
            start_time = time.time()
            communities, _ = run_spectral_clustering(G, n_clusters)
            execution_time = time.time() - start_time
            
            # Evaluate
            metrics = evaluate_against_ground_truth(G, communities, ground_truth_attr)
            
            # Store results
            all_results['SpectralClustering'] = {
                'communities': communities,
                'execution_time': execution_time,
                'metrics': metrics,
                'num_communities': len(set(communities.values()))
            }
            
            # Save results
            save_results(all_results['SpectralClustering'], 'SpectralClustering', results_dir)
            
            # Visualize
            print("Visualizing Spectral Clustering communities...")
            plot_graph(G, community_attr='detected_community', 
                     title="Spectral Clustering Community Detection")
        except Exception as e:
            print(f"Error running Spectral Clustering: {e}")
    
    return all_results


# Function to run all GNN-based methods
def run_all_gnn_methods(G, ground_truth_attr='community', results_dir='results',
                       embedding_dim=16, epochs=100):
    """
    Run all GNN-based community detection methods
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to analyze
    ground_truth_attr: str
        Node attribute for ground truth communities
    results_dir: str
        Directory to save results
    embedding_dim: int
        Dimension of node embeddings
    epochs: int
        Number of training epochs
        
    Returns:
    --------
    all_results: dict
        Dictionary containing results from all methods
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("PyTorch Geometric not available. Skipping GNN methods.")
        return {}
    
    print("\n=== Running GNN-Based Community Detection Methods ===")
    
    all_results = {}
    
    # Get number of clusters from ground truth if available
    if ground_truth_attr in G.nodes[0]:
        ground_truth = [G.nodes[i][ground_truth_attr] for i in range(len(G))]
        n_clusters = len(set(ground_truth))
    else:
        n_clusters = None
    
    # Run GCN
    print("Running GCN...")
    try:
        results = run_gnn_community_detection(G, model_type='gcn', 
                                            embedding_dim=embedding_dim,
                                            n_clusters=n_clusters, 
                                            epochs=epochs,
                                            ground_truth_attr=ground_truth_attr)
        
        all_results['GCN'] = {
            'communities': results['communities'],
            'embeddings': results['embeddings'],
            'training_time': results['training_time'],
            'metrics': results['metrics'] if 'metrics' in results else {},
            'num_communities': len(set(results['communities']))
        }
        
        # Save results
        save_results(all_results['GCN'], 'GCN', results_dir)
    except Exception as e:
        print(f"Error running GCN: {e}")
    
    # Run GraphSAGE
    print("Running GraphSAGE...")
    try:
        results = run_gnn_community_detection(G, model_type='graphsage', 
                                            embedding_dim=embedding_dim,
                                            n_clusters=n_clusters, 
                                            epochs=epochs,
                                            ground_truth_attr=ground_truth_attr)
        
        all_results['GraphSAGE'] = {
            'communities': results['communities'],
            'embeddings': results['embeddings'],
            'training_time': results['training_time'],
            'metrics': results['metrics'] if 'metrics' in results else {},
            'num_communities': len(set(results['communities']))
        }
        
        # Save results
        save_results(all_results['GraphSAGE'], 'GraphSAGE', results_dir)
    except Exception as e:
        print(f"Error running GraphSAGE: {e}")
    
    # Run GAT
    print("Running GAT...")
    try:
        results = run_gnn_community_detection(G, model_type='gat', 
                                            embedding_dim=embedding_dim,
                                            n_clusters=n_clusters, 
                                            epochs=epochs,
                                            ground_truth_attr=ground_truth_attr)
        
        all_results['GAT'] = {
            'communities': results['communities'],
            'embeddings': results['embeddings'],
            'training_time': results['training_time'],
            'metrics': results['metrics'] if 'metrics' in results else {},
            'num_communities': len(set(results['communities']))
        }
        
        # Save results
        save_results(all_results['GAT'], 'GAT', results_dir)
    except Exception as e:
        print(f"Error running GAT: {e}")
    
    # Run VGAE
    print("Running VGAE...")
    try:
        results = run_gnn_community_detection(G, model_type='vgae', 
                                            embedding_dim=embedding_dim,
                                            n_clusters=n_clusters, 
                                            epochs=epochs,
                                            ground_truth_attr=ground_truth_attr)
        
        all_results['VGAE'] = {
            'communities': results['communities'],
            'embeddings': results['embeddings'],
            'training_time': results['training_time'],
            'metrics': results['metrics'] if 'metrics' in results else {},
            'num_communities': len(set(results['communities']))
        }
        
        # Save results
        save_results(all_results['VGAE'], 'VGAE', results_dir)
    except Exception as e:
        print(f"Error running VGAE: {e}")
    
    return all_results


# Function to run all dynamic methods on a sequence of graphs
def run_all_dynamic_methods(graphs, ground_truth_attr='community', results_dir='results',
                          embedding_dim=16, epochs=50):
    """
    Run all dynamic community detection methods
    
    Parameters:
    -----------
    graphs: list
        List of NetworkX graphs for each time step
    ground_truth_attr: str
        Node attribute for ground truth communities
    results_dir: str
        Directory to save results
    embedding_dim: int
        Dimension of node embeddings
    epochs: int
        Number of training epochs
        
    Returns:
    --------
    all_results: dict
        Dictionary containing results from all methods
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("PyTorch Geometric not available. Skipping dynamic GNN methods.")
        return {}
    
    print("\n=== Running Dynamic Community Detection Methods ===")
    
    all_results = {}
    
    # Visualize ground truth communities
    print("Visualizing ground truth communities over time...")
    visualize_dynamic_communities(graphs, 
                               [nx.get_node_attributes(G, ground_truth_attr) for G in graphs],
                               layout='fixed',
                               title="Ground Truth Communities Over Time")
    
    # Get number of clusters from ground truth if available
    if ground_truth_attr in graphs[0].nodes[0]:
        ground_truth = [graphs[0].nodes[i][ground_truth_attr] for i in range(len(graphs[0]))]
        n_clusters = len(set(ground_truth))
    else:
        n_clusters = None
    
    # Run EvolveGCN
    print("Running EvolveGCN...")
    try:
        results = run_dynamic_community_detection(graphs, model_type='evolvegcn', 
                                               embedding_dim=embedding_dim,
                                               n_clusters=n_clusters, 
                                               epochs=epochs)
        
        all_results['EvolveGCN'] = {
            'communities_list': results['communities_list'],
            'embeddings_list': results['embeddings_list'],
            'training_time': results['training_time'],
            'metrics_list': results['metrics_list'] if 'metrics_list' in results else {}
        }
        
        # Save results
        save_results(all_results['EvolveGCN'], 'EvolveGCN', results_dir)
    except Exception as e:
        print(f"Error running EvolveGCN: {e}")
    
    # Run DySAT
    print("Running DySAT...")
    try:
        results = run_dynamic_community_detection(graphs, model_type='dysat', 
                                               embedding_dim=embedding_dim,
                                               n_clusters=n_clusters, 
                                               epochs=epochs)
        
        all_results['DySAT'] = {
            'communities_list': results['communities_list'],
            'embeddings_list': results['embeddings_list'],
            'training_time': results['training_time'],
            'metrics_list': results['metrics_list'] if 'metrics_list' in results else {}
        }
        
        # Save results
        save_results(all_results['DySAT'], 'DySAT', results_dir)
    except Exception as e:
        print(f"Error running DySAT: {e}")
    
    return all_results


# Function to run all overlapping community detection methods
def run_all_overlapping_methods(G, ground_truth, results_dir='results'):
    """
    Run all overlapping community detection methods
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to analyze
    ground_truth: list of lists
        Ground truth overlapping communities
    results_dir: str
        Directory to save results
        
    Returns:
    --------
    all_results: dict
        Dictionary containing results from all methods
    """
    if not CDLIB_AVAILABLE:
        print("cdlib not available. Some overlapping methods will be skipped.")
    
    print("\n=== Running Overlapping Community Detection Methods ===")
    
    all_results = {}
    
    # Visualize ground truth communities
    print("Visualizing ground truth overlapping communities...")
    plot_overlapping_communities(G, ground_truth)
    
    # Run BigCLAM
    if CDLIB_AVAILABLE:
        print("Running BigCLAM...")
        try:
            communities, execution_time = run_bigclam(G, k=len(ground_truth))
            
            # Evaluate
            metrics = evaluate_overlapping_communities(communities, ground_truth)
            
            # Store results
            all_results['BigCLAM'] = {
                'communities': communities,
                'execution_time': execution_time,
                'metrics': metrics,
                'num_communities': len(communities)
            }
            
            # Save results
            save_results(all_results['BigCLAM'], 'BigCLAM', results_dir)
            
            # Visualize
            print("Visualizing BigCLAM communities...")
            plot_overlapping_communities(G, communities)
        except Exception as e:
            print(f"Error running BigCLAM: {e}")
    
    # Run DEMON
    if CDLIB_AVAILABLE:
        print("Running DEMON...")
        try:
            communities, execution_time = run_demon(G)
            
            # Evaluate
            metrics = evaluate_overlapping_communities(communities, ground_truth)
            
            # Store results
            all_results['DEMON'] = {
                'communities': communities,
                'execution_time': execution_time,
                'metrics': metrics,
                'num_communities': len(communities)
            }
            
            # Save results
            save_results(all_results['DEMON'], 'DEMON', results_dir)
            
            # Visualize
            print("Visualizing DEMON communities...")
            plot_overlapping_communities(G, communities)
        except Exception as e:
            print(f"Error running DEMON: {e}")
    
    # Run SLPA
    if CDLIB_AVAILABLE:
        print("Running SLPA...")
        try:
            communities, execution_time = run_slpa(G)
            
            # Evaluate
            metrics = evaluate_overlapping_communities(communities, ground_truth)
            
            # Store results
            all_results['SLPA'] = {
                'communities': communities,
                'execution_time': execution_time,
                'metrics': metrics,
                'num_communities': len(communities)
            }
            
            # Save results
            save_results(all_results['SLPA'], 'SLPA', results_dir)
            
            # Visualize
            print("Visualizing SLPA communities...")
            plot_overlapping_communities(G, communities)
        except Exception as e:
            print(f"Error running SLPA: {e}")
    
    # Run GNN-Overlapping
    if TORCH_GEOMETRIC_AVAILABLE:
        print("Running GNN-based overlapping community detection...")
        try:
            communities, execution_time, _ = run_gnn_overlapping(G, ground_truth, epochs=50)
            
            # Evaluate
            metrics = evaluate_overlapping_communities(communities, ground_truth)
            
            # Store results
            all_results['GNN-Overlapping'] = {
                'communities': communities,
                'execution_time': execution_time,
                'metrics': metrics,
                'num_communities': len(communities)
            }
            
            # Save results
            save_results(all_results['GNN-Overlapping'], 'GNN_Overlapping', results_dir)
            
            # Visualize
            print("Visualizing GNN-based overlapping communities...")
            plot_overlapping_communities(G, communities)
        except Exception as e:
            print(f"Error running GNN-based overlapping community detection: {e}")
    
    return all_results


# Function to run the complete community detection comparison workflow
def run_complete_workflow(n_nodes=100, n_communities=5, results_dir='results',
                        run_traditional=True, run_gnn=True, run_dynamic=True, run_overlapping=True):
    """
    Run the complete community detection comparison workflow
    
    Parameters:
    -----------
    n_nodes: int
        Number of nodes in the synthetic graph
    n_communities: int
        Number of communities in the synthetic graph
    results_dir: str
        Directory to save results
    run_traditional: bool
        Whether to run traditional methods
    run_gnn: bool
        Whether to run GNN methods
    run_dynamic: bool
        Whether to run dynamic methods
    run_overlapping: bool
        Whether to run overlapping methods
        
    Returns:
    --------
    compiled_results: DataFrame
        DataFrame containing compiled results
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Dictionary to store all results
    all_results = {}
    
    # 1. Generate a synthetic non-overlapping graph
    print("\n1. Generating a synthetic SBM graph with non-overlapping communities...")
    G, _ = generate_synthetic_graph('sbm', n_nodes=n_nodes, n_communities=n_communities,
                                  p_in=0.3, p_out=0.05)
    
    # Calculate graph statistics
    print("\n2. Computing graph statistics...")
    stats = compute_graph_statistics(G)
    
    # Print statistics
    for key, value in stats.items():
        if isinstance(value, list) or isinstance(value, np.ndarray):
            print(f"{key}: [list of {len(value)} values]")
        else:
            print(f"{key}: {value}")
    
    # Visualize the graph with ground truth communities
    print("\n3. Visualizing ground truth communities...")
    plot_graph(G, community_attr='community', title="Ground Truth Communities")
    
    # 4. Run traditional community detection methods
    if run_traditional:
        traditional_results = run_all_traditional_methods(G, ground_truth_attr='community', 
                                                       results_dir=results_dir)
        all_results.update(traditional_results)
    
    # 5. Run GNN-based community detection methods
    if run_gnn and TORCH_GEOMETRIC_AVAILABLE:
        gnn_results = run_all_gnn_methods(G, ground_truth_attr='community', 
                                        results_dir=results_dir)
        all_results.update(gnn_results)
    
    # 6. Generate a sequence of dynamic graphs for dynamic methods
    if run_dynamic:
        print("\n6. Generating a sequence of dynamic graphs...")
        n_time_steps = 5
        dynamic_graphs = generate_dynamic_graphs(n_time_steps=n_time_steps, n_nodes=n_nodes, 
                                              n_communities=n_communities, change_fraction=0.1)
        
        # Run dynamic community detection methods
        if TORCH_GEOMETRIC_AVAILABLE:
            dynamic_results = run_all_dynamic_methods(dynamic_graphs, ground_truth_attr='community', 
                                                   results_dir=results_dir)
            
            # Store just the average metrics for comparison
            for method, results in dynamic_results.items():
                if 'metrics_list' in results:
                    avg_metrics = {}
                    for metric in results['metrics_list'][0].keys():
                        avg_metrics[metric] = np.mean([m[metric] for m in results['metrics_list']])
                    
                    all_results[f"{method}_avg"] = {
                        'communities': results['communities_list'][-1],  # Last time step
                        'execution_time': results['training_time'],
                        'metrics': avg_metrics,
                        'num_communities': len(set(results['communities_list'][-1]))
                    }
    
    # 7. Generate a synthetic graph with overlapping communities
    if run_overlapping:
        print("\n7. Generating a synthetic graph with overlapping communities...")
        G_overlap, ground_truth_overlap = generate_synthetic_overlapping_graph(
            n_nodes=n_nodes, n_communities=n_communities, overlap_size=20)
        
        # Run overlapping community detection methods
        overlapping_results = run_all_overlapping_methods(G_overlap, ground_truth_overlap, 
                                                       results_dir=results_dir)
        
        # Store just the metrics for comparison
        for method, results in overlapping_results.items():
            all_results[f"{method}_overlap"] = {
                'communities': results['communities'],
                'execution_time': results['execution_time'],
                'metrics': results['metrics'],
                'num_communities': results['num_communities']
            }
    
    # 8. Compile results
    print("\n8. Compiling results...")
    compiled_results = []
    
    for method, result in all_results.items():
        # Skip if result doesn't have 'metrics' key
        if 'metrics' not in result:
            continue
        
        # Determine method category
        if any(t in method.lower() for t in ['louvain', 'leiden', 'infomap', 'label', 'spectral']):
            category = 'traditional'
        elif any(g in method.lower() for g in ['gcn', 'graphsage', 'gat', 'vgae']):
            category = 'gnn'
        elif any(d in method.lower() for d in ['evolve', 'dysat']):
            category = 'dynamic'
        elif any(o in method.lower() for o in ['bigclam', 'demon', 'slpa', 'overlap']):
            category = 'overlapping'
        else:
            category = 'other'
        
        # Create row for method
        row = {
            'Method': method,
            'Category': category,
            'Training Time (s)': result.get('training_time', 0),
            'Execution Time (s)': result.get('execution_time', 0),
            'Num Communities': result.get('num_communities', 0)
        }
        
        # Add metrics
        metrics = result['metrics']
        for metric_name, value in metrics.items():
            row[metric_name] = value
        
        compiled_results.append(row)
    
    compiled_df = pd.DataFrame(compiled_results)
    
    # 9. Generate visualizations and evaluation report
    print("\n9. Generating visualizations and evaluation report...")
    
    # Available metrics
    metrics = ['nmi', 'ari', 'f1', 'omega', 'modularity']
    available_metrics = [m for m in metrics if m in compiled_df.columns]
    
    if len(available_metrics) > 0:
        # Plot heatmap
        plot_heatmap(compiled_df, metrics=available_metrics)
        
        # Plot performance vs. execution time
        if 'nmi' in available_metrics:
            plot_performance_vs_time(compiled_df, metric='nmi')
        
        # Plot performance by category
        plot_performance_by_category(compiled_df, metrics=available_metrics)
        
        # Plot top methods
        if 'nmi' in available_metrics:
            plot_top_methods_comparison(compiled_df, metric='nmi', n_methods=5)
        
        # Generate summary table
        summary_table = generate_summary_table(compiled_df, metrics=available_metrics)
        print("\nSummary Table:")
        print(summary_table)
        
        # Generate evaluation report
        generate_evaluation_report(compiled_df, output_dir='report')
    
    return compiled_df


# Function to analyze the results and provide recommendations
def analyze_results(compiled_df):
    """
    Analyze the results and provide recommendations
    
    Parameters:
    -----------
    compiled_df: DataFrame
        DataFrame containing compiled results
        
    Returns:
    --------
    recommendations: dict
        Dictionary containing recommendations
    """
    recommendations = {}
    
    # Check if DataFrame is empty
    if compiled_df.empty:
        print("No results to analyze.")
        return recommendations
    
    # Check which metrics are available
    metrics = ['nmi', 'ari', 'f1', 'omega', 'modularity']
    available_metrics = [m for m in metrics if m in compiled_df.columns]
    
    if len(available_metrics) == 0:
        print("No metrics found for analysis.")
        return recommendations
    
    # Identify best method overall
    if 'nmi' in available_metrics:
        best_overall = compiled_df.loc[compiled_df['nmi'].idxmax()]
        recommendations['best_overall'] = {
            'method': best_overall['Method'],
            'nmi': best_overall['nmi'],
            'category': best_overall['Category']
        }
    
    # Identify best method in each category
    for category in compiled_df['Category'].unique():
        category_df = compiled_df[compiled_df['Category'] == category]
        
        if 'nmi' in available_metrics and not category_df.empty:
            best_in_category = category_df.loc[category_df['nmi'].idxmax()]
            recommendations[f'best_{category}'] = {
                'method': best_in_category['Method'],
                'nmi': best_in_category['nmi']
            }
    
    # Identify fastest method with good performance
    if 'nmi' in available_metrics and 'Execution Time (s)' in compiled_df.columns:
        # Get methods with above-average NMI
        good_methods = compiled_df[compiled_df['nmi'] > compiled_df['nmi'].mean()]
        
        if not good_methods.empty:
            fastest_good = good_methods.loc[good_methods['Execution Time (s)'].idxmin()]
            recommendations['fastest_good'] = {
                'method': fastest_good['Method'],
                'nmi': fastest_good['nmi'],
                'execution_time': fastest_good['Execution Time (s)']
            }
    
    # Compare traditional vs. GNN methods
    if 'nmi' in available_metrics:
        trad_methods = compiled_df[compiled_df['Category'] == 'traditional']
        gnn_methods = compiled_df[compiled_df['Category'] == 'gnn']
        
        if not trad_methods.empty and not gnn_methods.empty:
            trad_avg_nmi = trad_methods['nmi'].mean()
            gnn_avg_nmi = gnn_methods['nmi'].mean()
            
            if trad_avg_nmi > gnn_avg_nmi:
                recommendations['trad_vs_gnn'] = {
                    'better_category': 'traditional',
                    'avg_nmi_diff': trad_avg_nmi - gnn_avg_nmi
                }
            else:
                recommendations['trad_vs_gnn'] = {
                    'better_category': 'gnn',
                    'avg_nmi_diff': gnn_avg_nmi - trad_avg_nmi
                }
    
    # Identify methods that perform well on overlapping communities
    overlap_methods = compiled_df[compiled_df['Category'] == 'overlapping']
    if 'f1' in available_metrics and not overlap_methods.empty:
        best_overlap = overlap_methods.loc[overlap_methods['f1'].idxmax()]
        recommendations['best_overlapping'] = {
            'method': best_overlap['Method'],
            'f1': best_overlap['f1']
        }
    
    return recommendations


# Function to generate a final report
def generate_final_report(compiled_df, recommendations, output_file='final_report.txt'):
    """
    Generate a final report with results and recommendations
    
    Parameters:
    -----------
    compiled_df: DataFrame
        DataFrame containing compiled results
    recommendations: dict
        Dictionary containing recommendations
    output_file: str
        Output file path
        
    Returns:
    --------
    None
    """
    with open(output_file, 'w') as f:
        f.write("==================================================\n")
        f.write("       COMMUNITY DETECTION COMPARISON REPORT      \n")
        f.write("==================================================\n\n")
        
        f.write("1. SUMMARY OF METHODS\n")
        f.write("-------------------\n\n")
        
        f.write(f"Total methods evaluated: {len(compiled_df)}\n")
        f.write(f"Categories: {', '.join(compiled_df['Category'].unique())}\n\n")
        
        f.write("Method counts by category:\n")
        for category, count in compiled_df['Category'].value_counts().items():
            f.write(f"  - {category}: {count} methods\n")
        f.write("\n")
        
        f.write("2. PERFORMANCE COMPARISON\n")
        f.write("-------------------------\n\n")
        
        # Check which metrics are available
        metrics = ['nmi', 'ari', 'f1', 'omega', 'modularity']
        available_metrics = [m for m in metrics if m in compiled_df.columns]
        
        if available_metrics:
            f.write("Top 5 methods by NMI:\n")
            if 'nmi' in available_metrics:
                top_nmi = compiled_df.sort_values('nmi', ascending=False).head(5)
                for i, (_, row) in enumerate(top_nmi.iterrows()):
                    f.write(f"  {i+1}. {row['Method']} ({row['Category']}): NMI = {row['nmi']:.4f}\n")
            f.write("\n")
            
            f.write("Average performance by category:\n")
            for category in compiled_df['Category'].unique():
                category_df = compiled_df[compiled_df['Category'] == category]
                if 'nmi' in available_metrics:
                    f.write(f"  - {category}: NMI = {category_df['nmi'].mean():.4f}\n")
            f.write("\n")
            
            f.write("3. RECOMMENDATIONS\n")
            f.write("------------------\n\n")
            
            if 'best_overall' in recommendations:
                best = recommendations['best_overall']
                f.write(f"Best overall method: {best['method']} ({best['category']}) with NMI = {best['nmi']:.4f}\n\n")
            
            f.write("Best method by category:\n")
            for category in compiled_df['Category'].unique():
                key = f'best_{category}'
                if key in recommendations:
                    best = recommendations[key]
                    f.write(f"  - {category}: {best['method']} with NMI = {best['nmi']:.4f}\n")
            f.write("\n")
            
            if 'fastest_good' in recommendations:
                fastest = recommendations['fastest_good']
                f.write(f"Fastest good method: {fastest['method']} with execution time = {fastest['execution_time']:.2f}s and NMI = {fastest['nmi']:.4f}\n\n")
            
            if 'trad_vs_gnn' in recommendations:
                trad_vs_gnn = recommendations['trad_vs_gnn']
                f.write(f"Traditional vs. GNN: {trad_vs_gnn['better_category']} methods perform better by {trad_vs_gnn['avg_nmi_diff']:.4f} NMI on average\n\n")
            
            if 'best_overlapping' in recommendations:
                best_overlap = recommendations['best_overlapping']
                f.write(f"Best for overlapping communities: {best_overlap['method']} with F1 = {best_overlap['f1']:.4f}\n\n")
            
            f.write("4. CONCLUSION\n")
            f.write("-------------\n\n")
            
            # Generate a conclusion based on the recommendations
            if 'best_overall' in recommendations:
                best = recommendations['best_overall']
                f.write(f"The {best['method']} method ({best['category']}) shows the best overall performance for community detection in the tested graphs. ")
            
            if 'trad_vs_gnn' in recommendations:
                trad_vs_gnn = recommendations['trad_vs_gnn']
                if trad_vs_gnn['better_category'] == 'traditional':
                    f.write(f"Traditional methods outperform GNN-based methods on average, suggesting that simpler approaches may be sufficient for this type of graph structure. ")
                else:
                    f.write(f"GNN-based methods outperform traditional methods on average, suggesting that learning node representations provides advantages for community detection in this type of graph. ")
            
            if 'fastest_good' in recommendations:
                fastest = recommendations['fastest_good']
                f.write(f"For applications where speed is important, {fastest['method']} provides a good balance between performance and efficiency. ")
            
            if 'best_overlapping' in recommendations:
                best_overlap = recommendations['best_overlapping']
                f.write(f"When overlapping communities need to be detected, {best_overlap['method']} is the recommended approach. ")
            
            f.write("\n\nFurther analysis could include testing on larger, real-world networks and exploring parameter sensitivity of the top-performing methods.")
        else:
            f.write("No performance metrics available for comparison.\n")
    
    print(f"Final report generated: {output_file}")


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("Integrated Workflow for Community Detection Comparison")
    print("=" * 80)
    
    # Run the complete workflow
    compiled_results = run_complete_workflow(n_nodes=100, n_communities=5, 
                                           results_dir='results')
    
    # Analyze results
    recommendations = analyze_results(compiled_results)
    
    # Generate final report
    generate_final_report(compiled_results, recommendations, output_file='final_report.txt')
    
    print("\nWorkflow complete!")
