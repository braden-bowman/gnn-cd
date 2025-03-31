# Community Detection Framework Package

# Import functions from data_prep.py
from .data_prep import (
    load_data, create_graph_from_edgelist, create_graph_from_adjacency,
    convert_rustworkx_to_pytorch_geometric, plot_graph, generate_synthetic_graph,
    compute_graph_statistics, display_graph_statistics
)

# Import functions from traditional_methods.py
from .traditional_methods import (
    run_louvain, run_leiden, run_infomap, run_label_propagation,
    run_spectral_clustering, run_walktrap, run_girvan_newman,
    evaluate_against_ground_truth, plot_communities, add_communities_to_graph,
    compare_methods, save_communities
)

# Import functions from gnn_community_detection.py
from .gnn_community_detection import (
    rwx_to_pyg, GCN, GraphSAGE, GAT, VGAE,
    train_gnn_embedding, extract_embeddings, detect_communities_from_embeddings,
    evaluate_communities, plot_embeddings, add_communities_to_graph,
    run_gnn_community_detection, compare_gnn_models
)

# Import functions from dynamic_gnn.py
from .dynamic_gnn import (
    generate_dynamic_graphs, visualize_dynamic_communities,
    EvolveGCN, DySAT, train_dynamic_gnn, extract_temporal_embeddings,
    detect_temporal_communities, evaluate_temporal_communities,
    visualize_community_evolution, run_dynamic_community_detection,
    compare_dynamic_gnn_models
)

# Import functions from overlapping_community_detection.py
from .overlapping_community_detection import (
    generate_synthetic_overlapping_graph, plot_overlapping_communities,
    run_bigclam, run_demon, run_slpa, GNN_Overlapping, rwx_to_pyg_overlapping,
    train_gnn_overlapping, predict_gnn_overlapping, run_gnn_overlapping,
    evaluate_overlapping_communities, compare_overlapping_methods
)

# Import functions from evaluation.py
from .evaluation import (
    load_results, save_result, compile_results, plot_heatmap, plot_radar_chart,
    plot_performance_vs_time, plot_performance_by_category, plot_top_methods_comparison,
    generate_summary_table, generate_evaluation_report, evaluate_communities,
    visualize_community_stability
)

# Import functions from visualization.py
from .visualization import (
    community_layout, layered_community_layout, combined_community_layout,
    visualize_communities, visualize_community_evolution, community_membership_heatmap,
    alluvial_diagram, visualize_communities_3d, interactive_3d_visualization,
    visualize_embeddings, visualize_metrics_over_time, visualize_overlapping_communities,
    vehlow_visualization
)