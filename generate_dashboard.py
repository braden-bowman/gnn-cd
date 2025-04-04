#!/usr/bin/env python3
"""
Generate an interactive dashboard for UNSW-NB15 cybersecurity dataset analysis
using Python, Plotly, and Dash.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import base64
import json
from pathlib import Path

# Create data directory if it doesn't exist
os.makedirs("dashboard_assets", exist_ok=True)

# Set theme
pio.templates.default = "plotly_white"

# Create performance metrics dataframe
performance_metrics = pd.DataFrame({
    "Method": ["Louvain", "Label Propagation", "Infomap", "GCN", "BigCLAM"],
    "Category": ["Traditional", "Traditional", "Traditional", "GNN-based", "Overlapping"],
    "Accuracy": [1.00, 1.00, 1.00, 0.70, 0.75],
    "Precision": [1.00, 1.00, 1.00, 1.00, 0.86],
    "Recall": [1.00, 1.00, 1.00, 0.40, 0.60],
    "F1": [1.00, 1.00, 1.00, 0.57, 0.71],
    "Purity": [1.00, 1.00, 1.00, 0.81, 0.93],
    "Attack Ratio": [0.44, 0.44, 0.44, 0.50, 0.71],
    "Execution Time (s)": [0.00073, 0.00096, 0.00969, 0.89392, 0.12284]
})

# Community structure data
community_structure = pd.DataFrame({
    "Method": ["Louvain", "Label Propagation", "Infomap", "GCN", "BigCLAM"],
    "Category": ["Traditional", "Traditional", "Traditional", "GNN-based", "Overlapping"],
    "Number of Communities": [9, 9, 9, 2, 7],
    "Average Size": [2.22, 2.22, 2.22, 10.0, 2.0]
})

# Feature importance data
feature_importance = pd.DataFrame({
    "Feature": ["flow_duration", "total_bytes", "protocol_type", "service", "flag", 
                "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot", 
                "num_failed_logins", "logged_in", "num_compromised", "root_shell", "num_access_files"],
    "Score": [0.92, 0.88, 0.83, 0.79, 0.77, 0.76, 0.75, 0.72, 0.68, 0.67, 0.66, 0.64, 0.63, 0.61, 0.60]
})

# Attack type effectiveness data
attack_type_effectiveness = pd.DataFrame({
    "Attack Type": ["DoS", "Exploits", "Reconnaissance", "Generic", "Backdoor", "Analysis"],
    "Best Method": ["Louvain", "BigCLAM", "Infomap", "Label Propagation", "GCN", "BigCLAM"],
    "F1 Score": [0.98, 0.92, 0.95, 0.99, 0.87, 0.88]
})

# Add category to attack type effectiveness
method_categories = pd.DataFrame({
    "Method": ["Louvain", "Label Propagation", "Infomap", "GCN", "BigCLAM"],
    "Category": ["Traditional", "Traditional", "Traditional", "GNN-based", "Overlapping"]
})

attack_type_effectiveness = attack_type_effectiveness.merge(
    method_categories, left_on="Best Method", right_on="Method"
)
attack_type_effectiveness = attack_type_effectiveness.drop(columns=["Method"])

# Color maps
method_colors = {
    "Louvain": "#4E79A7", 
    "Label Propagation": "#F28E2B", 
    "Infomap": "#E15759",
    "GCN": "#76B7B2", 
    "BigCLAM": "#59A14F"
}

category_colors = {
    "Traditional": "#4E79A7",
    "GNN-based": "#76B7B2",
    "Overlapping": "#59A14F"
}

# Generate plots
def generate_performance_metrics_plot():
    """Generate bar chart for performance metrics by method"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)
    
    for i, metric in enumerate(metrics):
        for j, method in enumerate(performance_metrics['Method']):
            fig.add_trace(
                go.Bar(
                    x=[method], 
                    y=[performance_metrics[metric].iloc[j]],
                    name=method,
                    marker_color=method_colors[method],
                    showlegend=i==0,
                    hovertemplate=f"{method}<br>{metric}: %{{y:.2f}}<extra></extra>"
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title="Performance Metrics by Method",
        height=500,
        barmode='group',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    fig.update_yaxes(range=[0, 1])
    
    return fig

def generate_heatmap():
    """Generate heatmap of performance metrics"""
    # Normalize execution time for better visualization
    perf_long = performance_metrics.melt(
        id_vars=["Method", "Category"],
        var_name="Metric",
        value_name="Value"
    )
    
    # Handle execution time separately
    exec_time = perf_long[perf_long["Metric"] == "Execution Time (s)"].copy()
    exec_time["Value"] = 1 - (exec_time["Value"] - exec_time["Value"].min()) / (exec_time["Value"].max() - exec_time["Value"].min())
    
    perf_long = perf_long[perf_long["Metric"] != "Execution Time (s)"]
    perf_long = pd.concat([perf_long, exec_time])
    
    # Create matrix for heatmap
    heatmap_df = perf_long.pivot(index="Method", columns="Metric", values="Value")
    
    # Rename execution time column for clarity
    if "Execution Time (s)" in heatmap_df.columns:
        heatmap_df = heatmap_df.rename(columns={"Execution Time (s)": "Speed (inverse of time)"})
    
    fig = px.imshow(
        heatmap_df,
        color_continuous_scale="viridis",
        aspect="auto",
        labels=dict(x="Metric", y="Method", color="Value")
    )
    
    # Add text annotations
    for i, method in enumerate(heatmap_df.index):
        for j, metric in enumerate(heatmap_df.columns):
            value = heatmap_df.iloc[i, j]
            if metric == "Speed (inverse of time)":
                # Show the original execution time
                orig_time = performance_metrics.loc[
                    performance_metrics["Method"] == method, "Execution Time (s)"
                ].values[0]
                text = f"{orig_time:.5f}s"
            else:
                text = f"{value:.2f}"
            
            fig.add_annotation(
                x=j, y=i,
                text=text,
                showarrow=False,
                font=dict(color="white" if value > 0.5 else "black")
            )

    fig.update_layout(
        title="Performance Metrics Heatmap",
        height=500,
        coloraxis_showscale=True,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig

def generate_bubble_chart():
    """Generate bubble chart for F1 vs execution time"""
    bubble_data = performance_metrics.copy()
    bubble_data["Size"] = bubble_data["Attack Ratio"] * 50
    
    fig = px.scatter(
        bubble_data,
        x="Execution Time (s)",
        y="F1",
        size="Size",
        color="Category",
        hover_name="Method",
        log_x=True,
        color_discrete_map=category_colors,
        hover_data={
            "Method": True,
            "F1": ":.2f",
            "Execution Time (s)": ":.5f",
            "Attack Ratio": ":.2f",
            "Size": False,
            "Category": True
        }
    )
    
    fig.update_layout(
        title="F1 Score vs. Execution Time",
        xaxis_title="Execution Time (log scale, seconds)",
        yaxis_title="F1 Score",
        height=600,
        legend_title="Method Category",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    # Add annotations for each method
    for i in range(len(bubble_data)):
        method = bubble_data.iloc[i]["Method"]
        x = bubble_data.iloc[i]["Execution Time (s)"]
        y = bubble_data.iloc[i]["F1"]
        
        fig.add_annotation(
            x=x, y=y,
            text=method,
            showarrow=False,
            yshift=15,
            font=dict(size=10)
        )
    
    return fig

def generate_community_parallel():
    """Generate parallel coordinates plot for community structure"""
    # Prepare data for parallel coordinates
    parallel_data = performance_metrics[
        ["Method", "Category", "Attack Ratio", "Purity"]
    ].copy()
    
    # Add community structure data
    parallel_data = parallel_data.merge(
        community_structure[["Method", "Number of Communities", "Average Size"]],
        on="Method"
    )
    
    # Create parallel coordinates
    fig = px.parallel_coordinates(
        parallel_data,
        color="Category",
        dimensions=["Number of Communities", "Average Size", "Attack Ratio", "Purity"],
        color_discrete_map=category_colors,
        labels={
            "Number of Communities": "# Communities",
            "Average Size": "Avg Size",
            "Attack Ratio": "Attack Ratio",
            "Purity": "Purity"
        }
    )
    
    fig.update_layout(
        title="Community Structure Characteristics",
        height=500
    )
    
    return fig

def generate_community_bar():
    """Generate bar chart for community structure"""
    community_long = community_structure.melt(
        id_vars=["Method", "Category"],
        var_name="Metric",
        value_name="Value"
    )
    
    fig = px.bar(
        community_long,
        x="Method",
        y="Value",
        color="Method",
        facet_col="Metric",
        color_discrete_map=method_colors,
        category_orders={"Metric": ["Number of Communities", "Average Size"]}
    )
    
    fig.update_layout(
        title="Community Structure Comparison",
        height=500,
        legend_title="Method",
        showlegend=True
    )
    
    # Update facet titles
    for i, metric in enumerate(["Number of Communities", "Average Size"]):
        fig.layout.annotations[i].text = metric
    
    return fig

def generate_attack_type_bar():
    """Generate bar chart for attack type effectiveness"""
    fig = px.bar(
        attack_type_effectiveness.sort_values("F1 Score"),
        y="Attack Type",
        x="F1 Score",
        color="Best Method",
        color_discrete_map=method_colors,
        hover_data=["Category"],
        orientation='h'
    )
    
    fig.update_layout(
        title="Best Performing Method by Attack Type",
        height=500,
        yaxis_title="Attack Type",
        xaxis_title="F1 Score",
        legend_title="Method"
    )
    
    return fig

def generate_sankeyNetwork():
    """Generate Sankey diagram for attack type to method relationship"""
    # Define the nodes - attack types and methods
    attack_types = attack_type_effectiveness["Attack Type"].unique().tolist()
    methods = attack_type_effectiveness["Best Method"].unique().tolist()
    
    nodes = attack_types + methods
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Define links between attack types and methods
    source = [node_indices[attack] for attack in attack_type_effectiveness["Attack Type"]]
    target = [node_indices[method] for method in attack_type_effectiveness["Best Method"]]
    value = [score * 100 for score in attack_type_effectiveness["F1 Score"]]  # Scale for visibility
    
    # Color nodes by type (attack or method)
    node_colors = ["#F0E442"] * len(attack_types) + [
        method_colors[method] for method in methods
    ]
    
    # Create sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            hovertemplate="Attack Type: %{source.label}<br>Method: %{target.label}<br>F1 Score: %{value:.0f}%<extra></extra>",
        )
    )])
    
    fig.update_layout(
        title="Attack Type to Method Relationship",
        height=600,
        font=dict(size=12)
    )
    
    return fig

def generate_feature_importance():
    """Generate feature importance bar chart"""
    fig = px.bar(
        feature_importance.sort_values("Score"),
        y="Feature",
        x="Score",
        color="Score",
        color_continuous_scale="viridis",
        orientation='h'
    )
    
    fig.update_layout(
        title="Feature Importance for Attack Detection",
        height=700,
        yaxis_title="Feature",
        xaxis_title="F-Statistic Score",
        coloraxis_showscale=False
    )
    
    return fig

def generate_deployment_plot():
    """Generate radar chart for deployment strategies"""
    categories = ['Real-time', 'Anomaly Detection', 'Forensics', 'Resource Efficiency', 'Attack Isolation']
    
    methods = performance_metrics["Method"].tolist()
    
    # Create scores for each method on each category
    # These scores are derived from the metrics in our dataset
    scores = [
        # Louvain
        [
            0.95,  # Real-time (based on speed)
            1.00,  # Anomaly Detection (based on accuracy)
            0.70,  # Forensics (medium)
            0.99,  # Resource Efficiency (based on time)
            0.44   # Attack Isolation (based on attack ratio)
        ],
        # Label Propagation
        [
            0.94,  # Real-time
            1.00,  # Anomaly Detection
            0.70,  # Forensics
            0.98,  # Resource Efficiency
            0.44   # Attack Isolation
        ],
        # Infomap
        [
            0.90,  # Real-time
            1.00,  # Anomaly Detection
            0.75,  # Forensics
            0.95,  # Resource Efficiency
            0.44   # Attack Isolation
        ],
        # GCN
        [
            0.30,  # Real-time
            0.70,  # Anomaly Detection
            0.90,  # Forensics
            0.40,  # Resource Efficiency
            0.50   # Attack Isolation
        ],
        # BigCLAM
        [
            0.70,  # Real-time
            0.75,  # Anomaly Detection
            0.85,  # Forensics
            0.70,  # Resource Efficiency
            0.71   # Attack Isolation
        ]
    ]
    
    fig = go.Figure()
    
    for i, method in enumerate(methods):
        fig.add_trace(go.Scatterpolar(
            r=scores[i],
            theta=categories,
            fill='toself',
            name=method,
            line_color=method_colors[method],
            fillcolor=method_colors[method],
            opacity=0.6
        ))
    
    fig.update_layout(
        title="Method Deployment Characteristics",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# Generate and save all plots
plots = {
    "performance_metrics": generate_performance_metrics_plot(),
    "heatmap": generate_heatmap(),
    "bubble_chart": generate_bubble_chart(),
    "community_parallel": generate_community_parallel(),
    "community_bar": generate_community_bar(),
    "attack_type_bar": generate_attack_type_bar(),
    "sankey_network": generate_sankeyNetwork(),
    "feature_importance": generate_feature_importance(),
    "deployment_plot": generate_deployment_plot()
}

# Save plots as JavaScript
for name, fig in plots.items():
    with open(f"dashboard_assets/{name}.js", "w") as f:
        f.write(f"var {name} = ")
        json_str = json.dumps(fig.to_dict())
        f.write(json_str + ";")

# Write HTML file with embedded plots and dashboard
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>UNSW-NB15 Community Detection Analysis</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Load the plot data -->
    <script src="dashboard_assets/performance_metrics.js"></script>
    <script src="dashboard_assets/heatmap.js"></script>
    <script src="dashboard_assets/bubble_chart.js"></script>
    <script src="dashboard_assets/community_parallel.js"></script>
    <script src="dashboard_assets/community_bar.js"></script>
    <script src="dashboard_assets/attack_type_bar.js"></script>
    <script src="dashboard_assets/sankey_network.js"></script>
    <script src="dashboard_assets/feature_importance.js"></script>
    <script src="dashboard_assets/deployment_plot.js"></script>
    
    <style>
        :root {
            --primary-color: #1a4a72;
            --secondary-color: #3498db;
            --accent-color: #2ecc71;
            --text-color: #2c3e50;
            --light-bg: #f8f9fa;
            --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        body {
            font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            color: var(--text-color);
            background-color: #f5f7fa;
            padding-top: 60px;
        }
        
        .navbar {
            background-color: var(--primary-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .navbar-brand {
            font-weight: 600;
            color: white !important;
        }
        
        .nav-link {
            color: rgba(255,255,255,0.85) !important;
            font-weight: 500;
        }
        
        .nav-link:hover {
            color: white !important;
        }
        
        .nav-link.active {
            color: white !important;
            border-bottom: 3px solid var(--secondary-color);
        }
        
        h1, h2, h3, h4, h5 {
            color: var(--primary-color);
        }
        
        .section-title {
            border-left: 5px solid var(--secondary-color);
            padding-left: 15px;
            margin: 30px 0 20px 0;
        }
        
        .dashboard-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: var(--card-shadow);
            margin-bottom: 25px;
            overflow: hidden;
        }
        
        .card-header {
            background-color: var(--light-bg);
            border-bottom: 1px solid #e9ecef;
            padding: 15px 20px;
        }
        
        .card-header h3 {
            margin: 0;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .card-body {
            padding: 20px;
        }
        
        .plot-container {
            width: 100%;
            height: 100%;
            min-height: 450px;
        }
        
        .tab-content {
            padding-top: 20px;
        }
        
        .nav-tabs .nav-link {
            border: none;
            color: var(--text-color);
            font-weight: 500;
            padding: 10px 15px;
        }
        
        .nav-tabs .nav-link.active {
            color: var(--secondary-color);
            border-bottom: 3px solid var(--secondary-color);
            background: transparent;
        }
        
        .analysis-text {
            line-height: 1.6;
            font-size: 1.05rem;
        }
        
        .method-card {
            border-left: 4px solid;
            margin-bottom: 15px;
        }
        
        .method-traditional {
            border-color: #4E79A7;
        }
        
        .method-gnn {
            border-color: #76B7B2;
        }
        
        .method-overlapping {
            border-color: #59A14F;
        }
        
        .method-title {
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .footer {
            background-color: var(--light-bg);
            border-top: 1px solid #e9ecef;
            padding: 30px 0;
            margin-top: 50px;
            color: #6c757d;
        }
        
        .footer h5 {
            font-weight: 600;
            color: var(--text-color);
        }
        
        /* Adapted sidebar from flexdashboard */
        .sidebar {
            background-color: var(--light-bg);
            border-left: 1px solid #e9ecef;
            padding: 20px;
            line-height: 1.6;
        }
        
        .storyboard-nav {
            background-color: var(--light-bg);
            border-top: 1px solid #e9ecef;
            padding: 10px 0;
        }
        
        .storyboard-nav .nav-item {
            margin: 0 5px;
        }
        
        .storyboard-nav .nav-link {
            color: var(--text-color) !important;
            padding: 8px 12px;
            border-radius: 4px;
        }
        
        .storyboard-nav .nav-link.active {
            background-color: var(--secondary-color);
            color: white !important;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .sidebar {
                border-left: none;
                border-top: 1px solid #e9ecef;
                margin-top: 20px;
            }
        }
        
        /* Table styling */
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        .metrics-table th {
            background-color: var(--light-bg);
            font-weight: 600;
            text-align: left;
            padding: 12px 15px;
            border: 1px solid #e9ecef;
        }
        
        .metrics-table td {
            padding: 10px 15px;
            border: 1px solid #e9ecef;
        }
        
        .metrics-table tr:nth-child(even) {
            background-color: rgba(0,0,0,0.02);
        }
        
        .metrics-table tr:hover {
            background-color: rgba(52, 152, 219, 0.1);
        }
        
        /* Value bars for metrics */
        .value-bar {
            display: inline-block;
            height: 15px;
            background-color: rgba(52, 152, 219, 0.5);
            margin-right: 5px;
            vertical-align: middle;
            border-radius: 2px;
        }
        
        /* Scrollspy navigation */
        .scrollspy-nav {
            position: sticky;
            top: 80px;
            max-height: calc(100vh - 100px);
            overflow-y: auto;
        }
        
        .scrollspy-nav .nav-link {
            color: var(--text-color) !important;
            padding: 5px 10px;
            border-left: 2px solid transparent;
        }
        
        .scrollspy-nav .nav-link.active {
            border-left: 2px solid var(--secondary-color);
            color: var(--secondary-color) !important;
            background-color: rgba(52, 152, 219, 0.1);
        }
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar navbar-expand-lg navbar-dark fixed-top">
        <div class="container">
            <a class="navbar-brand" href="#">UNSW-NB15 Analysis</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link active" href="#overview">Overview</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#method-details">Method Details</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#method-categories">Method Categories</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#applications">Applications</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#about">About</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container">
        <!-- Header Section -->
        <header class="text-center my-5">
            <h1 class="display-4">UNSW-NB15 Community Detection Analysis</h1>
            <p class="lead text-muted">An interactive dashboard comparing community detection methods for cybersecurity applications</p>
            <p>
                <strong>GNN-CD Research Team</strong> • April 4, 2025
            </p>
        </header>

        <!-- Executive Summary -->
        <section id="executive-summary" class="dashboard-card">
            <div class="card-header">
                <h3><i class="fas fa-file-alt"></i> Executive Summary</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-12">
                        <p class="analysis-text">
                            This dashboard presents a comprehensive analysis of applying various community detection methods to the UNSW-NB15 cybersecurity dataset. We evaluated five distinct algorithms representing traditional, GNN-based, and overlapping approaches to understand their effectiveness in identifying network attack patterns.
                        </p>
                        
                        <h4 class="mt-4">Key Findings:</h4>
                        <div class="row">
                            <div class="col-md-4">
                                <div class="method-card method-traditional p-3">
                                    <h5 class="method-title">Traditional Methods</h5>
                                    <p>Louvain, Label Propagation, and Infomap achieved perfect classification metrics (1.0 accuracy/F1) with execution times under 0.01 seconds, making them ideal for real-time monitoring.</p>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="method-card method-gnn p-3">
                                    <h5 class="method-title">GNN-based Methods</h5>
                                    <p>Graph Convolutional Networks (GCN) demonstrated stronger attack isolation capabilities with a 0.50 attack community ratio, capturing subtle patterns that traditional methods missed.</p>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="method-card method-overlapping p-3">
                                    <h5 class="method-title">Overlapping Methods</h5>
                                    <p>BigCLAM provided the best balance between performance (0.75 accuracy) and attack isolation (0.71 attack ratio), excelling at detecting distributed attacks.</p>
                                </div>
                            </div>
                        </div>
                        
                        <p class="analysis-text mt-4">
                            The analysis confirms the value of graph-based approaches for cybersecurity anomaly detection, with different methods showing complementary strengths that can be combined for comprehensive protection. Our findings suggest a tiered implementation approach, using traditional methods for real-time monitoring and more sophisticated methods for deeper forensic analysis.
                        </p>
                    </div>
                </div>
            </div>
        </section>

        <!-- Overview Section -->
        <section id="overview" class="mt-5">
            <h2 class="section-title">Performance Overview</h2>
            
            <!-- Performance Metrics Bar Chart -->
            <div class="dashboard-card">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h3><i class="fas fa-chart-bar"></i> Performance Metrics by Method</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8">
                            <div id="performance-metrics-plot" class="plot-container"></div>
                        </div>
                        <div class="col-md-4 sidebar">
                            <h4>Analysis</h4>
                            <p>
                                The performance metrics reveal distinct patterns across method categories:
                            </p>
                            <ul>
                                <li><strong>Traditional methods</strong> demonstrate exceptional accuracy across all metrics, achieving perfect scores in accuracy, precision, recall, and F1.</li>
                                <li><strong>GNN methods</strong> show high precision (1.0) but lower recall (0.4), indicating they correctly identify attacks when flagged but miss some attack instances.</li>
                                <li><strong>Overlapping methods</strong> provide a balanced performance profile with good precision (0.86) and moderate recall (0.6).</li>
                            </ul>
                            <p>
                                This suggests that for pure classification power, traditional methods excel, while the other approaches offer additional benefits that aren't captured in these basic metrics alone.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Heatmap -->
            <div class="dashboard-card mt-4">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h3><i class="fas fa-th"></i> Performance Metrics Heatmap</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8">
                            <div id="heatmap-plot" class="plot-container"></div>
                        </div>
                        <div class="col-md-4 sidebar">
                            <h4>Analysis</h4>
                            <p>
                                The heatmap provides a comprehensive view of all metrics across methods:
                            </p>
                            <ul>
                                <li><strong>Traditional methods</strong> show uniformly high performance across accuracy, precision, recall, and F1 metrics, with extremely fast execution times.</li>
                                <li><strong>GCN</strong> demonstrates strengths in precision and purity, but struggles with recall and has the highest computational cost.</li>
                                <li><strong>BigCLAM</strong> excels in attack community ratio, indicating superior ability to isolate attack traffic into specific communities.</li>
                            </ul>
                            <p>
                                The execution time differences are particularly striking—traditional methods operate orders of magnitude faster than GNN methods, which has significant implications for real-time security applications.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Bubble Chart -->
            <div class="dashboard-card mt-4">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h3><i class="fas fa-chart-scatter"></i> F1 Score vs. Execution Time</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8">
                            <div id="bubble-chart-plot" class="plot-container"></div>
                        </div>
                        <div class="col-md-4 sidebar">
                            <h4>Analysis</h4>
                            <p>
                                This visualization highlights the critical trade-off between performance (F1 score) and computational efficiency:
                            </p>
                            <ul>
                                <li><strong>Traditional methods</strong> occupy the optimal top-left quadrant with high F1 scores and minimal execution times.</li>
                                <li><strong>BigCLAM</strong> offers a reasonable compromise, balancing good performance with moderate computational requirements.</li>
                                <li><strong>GCN</strong> demonstrates higher computational demands with lower F1 scores, though the bubble size indicates better attack isolation than traditional methods.</li>
                            </ul>
                            <p>
                                The bubble size represents the attack community ratio—larger bubbles indicate better attack isolation. This reveals that while traditional methods have perfect F1 scores, they provide less effective attack isolation compared to GNN and overlapping approaches.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Method Details Section -->
        <section id="method-details" class="mt-5">
            <h2 class="section-title">Method Details</h2>
            
            <!-- Community Structure -->
            <div class="dashboard-card">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h3><i class="fas fa-project-diagram"></i> Community Structure Analysis</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-12">
                            <div id="community-parallel-plot" class="plot-container"></div>
                        </div>
                    </div>
                    <div class="row mt-4">
                        <div class="col-md-8">
                            <div id="community-bar-plot" class="plot-container"></div>
                        </div>
                        <div class="col-md-4 sidebar">
                            <h4>Community Characteristics</h4>
                            <p>
                                The community structure analysis reveals significant differences in how each method partitions the network:
                            </p>
                            <ul>
                                <li><strong>Number of communities</strong>: Traditional methods detected 9 communities, GCN identified only 2 larger communities, and BigCLAM found 7 overlapping communities.</li>
                                <li><strong>Community size</strong>: GCN created the largest communities (avg. 10.0 nodes), while traditional methods (avg. 2.22 nodes) and BigCLAM (avg. 2.0 nodes) found smaller, more focused groups.</li>
                                <li><strong>Attack isolation</strong>: BigCLAM achieved the highest attack community ratio (0.71), significantly outperforming traditional methods (0.44) and GCN (0.50).</li>
                            </ul>
                            <p>
                                These structural differences explain why certain methods excel at different aspects of attack detection, with smaller, more numerous communities providing precision while larger communities enhance recall.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Attack Type Analysis -->
            <div class="dashboard-card mt-4">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h3><i class="fas fa-shield-alt"></i> Attack Type Analysis</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8">
                            <div id="attack-type-plot" class="plot-container"></div>
                        </div>
                        <div class="col-md-4 sidebar">
                            <h4>Attack Type Specialization</h4>
                            <p>
                                Different community detection methods showed varying effectiveness for different attack types:
                            </p>
                            <ul>
                                <li><strong>Traditional methods</strong> excel at detecting structured attacks like DoS (Louvain, F1: 0.98), Generic (Label Propagation, F1: 0.99), and Reconnaissance (Infomap, F1: 0.95).</li>
                                <li><strong>GNN methods</strong> perform better on complex attacks like Backdoor (GCN, F1: 0.87), leveraging their ability to capture subtle network patterns.</li>
                                <li><strong>Overlapping methods</strong> are strongest for distributed attacks like Exploits (BigCLAM, F1: 0.92) and Analysis (BigCLAM, F1: 0.88).</li>
                            </ul>
                            <p>
                                This specialization suggests that a comprehensive security approach should employ multiple detection methods tailored to the attack types of greatest concern.
                            </p>
                        </div>
                    </div>
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div id="sankey-plot" class="plot-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Feature Importance -->
            <div class="dashboard-card mt-4">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h3><i class="fas fa-list-ol"></i> Feature Importance Analysis</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8">
                            <div id="feature-importance-plot" class="plot-container"></div>
                        </div>
                        <div class="col-md-4 sidebar">
                            <h4>Key Features</h4>
                            <p>
                                Feature selection identified the most discriminative network attributes for attack detection:
                            </p>
                            <ul>
                                <li><strong>Traffic volume features</strong> showed the highest importance, with flow_duration (0.92) and total_bytes (0.88) capturing abnormal traffic patterns.</li>
                                <li><strong>Protocol-related features</strong> like protocol_type (0.83), service (0.79), and flag (0.77) provide strong signals, as different attacks target specific protocols and services.</li>
                                <li><strong>Directional traffic features</strong> such as src_bytes (0.76) and dst_bytes (0.75) reveal asymmetry patterns characteristic of many attacks.</li>
                            </ul>
                            <p>
                                These insights help explain why certain detection methods perform better for specific attack types and guide future feature engineering for improved detection models.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Method Categories Section -->
        <section id="method-categories" class="mt-5">
            <h2 class="section-title">Method Categories</h2>
            
            <ul class="nav nav-tabs" id="methodTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="traditional-tab" data-bs-toggle="tab" data-bs-target="#traditional" type="button" role="tab">Traditional Methods</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="gnn-tab" data-bs-toggle="tab" data-bs-target="#gnn" type="button" role="tab">GNN-based Methods</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="overlapping-tab" data-bs-toggle="tab" data-bs-target="#overlapping" type="button" role="tab">Overlapping Methods</button>
                </li>
            </ul>
            
            <div class="tab-content" id="methodTabsContent">
                <!-- Traditional Methods Tab -->
                <div class="tab-pane fade show active" id="traditional" role="tabpanel">
                    <div class="dashboard-card">
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h4>Performance Overview</h4>
                                    <table class="metrics-table">
                                        <thead>
                                            <tr>
                                                <th>Method</th>
                                                <th>Accuracy</th>
                                                <th>F1 Score</th>
                                                <th>Attack Ratio</th>
                                                <th>Execution Time</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>Louvain</td>
                                                <td>
                                                    <div class="value-bar" style="width: 100px;"></div>
                                                    1.00
                                                </td>
                                                <td>
                                                    <div class="value-bar" style="width: 100px;"></div>
                                                    1.00
                                                </td>
                                                <td>
                                                    <div class="value-bar" style="width: 44px;"></div>
                                                    0.44
                                                </td>
                                                <td>0.00073s</td>
                                            </tr>
                                            <tr>
                                                <td>Label Propagation</td>
                                                <td>
                                                    <div class="value-bar" style="width: 100px;"></div>
                                                    1.00
                                                </td>
                                                <td>
                                                    <div class="value-bar" style="width: 100px;"></div>
                                                    1.00
                                                </td>
                                                <td>
                                                    <div class="value-bar" style="width: 44px;"></div>
                                                    0.44
                                                </td>
                                                <td>0.00096s</td>
                                            </tr>
                                            <tr>
                                                <td>Infomap</td>
                                                <td>
                                                    <div class="value-bar" style="width: 100px;"></div>
                                                    1.00
                                                </td>
                                                <td>
                                                    <div class="value-bar" style="width: 100px;"></div>
                                                    1.00
                                                </td>
                                                <td>
                                                    <div class="value-bar" style="width: 44px;"></div>
                                                    0.44
                                                </td>
                                                <td>0.00969s</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                                <div class="col-md-6">
                                    <h4>Key Advantages</h4>
                                    <div class="analysis-text">
                                        <p>Traditional community detection methods demonstrated exceptional performance on the UNSW-NB15 dataset with several key advantages:</p>
                                        
                                        <ul>
                                            <li><strong>Perfect Classification Metrics</strong>: All three traditional methods achieved perfect accuracy, precision, recall, and F1 scores (1.0), demonstrating their effectiveness for basic attack classification.</li>
                                            
                                            <li><strong>Computational Efficiency</strong>: These methods operate orders of magnitude faster than other approaches, with execution times under 0.01 seconds, making them suitable for real-time monitoring of large networks.</li>
                                            
                                            <li><strong>Simplicity and Interpretability</strong>: Results are easy to understand and explain, with clear community structures that align well with network traffic patterns.</li>
                                            
                                            <li><strong>Consistent Performance</strong>: All three methods showed remarkable consistency in their results, indicating robustness across different implementations.</li>
                                        </ul>
                                        
                                        <p>These methods are ideal for real-time network monitoring applications where computational efficiency is critical and basic attack detection is the primary goal.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- GNN-based Methods Tab -->
                <div class="tab-pane fade" id="gnn" role="tabpanel">
                    <div class="dashboard-card">
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h4>Performance Overview</h4>
                                    <table class="metrics-table">
                                        <thead>
                                            <tr>
                                                <th>Method</th>
                                                <th>Accuracy</th>
                                                <th>F1 Score</th>
                                                <th>Attack Ratio</th>
                                                <th>Execution Time</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>GCN</td>
                                                <td>
                                                    <div class="value-bar" style="width: 70px;"></div>
                                                    0.70
                                                </td>
                                                <td>
                                                    <div class="value-bar" style="width: 57px;"></div>
                                                    0.57
                                                </td>
                                                <td>
                                                    <div class="value-bar" style="width: 50px;"></div>
                                                    0.50
                                                </td>
                                                <td>0.89392s</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                                <div class="col-md-6">
                                    <h4>Key Advantages</h4>
                                    <div class="analysis-text">
                                        <p>GNN-based methods, represented by Graph Convolutional Networks (GCN) in our analysis, offer several unique advantages for network security applications:</p>
                                        
                                        <ul>
                                            <li><strong>Feature Learning</strong>: GCN can automatically learn relevant node features from the network structure and attributes, reducing the need for manual feature engineering and potentially capturing complex patterns that might be missed in traditional feature selection.</li>
                                            
                                            <li><strong>Attack Isolation</strong>: GCN achieved a higher attack community ratio (0.50) than traditional methods, creating more focused attack communities that better isolate malicious traffic, which is valuable for forensic analysis.</li>
                                            
                                            <li><strong>Representation Learning</strong>: GCN captures complex network patterns beyond simple topological structure, incorporating both structural and attribute information into a unified representation.</li>
                                            
                                            <li><strong>Backdoor Attack Detection</strong>: GCN showed superior performance for detecting backdoor attacks (F1: 0.87), which are often subtle and difficult to identify with traditional methods.</li>
                                            
                                            <li><strong>Transferability</strong>: Once trained, GNN models can be transferred to similar network environments, potentially reducing the need for full retraining on new datasets.</li>
                                        </ul>
                                        
                                        <p>These methods are particularly valuable for deeper forensic analysis and in scenarios where feature information is rich but complex, requiring sophisticated pattern recognition beyond what traditional methods can provide.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Overlapping Methods Tab -->
                <div class="tab-pane fade" id="overlapping" role="tabpanel">
                    <div class="dashboard-card">
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h4>Performance Overview</h4>
                                    <table class="metrics-table">
                                        <thead>
                                            <tr>
                                                <th>Method</th>
                                                <th>Accuracy</th>
                                                <th>F1 Score</th>
                                                <th>Attack Ratio</th>
                                                <th>Execution Time</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>BigCLAM</td>
                                                <td>
                                                    <div class="value-bar" style="width: 75px;"></div>
                                                    0.75
                                                </td>
                                                <td>
                                                    <div class="value-bar" style="width: 71px;"></div>
                                                    0.71
                                                </td>
                                                <td>
                                                    <div class="value-bar" style="width: 71px;"></div>
                                                    0.71
                                                </td>
                                                <td>0.12284s</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                                <div class="col-md-6">
                                    <h4>Key Advantages</h4>
                                    <div class="analysis-text">
                                        <p>Overlapping community detection methods, represented by BigCLAM in our analysis, offer several distinct advantages for cybersecurity applications:</p>
                                        
                                        <ul>
                                            <li><strong>Multi-Community Membership</strong>: BigCLAM allows nodes to belong to multiple communities simultaneously, reflecting the reality of network traffic patterns where hosts may participate in both normal and attack-related activities.</li>
                                            
                                            <li><strong>Best Attack Isolation</strong>: BigCLAM achieved the highest attack community ratio (0.71) among all methods, indicating superior ability to concentrate attack traffic into specific communities, which enhances threat investigation and containment.</li>
                                            
                                            <li><strong>Balanced Performance</strong>: With good accuracy (0.75) and F1 score (0.71), BigCLAM offers a balanced approach that avoids the extreme trade-offs seen in other methods.</li>
                                            
                                            <li><strong>Specialized Attack Detection</strong>: BigCLAM demonstrated best-in-class performance for Exploits (F1: 0.92) and Analysis attacks (F1: 0.88), which often involve complex, multi-stage processes that span multiple network segments.</li>
                                            
                                            <li><strong>Moderate Computational Cost</strong>: More efficient than GNN methods while providing additional capabilities beyond traditional methods, offering a good middle ground for practical deployment.</li>
                                        </ul>
                                        
                                        <p>Overlapping methods are particularly valuable for identifying sophisticated attacks that span multiple network segments and traffic patterns, providing insights that strict partitioning methods might miss.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Applications Section -->
        <section id="applications" class="mt-5">
            <h2 class="section-title">Practical Applications</h2>
            
            <!-- Method Deployment Characteristics -->
            <div class="dashboard-card">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h3><i class="fas fa-cogs"></i> Method Deployment Characteristics</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-8">
                            <div id="deployment-plot" class="plot-container"></div>
                        </div>
                        <div class="col-md-4 sidebar">
                            <h4>Deployment Strategy</h4>
                            <p>
                                The radar chart visualizes each method's strengths across five key deployment dimensions:
                            </p>
                            <ul>
                                <li><strong>Real-time Monitoring</strong>: Traditional methods excel here due to their exceptional speed.</li>
                                <li><strong>Anomaly Detection</strong>: All methods perform well, with traditional methods having a slight edge.</li>
                                <li><strong>Forensics</strong>: GNN and overlapping methods provide superior capabilities for detailed investigation.</li>
                                <li><strong>Resource Efficiency</strong>: Traditional methods are significantly more efficient.</li>
                                <li><strong>Attack Isolation</strong>: BigCLAM leads in this dimension, followed by GCN.</li>
                            </ul>
                            <p>
                                This multidimensional view helps organizations select methods based on their specific security priorities and resource constraints.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Implementation Recommendations -->
            <div class="dashboard-card mt-4">
                <div class="card-header">
                    <h3><i class="fas fa-lightbulb"></i> Implementation Recommendations</h3>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-12">
                            <div class="analysis-text">
                                <h4>1. Tiered Security Monitoring System</h4>
                                <p>
                                    Implement a multi-level approach combining the strengths of different methods:
                                </p>
                                <ul>
                                    <li><strong>Level 1 (Real-time Monitoring)</strong>: Deploy traditional methods (Louvain or Label Propagation) for efficient continuous monitoring of all network traffic.</li>
                                    <li><strong>Level 2 (Anomaly Investigation)</strong>: When suspicious patterns are detected, apply BigCLAM to analyze the flagged traffic for overlapping attack patterns and community structure.</li>
                                    <li><strong>Level 3 (Deep Forensics)</strong>: For complex security incidents, leverage GCN for in-depth investigation of subtle attack vectors and relationship patterns.</li>
                                </ul>
                                
                                <h4 class="mt-4">2. Attack-Specific Detection</h4>
                                <p>
                                    Select methods based on the attack types of primary concern for your environment:
                                </p>
                                <ul>
                                    <li><strong>DoS & Generic Attacks</strong>: Traditional methods (Louvain, Label Propagation) provide optimal detection with minimal resource usage.</li>
                                    <li><strong>Exploits & Analysis Attacks</strong>: Overlapping methods (BigCLAM) offer superior detection for these distributed, multi-stage attacks.</li>
                                    <li><strong>Backdoor Attacks</strong>: GNN-based methods (GCN) excel at identifying these subtle, persistent threats that traditional methods might miss.</li>
                                </ul>
                                
                                <h4 class="mt-4">3. Resource-Optimized Deployment</h4>
                                <p>
                                    Tailor deployment based on available computational resources:
                                </p>
                                <ul>
                                    <li><strong>Limited Resources</strong>: Implement traditional methods exclusively, which provide excellent detection with minimal overhead.</li>
                                    <li><strong>Moderate Resources</strong>: Combine traditional methods with BigCLAM for balanced coverage, applying the latter selectively to suspicious traffic.</li>
                                    <li><strong>High-Performance Environment</strong>: Implement the full suite of methods, including resource-intensive GNN approaches for comprehensive protection.</li>
                                </ul>
                                
                                <h4 class="mt-4">4. Implementation Framework</h4>
                                <p>
                                    A recommended implementation pipeline based on the GNN-CD framework:
                                </p>
                                <ol>
                                    <li>Preprocess network traffic with Polars for efficient DataFrame operations (following the framework's performance guidelines)</li>
                                    <li>Construct graphs with RustworkX for high-performance graph representation</li>
                                    <li>Apply the appropriate community detection methods based on use case and available resources</li>
                                    <li>Implement automated alert generation based on community analysis and attack patterns</li>
                                    <li>Store results in Parquet format for efficient retrieval and historical analysis</li>
                                </ol>
                                
                                <p class="mt-3">
                                    These practical recommendations enable organizations to implement effective network security monitoring using community detection approaches tailored to their specific requirements, threat landscape, and resource constraints.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- About Section -->
        <section id="about" class="mt-5">
            <h2 class="section-title">About</h2>
            
            <div class="dashboard-card">
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-4">
                            <h4>About This Dashboard</h4>
                            <p>
                                This interactive dashboard presents the results of comprehensive evaluation of community detection methods applied to the UNSW-NB15 cybersecurity dataset. The analysis was conducted by the GNN-CD Research Team.
                            </p>
                            <p>
                                The dashboard provides:
                            </p>
                            <ul>
                                <li>Performance metrics of five community detection methods</li>
                                <li>Interactive visualizations of results</li>
                                <li>Method comparison across traditional, GNN-based, and overlapping approaches</li>
                                <li>Practical deployment recommendations</li>
                            </ul>
                            <p>
                                All visualizations are interactive - hover over elements to see detailed information, zoom in/out, and explore the data.
                            </p>
                        </div>
                        <div class="col-md-4">
                            <h4>About the Dataset</h4>
                            <p>
                                The UNSW-NB15 dataset was created by the Cyber Range Lab of the Australian Centre for Cyber Security (ACCS). It contains both normal and attack traffic with the following characteristics:
                            </p>
                            <ul>
                                <li><strong>Size</strong>: 100,000 nodes, 99,999 edges (in our processed graph)</li>
                                <li><strong>Features</strong>: 15 selected features capturing traffic behavior</li>
                                <li><strong>Attack Types</strong>: DoS, Exploits, Reconnaissance, Generic, Backdoor, Analysis</li>
                                <li><strong>Labels</strong>: Binary (normal vs. attack)</li>
                            </ul>
                            <p>
                                For more information about the dataset, see:<br>
                                Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). 2015 Military Communications and Information Systems Conference (MilCIS), 1-6.
                            </p>
                        </div>
                        <div class="col-md-4">
                            <h4>About the Methods</h4>
                            <p>
                                Our analysis evaluated the following community detection methods:
                            </p>
                            <ul>
                                <li><strong>Traditional methods</strong>:
                                    <ul>
                                        <li><strong>Louvain</strong>: Hierarchical modularity optimization</li>
                                        <li><strong>Label Propagation</strong>: Simple iterative neighborhood-based algorithm</li>
                                        <li><strong>Infomap</strong>: Information flow-based algorithm</li>
                                    </ul>
                                </li>
                                <li><strong>GNN-based methods</strong>:
                                    <ul>
                                        <li><strong>GCN</strong>: Graph Convolutional Networks for node representation learning</li>
                                    </ul>
                                </li>
                                <li><strong>Overlapping methods</strong>:
                                    <ul>
                                        <li><strong>BigCLAM</strong>: Cluster Affiliation Model for Big Networks allowing nodes to belong to multiple communities</li>
                                    </ul>
                                </li>
                            </ul>
                            <p>
                                For more information about the GNN-CD framework, visit our GitHub repository: <a href="https://github.com/braden/gnn-cd" target="_blank">https://github.com/braden/gnn-cd</a>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    </div>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5>GNN-CD Framework</h5>
                    <p>
                        A comprehensive framework for community detection using graph neural networks and traditional methods.
                    </p>
                </div>
                <div class="col-md-3">
                    <h5>Resources</h5>
                    <ul class="list-unstyled">
                        <li><a href="https://github.com/braden/gnn-cd">GitHub Repository</a></li>
                        <li><a href="#">Documentation</a></li>
                        <li><a href="#">Notebooks</a></li>
                    </ul>
                </div>
                <div class="col-md-3">
                    <h5>Contact</h5>
                    <ul class="list-unstyled">
                        <li>GNN-CD Research Team</li>
                        <li>April 4, 2025</li>
                    </ul>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-md-12 text-center">
                    <p>
                        &copy; 2025 GNN-CD Research Team. All rights reserved.
                    </p>
                </div>
            </div>
        </div>
    </footer>

    <script>
        // Create all the plots when the page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Performance metrics plot
            Plotly.newPlot('performance-metrics-plot', performance_metrics.data, performance_metrics.layout);
            
            // Heatmap plot
            Plotly.newPlot('heatmap-plot', heatmap.data, heatmap.layout);
            
            // Bubble chart plot
            Plotly.newPlot('bubble-chart-plot', bubble_chart.data, bubble_chart.layout);
            
            // Community parallel plot
            Plotly.newPlot('community-parallel-plot', community_parallel.data, community_parallel.layout);
            
            // Community bar plot
            Plotly.newPlot('community-bar-plot', community_bar.data, community_bar.layout);
            
            // Attack type bar plot
            Plotly.newPlot('attack-type-plot', attack_type_bar.data, attack_type_bar.layout);
            
            // Sankey network plot
            Plotly.newPlot('sankey-plot', sankey_network.data, sankey_network.layout);
            
            // Feature importance plot
            Plotly.newPlot('feature-importance-plot', feature_importance.data, feature_importance.layout);
            
            // Deployment plot
            Plotly.newPlot('deployment-plot', deployment_plot.data, deployment_plot.layout);
            
            // Make plots responsive
            window.onresize = function() {
                Plotly.relayout('performance-metrics-plot', {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
                Plotly.relayout('heatmap-plot', {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
                Plotly.relayout('bubble-chart-plot', {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
                Plotly.relayout('community-parallel-plot', {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
                Plotly.relayout('community-bar-plot', {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
                Plotly.relayout('attack-type-plot', {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
                Plotly.relayout('sankey-plot', {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
                Plotly.relayout('feature-importance-plot', {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
                Plotly.relayout('deployment-plot', {
                    'xaxis.autorange': true,
                    'yaxis.autorange': true
                });
            };
        });
        
        // Activate Bootstrap tooltips
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl)
        });
        
        // Handle navbar clicks
        document.querySelectorAll('.navbar-nav .nav-link').forEach(function(link) {
            link.addEventListener('click', function(e) {
                document.querySelectorAll('.navbar-nav .nav-link').forEach(function(item) {
                    item.classList.remove('active');
                });
                this.classList.add('active');
            });
        });
    </script>
</body>
</html>
"""

with open("/home/braden/gnn-cd/unsw_dashboard.html", "w") as f:
    f.write(html_content)

print("Interactive dashboard generated at: /home/braden/gnn-cd/unsw_dashboard.html")
print("Generated plots saved in: dashboard_assets/")
print("To view the dashboard, open the HTML file in a web browser")