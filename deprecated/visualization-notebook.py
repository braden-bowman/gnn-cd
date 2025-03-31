# Advanced Visualization for Community Detection
# =============================================

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')

# For interactive/dynamic visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Some visualizations will be limited.")

# For 3D visualization
try:
    from mpl_toolkits.mplot3d import Axes3D
    MPL_3D_AVAILABLE = True
except ImportError:
    MPL_3D_AVAILABLE = False
    print("Matplotlib 3D tools not available. 3D visualizations will be limited.")

# For network visualization using node-link diagrams with D3.js-like rendering
try:
    import networkx.drawing.nx_pylab as nx_pylab
    NX_PYLAB_AVAILABLE = True
except ImportError:
    NX_PYLAB_AVAILABLE = False

# For community evaluation
try:
    from cdlib import evaluation
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False

# Import utility functions from other notebooks
# In a real scenario, these would be imported from a module
from data_prep import (load_data, generate_synthetic_graph, plot_graph)


# Function to create circular community layout
def community_layout(G, communities):
    """
    Position nodes in a circular layout for each community
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to visualize
    communities: dict or list
        Community assignments (dict) or list of community nodes
        
    Returns:
    --------
    pos: dict
        Node positions
    """
    # Convert list of communities to dict if needed
    if isinstance(communities, list):
        comm_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_dict[node] = i
        communities = comm_dict
    
    # Create positions dictionary
    pos = {}
    # For each community, create a circle layout
    for comm_id in set(communities.values()):
        # Get nodes in the community
        comm_nodes = [n for n, c in communities.items() if c == comm_id]
        # Skip empty communities
        if not comm_nodes:
            continue
        # Create a subgraph
        subgraph = G.subgraph(comm_nodes)
        # Create a circular layout for the subgraph
        c_pos = nx.circular_layout(subgraph)
        
        # Scale and shift the layout to separate communities
        offset = np.array([comm_id * 2, 0])
        for node, p in c_pos.items():
            pos[node] = p + offset
    
    return pos


# Function to create layered community layout
def layered_community_layout(G, communities, layer_distance=3):
    """
    Position nodes in a layered layout based on communities
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to visualize
    communities: dict or list
        Community assignments (dict) or list of community nodes
    layer_distance: float
        Distance between layers
        
    Returns:
    --------
    pos: dict
        Node positions
    """
    # Convert list of communities to dict if needed
    if isinstance(communities, list):
        comm_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_dict[node] = i
        communities = comm_dict
    
    # Create positions dictionary
    pos = {}
    # Get unique community IDs
    comm_ids = sorted(set(communities.values()))
    
    for layer, comm_id in enumerate(comm_ids):
        # Get nodes in the community
        comm_nodes = [n for n, c in communities.items() if c == comm_id]
        # Skip empty communities
        if not comm_nodes:
            continue
        
        # Create a subgraph
        subgraph = G.subgraph(comm_nodes)
        # Create a spring layout for the subgraph
        c_pos = nx.spring_layout(subgraph, seed=42)
        
        # Shift the layout to separate layers
        offset = np.array([0, layer * layer_distance])
        for node, p in c_pos.items():
            pos[node] = p + offset
    
    return pos


# Function to create a combined layout for better community visualization
def combined_community_layout(G, communities, scale=1.0):
    """
    Position nodes using a combination of layouts for optimal community visualization
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to visualize
    communities: dict or list
        Community assignments (dict) or list of community nodes
    scale: float
        Scaling factor for layout
        
    Returns:
    --------
    pos: dict
        Node positions
    """
    # Convert list of communities to dict if needed
    if isinstance(communities, list):
        comm_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_dict[node] = i
        communities = comm_dict
    
    # First get a global layout
    global_pos = nx.spring_layout(G, seed=42)
    
    # Create positions dictionary
    pos = {}
    # Get community centroids from global layout
    centroids = {}
    
    for comm_id in set(communities.values()):
        # Get nodes in the community
        comm_nodes = [n for n, c in communities.items() if c == comm_id]
        # Skip empty communities
        if not comm_nodes:
            continue
        
        # Calculate centroid
        centroid = np.mean([global_pos[n] for n in comm_nodes], axis=0)
        centroids[comm_id] = centroid
    
    # Scale centroids to increase separation
    mean_centroid = np.mean(list(centroids.values()), axis=0)
    for comm_id, centroid in centroids.items():
        centroids[comm_id] = (centroid - mean_centroid) * scale + mean_centroid
    
    # Create local layouts for each community
    for comm_id in set(communities.values()):
        # Get nodes in the community
        comm_nodes = [n for n, c in communities.items() if c == comm_id]
        # Skip empty communities
        if not comm_nodes:
            continue
        
        # Create a subgraph
        subgraph = G.subgraph(comm_nodes)
        # Create a spring layout for the subgraph
        c_pos = nx.spring_layout(subgraph, seed=comm_id)
        
        # Scale down the local layout
        c_pos = {n: p * 0.5 for n, p in c_pos.items()}
        
        # Shift to the community centroid
        for node, p in c_pos.items():
            pos[node] = p + centroids[comm_id]
    
    return pos


# Function to visualize communities using a static plot
def visualize_communities(G, communities, layout='spring', 
                         title="Community Structure", figsize=(12, 10)):
    """
    Visualize communities in a network
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to visualize
    communities: dict or list
        Community assignments (dict) or list of community nodes
    layout: str
        Layout type ('spring', 'circular', 'community', 'layered', 'combined')
    title: str
        Plot title
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    
    # Convert list of communities to dict if needed
    if isinstance(communities, list):
        comm_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_dict[node] = i
        communities = comm_dict
    
    # Get the layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'community':
        pos = community_layout(G, communities)
    elif layout == 'layered':
        pos = layered_community_layout(G, communities)
    elif layout == 'combined':
        pos = combined_community_layout(G, communities)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Get community assignments for each node
    community_ids = [communities.get(n, -1) for n in G.nodes()]
    
    # Draw the network
    nx.draw_networkx(
        G, pos=pos, 
        node_color=community_ids, 
        cmap=plt.cm.rainbow,
        node_size=100,
        with_labels=False,
        edge_color='gray',
        alpha=0.8
    )
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(
        vmin=min(community_ids), vmax=max(community_ids)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Community')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Function to visualize community evolution over time
def visualize_community_evolution(graphs, communities_list, layout='spring',
                               title="Community Evolution", figsize=(16, 12)):
    """
    Visualize community evolution over time
    
    Parameters:
    -----------
    graphs: list
        List of NetworkX graphs for each time step
    communities_list: list
        List of community assignments for each time step
    layout: str
        Layout type ('spring', 'community', 'fixed')
    title: str
        Plot title
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    n_time_steps = len(graphs)
    n_cols = min(3, n_time_steps)
    n_rows = (n_time_steps - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Get a fixed layout if requested
    if layout == 'fixed':
        # Use first graph for layout
        fixed_pos = nx.spring_layout(graphs[0], seed=42)
    
    # Draw each time step
    for t, (G, communities) in enumerate(zip(graphs, communities_list)):
        # Get the axis
        if n_rows == 1 and n_cols == 1:
            ax = axes
        else:
            row = t // n_cols
            col = t % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[t]
        
        # Get the layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'community':
            pos = community_layout(G, communities)
        elif layout == 'fixed':
            # Create positions for any new nodes
            missing_nodes = set(G.nodes()) - set(fixed_pos.keys())
            if missing_nodes:
                sub_pos = nx.spring_layout(G.subgraph(missing_nodes), seed=42)
                fixed_pos.update(sub_pos)
            pos = {n: fixed_pos.get(n, (0, 0)) for n in G.nodes()}
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Convert list of communities to dict if needed
        if isinstance(communities, list):
            comm_dict = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    comm_dict[node] = i
            communities = comm_dict
        
        # Get community assignments for each node
        community_ids = [communities.get(n, -1) for n in G.nodes()]
        
        # Draw the network
        nx.draw_networkx(
            G, pos=pos, 
            node_color=community_ids, 
            cmap=plt.cm.rainbow,
            node_size=80,
            with_labels=False,
            edge_color='gray',
            alpha=0.7,
            ax=ax
        )
        
        ax.set_title(f'Time Step {t+1}')
        ax.axis('off')
    
    # Remove empty subplots
    for t in range(len(graphs), n_rows * n_cols):
        row = t // n_cols
        col = t % n_cols
        if n_rows > 1 and n_cols > 1:
            fig.delaxes(axes[row, col])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


# Function to create a community membership heatmap
def community_membership_heatmap(communities_list, node_list=None, figsize=(12, 10)):
    """
    Create a heatmap showing community membership over time
    
    Parameters:
    -----------
    communities_list: list
        List of community assignments for each time step
    node_list: list
        List of nodes to include (if None, uses all nodes)
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    # Prepare data for heatmap
    n_time_steps = len(communities_list)
    
    # Get all nodes if not provided
    if node_list is None:
        # Get all unique nodes across all time steps
        node_list = set()
        for communities in communities_list:
            if isinstance(communities, dict):
                node_list.update(communities.keys())
            else:
                for comm in communities:
                    node_list.update(comm)
        node_list = sorted(node_list)
    
    # Create matrix of community assignments
    community_matrix = np.zeros((len(node_list), n_time_steps))
    
    for t, communities in enumerate(communities_list):
        # Convert list of communities to dict if needed
        if isinstance(communities, list):
            comm_dict = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    comm_dict[node] = i
            communities = comm_dict
        
        # Fill in the matrix
        for i, node in enumerate(node_list):
            community_matrix[i, t] = communities.get(node, -1)
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(community_matrix, cmap='viridis', 
               xticklabels=[f'T{t+1}' for t in range(n_time_steps)],
               yticklabels=node_list if len(node_list) <= 50 else False)
    plt.title('Community Membership Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Node')
    plt.tight_layout()
    plt.show()


# Function to create an alluvial diagram for community evolution
def alluvial_diagram(communities_list, node_list=None, figsize=(15, 10)):
    """
    Create an alluvial diagram showing community evolution over time
    
    Parameters:
    -----------
    communities_list: list
        List of community assignments for each time step
    node_list: list
        List of nodes to include (if None, uses all nodes)
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Cannot create alluvial diagram.")
        return
    
    # Get all nodes if not provided
    if node_list is None:
        # Get all unique nodes across all time steps
        node_list = set()
        for communities in communities_list:
            if isinstance(communities, dict):
                node_list.update(communities.keys())
            else:
                for comm in communities:
                    node_list.update(comm)
        node_list = sorted(node_list)
    
    # Prepare data for the diagram
    n_time_steps = len(communities_list)
    
    # Create source, target, and value lists for Sankey diagram
    sources = []
    targets = []
    values = []
    labels = []
    
    # Create node labels for each community at each time step
    for t in range(n_time_steps):
        communities = communities_list[t]
        
        # Convert list of communities to dict if needed
        if isinstance(communities, list):
            comm_dict = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    comm_dict[node] = i
            communities = comm_dict
        
        # Get unique community IDs for this time step
        comm_ids = sorted(set(communities.values()))
        
        # Add labels for each community
        for comm_id in comm_ids:
            labels.append(f'T{t+1}_C{comm_id}')
    
    # Add links between communities at consecutive time steps
    node_index = 0  # Current index in the labels list
    
    for t in range(n_time_steps - 1):
        communities_t = communities_list[t]
        communities_t1 = communities_list[t+1]
        
        # Convert to dicts if needed
        if isinstance(communities_t, list):
            comm_dict_t = {}
            for i, comm in enumerate(communities_t):
                for node in comm:
                    comm_dict_t[node] = i
            communities_t = comm_dict_t
        
        if isinstance(communities_t1, list):
            comm_dict_t1 = {}
            for i, comm in enumerate(communities_t1):
                for node in comm:
                    comm_dict_t1[node] = i
            communities_t1 = comm_dict_t1
        
        # Get unique community IDs for these time steps
        comm_ids_t = sorted(set(communities_t.values()))
        comm_ids_t1 = sorted(set(communities_t1.values()))
        
        # Count transitions between communities
        transitions = {}
        
        for node in node_list:
            if node in communities_t and node in communities_t1:
                source_comm = communities_t[node]
                target_comm = communities_t1[node]
                
                source_idx = node_index + comm_ids_t.index(source_comm)
                target_idx = node_index + len(comm_ids_t) + comm_ids_t1.index(target_comm)
                
                transition_key = (source_idx, target_idx)
                transitions[transition_key] = transitions.get(transition_key, 0) + 1
        
        # Add transitions to the lists
        for (source_idx, target_idx), count in transitions.items():
            sources.append(source_idx)
            targets.append(target_idx)
            values.append(count)
        
        # Update node index for the next time step
        node_index += len(comm_ids_t)
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    
    fig.update_layout(
        title_text="Community Evolution Over Time",
        font_size=10,
        width=figsize[0]*60,
        height=figsize[1]*60
    )
    
    fig.show()


# Function to create a 3D visualization of communities
def visualize_communities_3d(G, communities, layout='spring', 
                           title="Community Structure in 3D", figsize=(12, 10)):
    """
    Create a 3D visualization of communities
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to visualize
    communities: dict or list
        Community assignments
    layout: str
        Layout type ('spring', 'community')
    title: str
        Plot title
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    if not MPL_3D_AVAILABLE:
        print("Matplotlib 3D tools not available. Cannot create 3D visualization.")
        return
    
    # Convert list of communities to dict if needed
    if isinstance(communities, list):
        comm_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_dict[node] = i
        communities = comm_dict
    
    # Get 2D layout first
    if layout == 'spring':
        pos_2d = nx.spring_layout(G, seed=42)
    elif layout == 'community':
        pos_2d = community_layout(G, communities)
    else:
        pos_2d = nx.spring_layout(G, seed=42)
    
    # Convert to 3D: use community ID for z-coordinate
    pos_3d = {}
    for node, (x, y) in pos_2d.items():
        z = communities.get(node, 0) * 0.5  # Scale z by community ID
        pos_3d[node] = (x, y, z)
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw nodes
    node_colors = [communities.get(n, -1) for n in G.nodes()]
    
    # Extract node positions
    node_xyz = np.array([pos_3d[v] for v in G.nodes()])
    
    # Plot nodes
    ax.scatter(
        node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2],
        c=node_colors,
        cmap=plt.cm.rainbow,
        s=100,
        alpha=0.8
    )
    
    # Draw edges
    for u, v in G.edges():
        x = [pos_3d[u][0], pos_3d[v][0]]
        y = [pos_3d[u][1], pos_3d[v][1]]
        z = [pos_3d[u][2], pos_3d[v][2]]
        ax.plot(x, y, z, 'gray', alpha=0.5)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(
        vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Community')
    
    ax.set_title(title)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()


# Function to create an interactive 3D visualization
def interactive_3d_visualization(G, communities, layout='spring', title="Interactive Community Visualization"):
    """
    Create an interactive 3D visualization of communities using Plotly
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to visualize
    communities: dict or list
        Community assignments
    layout: str
        Layout type ('spring', 'community')
    title: str
        Plot title
        
    Returns:
    --------
    None
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Cannot create interactive visualization.")
        return
    
    # Convert list of communities to dict if needed
    if isinstance(communities, list):
        comm_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_dict[node] = i
        communities = comm_dict
    
    # Get 2D layout first
    if layout == 'spring':
        pos_2d = nx.spring_layout(G, seed=42)
    elif layout == 'community':
        pos_2d = community_layout(G, communities)
    else:
        pos_2d = nx.spring_layout(G, seed=42)
    
    # Convert to 3D: use community ID for z-coordinate
    pos_3d = {}
    for node, (x, y) in pos_2d.items():
        z = communities.get(node, 0) * 0.5  # Scale z by community ID
        pos_3d[node] = (x, y, z)
    
    # Create node trace
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y, z = pos_3d[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(f'Node {node}, Community {communities.get(node, -1)}')
        node_color.append(communities.get(node, -1))
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=8,
            color=node_color,
            colorscale='Viridis',
            colorbar=dict(title='Community'),
            line=dict(width=0.5, color='white')
        ),
        text=node_text,
        hoverinfo='text'
    )
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_z = []
    
    for u, v in G.edges():
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(150,150,150,0.3)', width=1),
        hoverinfo='none'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title=title,
        showlegend=False,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            bgcolor='rgba(255,255,255,1)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.show()


# Function to visualize node embeddings
def visualize_embeddings(embeddings, communities, method='tsne', 
                       title="Node Embeddings", figsize=(12, 10)):
    """
    Visualize node embeddings colored by community
    
    Parameters:
    -----------
    embeddings: numpy.ndarray
        Node embeddings
    communities: dict or list
        Community assignments
    method: str
        Dimensionality reduction method ('tsne', 'pca')
    title: str
        Plot title
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    # Convert list of communities to dict if needed
    if isinstance(communities, list):
        comm_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_dict[node] = i
        communities = comm_dict
    
    # Get community assignments as a list
    node_indices = sorted(communities.keys())
    community_ids = [communities[i] for i in node_indices]
    
    # Extract embeddings for these nodes
    node_embeddings = embeddings[node_indices]
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # Default to PCA
        reducer = PCA(n_components=2, random_state=42)
    
    embeddings_2d = reducer.fit_transform(node_embeddings)
    
    # Create plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=community_ids,
        cmap=plt.cm.rainbow,
        s=100,
        alpha=0.8
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Community')
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Function to visualize community detection metrics over time
def visualize_metrics_over_time(metrics_list, figsize=(12, 8)):
    """
    Visualize community detection metrics over time
    
    Parameters:
    -----------
    metrics_list: list
        List of dictionaries containing metrics for each time step
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    # Extract metrics from the list
    time_steps = list(range(1, len(metrics_list) + 1))
    
    # Get all available metrics
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    # Create plot
    plt.figure(figsize=figsize)
    
    for metric in sorted(all_metrics):
        # Skip if metric is not present in all time steps
        if not all(metric in m for m in metrics_list):
            continue
        
        # Extract values for this metric
        values = [m[metric] for m in metrics_list]
        
        # Plot
        plt.plot(time_steps, values, 'o-', label=metric.upper())
    
    plt.title('Community Detection Metrics Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Score')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Function to create an animated visualization of community evolution
def animate_community_evolution(graphs, communities_list, layout='spring', 
                             title="Community Evolution", figsize=(12, 10)):
    """
    Create an animated visualization of community evolution
    
    Parameters:
    -----------
    graphs: list
        List of NetworkX graphs for each time step
    communities_list: list
        List of community assignments for each time step
    layout: str
        Layout type ('spring', 'fixed')
    title: str
        Plot title
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get a fixed layout if requested
    if layout == 'fixed':
        # Use first graph for layout
        fixed_pos = nx.spring_layout(graphs[0], seed=42)
        
        # Add positions for any new nodes in subsequent graphs
        for G in graphs[1:]:
            missing_nodes = set(G.nodes()) - set(fixed_pos.keys())
            if missing_nodes:
                sub_pos = nx.spring_layout(G.subgraph(missing_nodes), seed=42)
                fixed_pos.update(sub_pos)
    
    # Function to draw a frame
    def draw_frame(t):
        ax.clear()
        G = graphs[t]
        communities = communities_list[t]
        
        # Convert list of communities to dict if needed
        if isinstance(communities, list):
            comm_dict = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    comm_dict[node] = i
            communities = comm_dict
        
        # Get the layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'fixed':
            pos = {n: fixed_pos.get(n, (0, 0)) for n in G.nodes()}
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Get community assignments for each node
        community_ids = [communities.get(n, -1) for n in G.nodes()]
        
        # Draw the network
        nx.draw_networkx(
            G, pos=pos, 
            node_color=community_ids, 
            cmap=plt.cm.rainbow,
            node_size=100,
            with_labels=False,
            edge_color='gray',
            alpha=0.8,
            ax=ax
        )
        
        ax.set_title(f'{title} - Time Step {t+1}')
        ax.axis('off')
    
    # Create animation
    ani = FuncAnimation(fig, draw_frame, frames=len(graphs), interval=1000, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return ani


# Function to visualize community overlaps
def visualize_overlapping_communities(G, communities, title="Overlapping Communities", figsize=(12, 10)):
    """
    Visualize overlapping communities
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to visualize
    communities: list
        List of overlapping communities (each node can appear in multiple communities)
    title: str
        Plot title
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=figsize)
    
    # Get positions
    pos = nx.spring_layout(G, seed=42)
    
    # Base layer: draw all nodes in light gray
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=100)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Get a color map with distinct colors
    cmap = plt.cm.rainbow
    colors = cmap(np.linspace(0, 1, len(communities)))
    
    # Draw each community with a different color
    for i, (community, color) in enumerate(zip(communities, colors)):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=[color], 
                             alpha=0.6, node_size=100, label=f'Community {i+1}')
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Additionally, create a visualization showing node membership count
    plt.figure(figsize=figsize)
    
    # Count how many communities each node belongs to
    membership_count = {}
    for comm in communities:
        for node in comm:
            membership_count[node] = membership_count.get(node, 0) + 1
    
    # Color nodes by membership count
    node_colors = [membership_count.get(n, 0) for n in G.nodes()]
    
    # Draw the network
    nx.draw_networkx(G, pos, node_color=node_colors, cmap='viridis', 
                   with_labels=True, node_size=100, font_size=8)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(
        vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Number of Communities')
    
    plt.title('Node Membership Count')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Function to visualize community importance metrics
def visualize_community_importance(G, communities, metric='size', 
                                title="Community Importance", figsize=(12, 10)):
    """
    Visualize communities with size reflecting importance metrics
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to visualize
    communities: dict or list
        Community assignments
    metric: str
        Importance metric ('size', 'density', 'modularity')
    title: str
        Plot title
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    # Convert list of communities to dict if needed
    if isinstance(communities, list):
        comm_dict = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_dict[node] = i
        communities = comm_dict
    
    # Get unique community IDs
    community_ids = sorted(set(communities.values()))
    
    # Calculate importance metrics for each community
    importance = {}
    
    for comm_id in community_ids:
        # Get nodes in this community
        comm_nodes = [n for n, c in communities.items() if c == comm_id]
        
        if metric == 'size':
            # Use community size
            importance[comm_id] = len(comm_nodes)
        elif metric == 'density':
            # Use community density
            subgraph = G.subgraph(comm_nodes)
            if len(comm_nodes) > 1:
                importance[comm_id] = nx.density(subgraph)
            else:
                importance[comm_id] = 0
        elif metric == 'modularity':
            # Use local modularity (simplified)
            subgraph = G.subgraph(comm_nodes)
            internal_edges = subgraph.number_of_edges()
            external_edges = sum(G.degree(n) for n in comm_nodes) - 2 * internal_edges
            if internal_edges + external_edges > 0:
                importance[comm_id] = internal_edges / (internal_edges + external_edges)
            else:
                importance[comm_id] = 0
        else:
            # Default to community size
            importance[comm_id] = len(comm_nodes)
    
    # Normalize importance values
    max_imp = max(importance.values())
    min_imp = min(importance.values())
    if max_imp > min_imp:
        normalized_imp = {c: (v - min_imp) / (max_imp - min_imp) * 0.8 + 0.2 
                         for c, v in importance.items()}
    else:
        normalized_imp = {c: 0.5 for c in importance}
    
    # Create a layout that groups nodes by community
    pos = community_layout(G, communities)
    
    # Draw the network
    plt.figure(figsize=figsize)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Draw nodes colored by community, sized by importance
    for comm_id in community_ids:
        # Get nodes in this community
        comm_nodes = [n for n, c in communities.items() if c == comm_id]
        
        # Calculate node size based on community importance
        node_size = 100 + 400 * normalized_imp[comm_id]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, nodelist=comm_nodes, 
                             node_color=[comm_id] * len(comm_nodes),
                             node_size=node_size, 
                             cmap=plt.cm.rainbow,
                             alpha=0.8)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(
        vmin=min(community_ids), vmax=max(community_ids)))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Community')
    
    plt.title(f'{title} ({metric.capitalize()})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Function implementing Vehlow et al. visualization
def vehlow_visualization(graphs, communities_list, figsize=(15, 10)):
    """
    Implement the visualization approach from Vehlow et al.
    
    Parameters:
    -----------
    graphs: list
        List of NetworkX graphs for each time step
    communities_list: list
        List of community assignments for each time step
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Convert all list-based communities to dicts
    for t, communities in enumerate(communities_list):
        if isinstance(communities, list):
            comm_dict = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    comm_dict[node] = i
            communities_list[t] = comm_dict
    
    # Parameters
    n_time_steps = len(graphs)
    x_spacing = 1.0  # Spacing between time steps
    y_spacing = 0.1  # Spacing between communities
    comm_height = 0.8  # Height of community blocks
    
    # Get all unique nodes across all time steps
    all_nodes = set()
    for t, G in enumerate(graphs):
        all_nodes.update(G.nodes())
    all_nodes = sorted(all_nodes)
    
    # Track node positions for drawing transitions
    node_positions = {}  # (t, node) -> (x, y)
    
    # Draw each time step
    for t, (G, communities) in enumerate(zip(graphs, communities_list)):
        x_offset = t * x_spacing
        
        # Get communities at this time step
        comm_ids = sorted(set(communities.values()))
        
        # Calculate community sizes
        comm_sizes = {}
        for comm_id in comm_ids:
            comm_nodes = [n for n, c in communities.items() if c == comm_id]
            comm_sizes[comm_id] = len(comm_nodes)
        
        # Calculate y positions for communities
        y_positions = {}
        y_offset = 0
        for comm_id in comm_ids:
            size = comm_sizes[comm_id]
            y_positions[comm_id] = y_offset
            y_offset += size * y_spacing + comm_height
        
        # Draw community blocks
        for comm_id in comm_ids:
            size = comm_sizes[comm_id]
            y_pos = y_positions[comm_id]
            
            # Calculate block dimensions
            block_height = size * y_spacing + comm_height
            
            # Draw the block
            rect = plt.Rectangle((x_offset, y_pos), 0.8, block_height, 
                               facecolor=plt.cm.rainbow(comm_id / len(comm_ids)),
                               alpha=0.7, edgecolor='black')
            plt.gca().add_patch(rect)
            
            # Add community label
            plt.text(x_offset + 0.4, y_pos + block_height / 2, f'C{comm_id}',
                   ha='center', va='center')
            
            # Track positions of nodes in this community
            node_y = y_pos + y_spacing
            for n in G.nodes():
                if communities.get(n) == comm_id:
                    node_positions[(t, n)] = (x_offset + 0.4, node_y)
                    node_y += y_spacing
    
    # Draw transitions between consecutive time steps
    for t in range(n_time_steps - 1):
        for node in all_nodes:
            if (t, node) in node_positions and (t+1, node) in node_positions:
                x1, y1 = node_positions[(t, node)]
                x2, y2 = node_positions[(t+1, node)]
                plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
    
    plt.title('Community Evolution Visualization (Vehlow et al.)')
    plt.xlabel('Time Step')
    plt.xticks([t * x_spacing + 0.4 for t in range(n_time_steps)], 
             [f'T{t+1}' for t in range(n_time_steps)])
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    print("Advanced Visualization for Community Detection")
    print("="*50)
    
    # Generate a synthetic graph with communities
    print("\n1. Generating a synthetic graph with communities...")
    n_communities = 5
    G, ground_truth = generate_synthetic_graph('sbm', n_nodes=100, 
                                             n_communities=n_communities,
                                             p_in=0.3, p_out=0.05)
    
    # Convert ground truth to community format
    communities = {i: G.nodes[i]['community'] for i in range(len(G))}
    
    # Create standard visualization
    print("\n2. Creating standard community visualization...")
    visualize_communities(G, communities, layout='spring', 
                        title="Standard Community Visualization")
    
    # Create community-aware layout
    print("\n3. Creating community-aware layout visualization...")
    visualize_communities(G, communities, layout='community', 
                        title="Community-Aware Layout")
    
    # Create combined layout
    print("\n4. Creating combined layout visualization...")
    visualize_communities(G, communities, layout='combined', 
                        title="Combined Community Layout")
    
    # Create 3D visualization
    print("\n5. Creating 3D community visualization...")
    visualize_communities_3d(G, communities, layout='community',
                           title="3D Community Visualization")
    
    # Create interactive visualization if Plotly is available
    if PLOTLY_AVAILABLE:
        print("\n6. Creating interactive community visualization...")
        interactive_3d_visualization(G, communities, layout='community',
                                   title="Interactive Community Visualization")
    
    # Generate a sequence of dynamic graphs
    print("\n7. Generating a sequence of dynamic graphs...")
    graphs = []
    communities_list = []
    
    for t in range(3):
        # Modify the previous graph by changing some community assignments
        if t == 0:
            G_t = G.copy()
            comms_t = communities.copy()
        else:
            G_t = G.copy()
            comms_t = communities_list[-1].copy()
            
            # Change community for some nodes
            for i in range(len(G_t) // 10):  # Change 10% of nodes
                node = np.random.randint(0, len(G_t))
                current_comm = comms_t[node]
                new_comm = (current_comm + 1) % n_communities
                comms_t[node] = new_comm
        
        graphs.append(G_t)
        communities_list.append(comms_t)
    
    # Visualize community evolution
    print("\n8. Visualizing community evolution...")
    visualize_community_evolution(graphs, communities_list, layout='fixed',
                               title="Community Evolution")
    
    # Create community membership heatmap
    print("\n9. Creating community membership heatmap...")
    community_membership_heatmap(communities_list)
    
    # Create an alluvial diagram if Plotly is available
    if PLOTLY_AVAILABLE:
        print("\n10. Creating alluvial diagram...")
        alluvial_diagram(communities_list)
    
    # Visualize community importance
    print("\n11. Visualizing community importance...")
    visualize_community_importance(G, communities, metric='size',
                                title="Community Importance by Size")
    
    # Create Vehlow et al. visualization
    print("\n12. Creating Vehlow et al. visualization...")
    vehlow_visualization(graphs, communities_list)
    
    print("\nVisualization complete!")
