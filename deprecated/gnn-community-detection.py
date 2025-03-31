# Graph Neural Network-Based Community Detection
# =============================================

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import time
import warnings
warnings.filterwarnings('ignore')

# For PyTorch and PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DeepGraphInfomax
    from torch_geometric.data import Data
    from torch_geometric.utils import to_networkx, from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. Please install it for GNN-based methods.")

# For community evaluation
try:
    from cdlib import evaluation
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False

# Import utility functions from the data preparation notebook
# In a real scenario, these would be imported from a module
from data_prep import (load_data, create_graph_from_edgelist, convert_nx_to_pytorch_geometric,
                      generate_synthetic_graph, plot_graph)


# Function to convert NetworkX graph to PyTorch Geometric Data object
def nx_to_pyg(G, node_features=None):
    """
    Convert NetworkX graph to PyTorch Geometric Data object
    
    Parameters:
    -----------
    G: NetworkX Graph
        The graph to convert
    node_features: numpy.ndarray
        Node features (if None, uses one-hot encoded degrees)
        
    Returns:
    --------
    data: torch_geometric.data.Data
        PyTorch Geometric Data object
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric not available")
    
    # Get edges (convert to directed edges for PyG)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add reverse edges
    
    # If node features not provided, use degree as features
    if node_features is None:
        degrees = dict(G.degree())
        max_degree = max(degrees.values())
        x = torch.zeros((len(G), max_degree + 1), dtype=torch.float)
        for node_id, degree in degrees.items():
            x[node_id, degree] = 1.0
    else:
        x = torch.tensor(node_features, dtype=torch.float)
    
    # Get community labels if available
    if 'community' in G.nodes[0]:
        y = torch.tensor([G.nodes[i]['community'] for i in range(len(G))], dtype=torch.long)
    else:
        y = None
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


# 1. Graph Convolutional Network (GCN) for Node Embedding
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# 2. GraphSAGE for Node Embedding
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# 3. Graph Attention Network (GAT) for Node Embedding
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, heads=8):
        super(GAT, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
            
        self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False))
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# 4. Variational Graph Auto-Encoder (not full implementation - simplified for example)
class VGAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VGAE, self).__init__()
        
        # Encoder
        self.encoder_conv1 = GCNConv(input_dim, hidden_dim)
        self.encoder_conv_mu = GCNConv(hidden_dim, latent_dim)
        self.encoder_conv_logstd = GCNConv(hidden_dim, latent_dim)
        
        # Decoder (inner product between node embeddings)
        
    def encode(self, x, edge_index):
        hidden = F.relu(self.encoder_conv1(x, edge_index))
        mu = self.encoder_conv_mu(hidden, edge_index)
        logstd = self.encoder_conv_logstd(hidden, edge_index)
        return mu, logstd
        
    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def forward(self, x, edge_index):
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        return z, mu, logstd


# Function to train a GNN embedding model
def train_gnn_embedding(model, data, epochs=100, lr=0.01, weight_decay=5e-4, verbose=False):
    """
    Train a GNN model for node embedding
    
    Parameters:
    -----------
    model: torch.nn.Module
        GNN model for node embedding
    data: torch_geometric.data.Data
        PyTorch Geometric data object
    epochs: int
        Number of training epochs
    lr: float
        Learning rate
    weight_decay: float
        Weight decay for regularization
    verbose: bool
        Whether to print training progress
        
    Returns:
    --------
    model: torch.nn.Module
        Trained GNN model
    """
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, VGAE):
            z, mu, logstd = model(data.x, data.edge_index)
            # For VGAE, the loss would be a combination of reconstruction and KL divergence
            # This is simplified for the example
            loss = torch.tensor(0.0)
        else:
            z = model(data.x, data.edge_index)
            
            # If we have labels (supervised setting)
            if data.y is not None:
                loss = F.cross_entropy(z, data.y)
            else:
                # For unsupervised setting, we could use graph InfoMax or other losses
                # Simplified here
                loss = torch.tensor(0.0)
        
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model


# Function to extract node embeddings from trained GNN model
def extract_embeddings(model, data):
    """
    Extract node embeddings from a trained GNN model
    
    Parameters:
    -----------
    model: torch.nn.Module
        Trained GNN model
    data: torch_geometric.data.Data
        PyTorch Geometric data object
        
    Returns:
    --------
    embeddings: numpy.ndarray
        Node embeddings
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, VGAE):
            z, _, _ = model(data.x, data.edge_index)
        else:
            z = model(data.x, data.edge_index)
    
    return z.detach().cpu().numpy()


# Function to perform community detection on node embeddings
def detect_communities_from_embeddings(embeddings, n_clusters=None, method='kmeans'):
    """
    Detect communities from node embeddings
    
    Parameters:
    -----------
    embeddings: numpy.ndarray
        Node embeddings
    n_clusters: int
        Number of clusters (communities)
    method: str
        Clustering method ('kmeans' or 'spectral')
        
    Returns:
    --------
    communities: numpy.ndarray
        Community assignments for each node
    """
    if method.lower() == 'kmeans':
        if n_clusters is None:
            # If n_clusters not provided, use silhouette score to find optimal k
            from sklearn.metrics import silhouette_score
            best_score = -1
            best_k = 2  # Start with 2 clusters
            
            # Try different values of k
            for k in range(2, min(10, embeddings.shape[0] // 5 + 1)):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                if len(set(labels)) <= 1:  # Skip if only one cluster found
                    continue
                    
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            
            n_clusters = best_k
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        communities = kmeans.fit_predict(embeddings)
    
    elif method.lower() == 'spectral':
        from sklearn.cluster import SpectralClustering
        if n_clusters is None:
            n_clusters = 5  # Default to 5 clusters if not specified
            
        spectral = SpectralClustering(n_clusters=n_clusters, 
                                      random_state=42, 
                                      affinity='nearest_neighbors')
        communities = spectral.fit_predict(embeddings)
    
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    return communities


# Function to visualize node embeddings colored by communities
def plot_embeddings(embeddings, communities, method='tsne', figsize=(10, 8)):
    """
    Visualize node embeddings colored by community assignment
    
    Parameters:
    -----------
    embeddings: numpy.ndarray
        Node embeddings
    communities: numpy.ndarray
        Community assignments
    method: str
        Dimensionality reduction method ('tsne' or 'pca')
    figsize: tuple
        Figure size
        
    Returns:
    --------
    None
    """
    if method.lower() == 'tsne':
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
    elif method.lower() == 'pca':
        # Use PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        raise ValueError(f"Unsupported visualization method: {method}")
    
    plt.figure(figsize=figsize)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=communities, cmap='rainbow', s=50, alpha=0.8)
    plt.colorbar(label='Community')
    plt.title(f'Node Embeddings ({method.upper()}) colored by Community')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.show()


# Function to evaluate communities against ground truth
def evaluate_communities(communities, ground_truth):
    """
    Evaluate detected communities against ground truth
    
    Parameters:
    -----------
    communities: numpy.ndarray
        Detected community assignments
    ground_truth: numpy.ndarray
        Ground truth community assignments
        
    Returns:
    --------
    metrics: dict
        Evaluation metrics
    """
    metrics = {}
    metrics['nmi'] = normalized_mutual_info_score(ground_truth, communities)
    metrics['ari'] = adjusted_rand_score(ground_truth, communities)
    
    return metrics


# Function to convert community assignments to a NetworkX node attribute
def add_communities_to_graph(G, communities):
    """
    Add community assignments as a node attribute to NetworkX graph
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to add communities to
    communities: numpy.ndarray or dict
        Community assignments
        
    Returns:
    --------
    G: NetworkX Graph
        Graph with community attributes
    """
    if isinstance(communities, np.ndarray):
        # Convert numpy array to dict
        communities_dict = {i: communities[i] for i in range(len(communities))}
    else:
        communities_dict = communities
    
    # Add communities as node attribute
    nx.set_node_attributes(G, communities_dict, 'detected_community')
    
    return G


# Main function to run GNN-based community detection
def run_gnn_community_detection(G, model_type='gcn', embedding_dim=16, n_clusters=None, 
                                epochs=100, ground_truth_attr='community'):
    """
    Run GNN-based community detection
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to analyze
    model_type: str
        GNN model type ('gcn', 'graphsage', 'gat', 'vgae')
    embedding_dim: int
        Dimension of node embeddings
    n_clusters: int
        Number of communities to detect
    epochs: int
        Number of training epochs
    ground_truth_attr: str
        Node attribute for ground truth communities
        
    Returns:
    --------
    results: dict
        Results including embeddings, communities, and evaluation metrics
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric not available")
    
    # Convert to PyTorch Geometric Data
    data = nx_to_pyg(G)
    
    # If n_clusters not specified, try to infer from ground truth
    if n_clusters is None and ground_truth_attr in G.nodes[0]:
        communities = [G.nodes[i][ground_truth_attr] for i in range(len(G))]
        n_clusters = len(set(communities))
    
    # Select model type
    if model_type.lower() == 'gcn':
        model = GCN(data.x.size(1), hidden_dim=32, output_dim=embedding_dim)
    elif model_type.lower() == 'graphsage':
        model = GraphSAGE(data.x.size(1), hidden_dim=32, output_dim=embedding_dim)
    elif model_type.lower() == 'gat':
        model = GAT(data.x.size(1), hidden_dim=32, output_dim=embedding_dim)
    elif model_type.lower() == 'vgae':
        model = VGAE(data.x.size(1), hidden_dim=32, latent_dim=embedding_dim)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    print(f"Training {model_type.upper()} model...")
    start_time = time.time()
    model = train_gnn_embedding(model, data, epochs=epochs, verbose=True)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Extract embeddings
    print("Extracting node embeddings...")
    embeddings = extract_embeddings(model, data)
    
    # Detect communities
    print(f"Detecting communities with KMeans (n_clusters={n_clusters})...")
    communities = detect_communities_from_embeddings(embeddings, n_clusters=n_clusters)
    
    # Visualize embeddings
    print("Visualizing node embeddings...")
    plot_embeddings(embeddings, communities)
    
    # Add communities to graph
    G = add_communities_to_graph(G, communities)
    
    # Visualize communities in the graph
    print("Visualizing detected communities...")
    plot_graph(G, community_attr='detected_community', title=f"{model_type.upper()} Detected Communities")
    
    # Evaluate against ground truth if available
    results = {
        'model_type': model_type,
        'embeddings': embeddings,
        'communities': communities,
        'training_time': training_time
    }
    
    if ground_truth_attr in G.nodes[0]:
        ground_truth = np.array([G.nodes[i][ground_truth_attr] for i in range(len(G))])
        metrics = evaluate_communities(communities, ground_truth)
        results['metrics'] = metrics
        
        print("\nEvaluation against ground truth:")
        print(f"NMI: {metrics['nmi']:.4f}")
        print(f"ARI: {metrics['ari']:.4f}")
    
    return results


# Function to compare different GNN models
def compare_gnn_models(G, embedding_dim=16, n_clusters=None, epochs=100, ground_truth_attr='community'):
    """
    Compare different GNN models for community detection
    
    Parameters:
    -----------
    G: NetworkX Graph
        Graph to analyze
    embedding_dim: int
        Dimension of node embeddings
    n_clusters: int
        Number of communities to detect
    epochs: int
        Number of training epochs
    ground_truth_attr: str
        Node attribute for ground truth communities
        
    Returns:
    --------
    results_df: DataFrame
        DataFrame with comparison results
    """
    model_types = ['gcn', 'graphsage', 'gat', 'vgae']
    results_list = []
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Running {model_type.upper()}")
        print(f"{'='*50}")
        
        try:
            results = run_gnn_community_detection(G, model_type=model_type, 
                                                 embedding_dim=embedding_dim,
                                                 n_clusters=n_clusters, 
                                                 epochs=epochs,
                                                 ground_truth_attr=ground_truth_attr)
            
            # Collect results
            result_entry = {
                'Model': model_type.upper(),
                'Training Time (s)': results['training_time'],
                'Num Communities': len(set(results['communities']))
            }
            
            if 'metrics' in results:
                result_entry['NMI'] = results['metrics']['nmi']
                result_entry['ARI'] = results['metrics']['ari']
            
            results_list.append(result_entry)
            
        except Exception as e:
            print(f"Error running {model_type}: {e}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Visualize comparison
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    if 'NMI' in results_df.columns:
        plt.subplot(2, 1, 1)
        results_df.plot(x='Model', y=['NMI', 'ARI'], kind='bar', ax=plt.gca())
        plt.title('Community Detection Quality')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
    
    # Plot training time
    plt.subplot(2, 1, 2)
    results_df.plot(x='Model', y='Training Time (s)', kind='bar', ax=plt.gca(), color='green')
    plt.title('Training Time')
    plt.ylabel('Seconds')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df


# Main execution example
if __name__ == "__main__":
    print("Graph Neural Network-Based Community Detection")
    print("="*50)
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("Error: PyTorch Geometric not available. Please install it first.")
        import sys
        sys.exit(1)
    
    # Generate a synthetic graph with ground truth communities
    print("\n1. Generating a Stochastic Block Model graph...")
    n_communities = 5
    G, ground_truth = generate_synthetic_graph('sbm', n_nodes=100, n_communities=n_communities,
                                              p_in=0.3, p_out=0.05)
    
    # Visualize original graph with ground truth
    print("\n2. Visualizing ground truth communities...")
    plot_graph(G, community_attr='community', title="Ground Truth Communities")
    
    # Run GNN-based community detection with a specific model
    print("\n3. Running GCN-based community detection...")
    results_gcn = run_gnn_community_detection(G, model_type='gcn', embedding_dim=16, 
                                            n_clusters=n_communities, epochs=100)
    
    # Compare different GNN models
    print("\n4. Comparing different GNN models...")
    comparison_results = compare_gnn_models(G, embedding_dim=16, n_clusters=n_communities, epochs=100)
    
    print("\n5. Summary of results:")
    print(comparison_results)
    
    print("\nAnalysis complete!")
