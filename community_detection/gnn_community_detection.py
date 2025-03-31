# Graph Neural Network-Based Community Detection
# =============================================

import os
import torch
import polars as pl
import rustworkx as rx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
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

# For PyTorch and PyTorch Geometric
try:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Some methods will be skipped.")

try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv, DeepGraphInfomax
    from torch_geometric.data import Data
    from torch_geometric.utils import to_networkx, from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available. GNN methods will be skipped.")

# For community evaluation
try:
    from cdlib import evaluation
    from cdlib.classes import NodeClustering
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False


# Function to convert RustWorkX graph to PyTorch Geometric Data object
def rwx_to_pyg(G: rx.PyGraph, node_features: Optional[torch.Tensor] = None) -> Data:
    """
    Convert RustWorkX graph to PyTorch Geometric Data object
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        The graph to convert
    node_features: torch.Tensor
        Node features (if None, uses one-hot encoded degrees)
        
    Returns:
    --------
    data: torch_geometric.data.Data
        PyTorch Geometric Data object
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric not available")
    
    # Get edges (convert to directed edges for PyG)
    edge_list = []
    for edge in G.edge_list():
        edge_list.append((edge[0], edge[1]))
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # For undirected graph in PyG, we need both directions
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # If node features not provided, use degree as features
    if node_features is None:
        degrees = torch.tensor([G.degree(i) for i in range(len(G))])
        max_degree = degrees.max().item()
        x = torch.zeros((len(G), max_degree + 1), dtype=torch.float)
        for i, degree in enumerate(degrees):
            x[i, degree] = 1.0
    else:
        x = node_features
    
    # Get community labels if available
    y = None
    for i in range(len(G)):
        node_data = G.get_node_data(i)
        if node_data and 'community' in node_data:
            if y is None:
                y = torch.zeros(len(G), dtype=torch.long)
            y[i] = node_data['community']
    
    # Extract edge weights if available
    edge_attr = None
    if G.num_edges() > 0:
        edge_weights = []
        for edge in G.edge_list():
            source, target = edge[0], edge[1]
            edge_data = G.get_edge_data(source, target)
            if edge_data and 'weight' in edge_data:
                edge_weights.append(edge_data['weight'])
                edge_weights.append(edge_data['weight'])  # Add twice for both directions
            else:
                edge_weights.append(1.0)  # Default weight
                edge_weights.append(1.0)  # Add twice for both directions
                
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data


# 1. Graph Convolutional Network (GCN) for Node Embedding
class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                num_layers: int = 2, dropout: float = 0.5):
        super(GCN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        for i, conv in enumerate(self.convs[:-1]):
            if edge_weight is not None:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if edge_weight is not None:
            x = self.convs[-1](x, edge_index, edge_weight)
        else:
            x = self.convs[-1](x, edge_index)
        
        return x


# 2. GraphSAGE for Node Embedding
class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                num_layers: int = 2, dropout: float = 0.5):
        super(GraphSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# 3. Graph Attention Network (GAT) for Node Embedding
class GAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                num_layers: int = 2, dropout: float = 0.5, heads: int = 8):
        super(GAT, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
            
        self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False))
        
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# 4. Variational Graph Auto-Encoder (simplified)
class VGAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(VGAE, self).__init__()
        
        # Encoder
        self.encoder_conv1 = GCNConv(input_dim, hidden_dim)
        self.encoder_conv_mu = GCNConv(hidden_dim, latent_dim)
        self.encoder_conv_logstd = GCNConv(hidden_dim, latent_dim)
        
        # Decoder (inner product between node embeddings)
        
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        hidden = F.relu(self.encoder_conv1(x, edge_index))
        mu = self.encoder_conv_mu(hidden, edge_index)
        logstd = self.encoder_conv_logstd(hidden, edge_index)
        return mu, logstd
        
    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        return z, mu, logstd


# Function to train a GNN embedding model
def train_gnn_embedding(model: nn.Module, data: Data, epochs: int = 100, 
                      lr: float = 0.01, weight_decay: float = 5e-4, 
                      verbose: bool = False) -> nn.Module:
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
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, VGAE):
            z, mu, logstd = model(data.x, data.edge_index)
            # For VGAE, use a reconstruction loss
            adj_orig = torch.sparse.FloatTensor(
                data.edge_index, torch.ones(data.edge_index.size(1)).to(device),
                torch.Size([data.num_nodes, data.num_nodes])
            ).to_dense()
            
            # Reconstruction loss
            adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
            pos_weight = float((data.num_nodes * data.num_nodes - data.edge_index.size(1)) / data.edge_index.size(1))
            norm = data.num_nodes * data.num_nodes / float((data.num_nodes * data.num_nodes - data.edge_index.size(1)) * 2)
            
            pos_loss = -torch.mean(torch.log(torch.clamp(adj_pred, min=1e-8)) * adj_orig)
            neg_loss = -torch.mean(torch.log(torch.clamp(1 - adj_pred, min=1e-8)) * (1 - adj_orig))
            rec_loss = pos_loss + neg_loss
            
            # KL divergence
            kl_loss = 0.5 / data.num_nodes * torch.mean(
                torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), 1))
            
            loss = rec_loss - kl_loss
        else:
            z = model(data.x, data.edge_index)
            
            # If we have labels (supervised setting)
            if data.y is not None:
                loss = F.cross_entropy(z, data.y)
            else:
# For unsupervised setting, use graph InfoMax or other losses
                # Here, we'll use a simple contrastive loss
                # Create negative samples by permuting nodes
                neg_edge_index = data.edge_index[:, torch.randperm(data.edge_index.size(1))]
                
                # Positive loss (real edges should have high similarity)
                pos_z = z[data.edge_index[0]] * z[data.edge_index[1]]
                pos_loss = -torch.log(torch.sigmoid(pos_z.sum(dim=1))).mean()
                
                # Negative loss (fake edges should have low similarity)
                neg_z = z[neg_edge_index[0]] * z[neg_edge_index[1]]
                neg_loss = -torch.log(1 - torch.sigmoid(neg_z.sum(dim=1))).mean()
                
                loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model


# Function to extract node embeddings from trained GNN model
def extract_embeddings(model: nn.Module, data: Data) -> torch.Tensor:
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
    embeddings: torch.Tensor
        Node embeddings
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, VGAE):
            z, _, _ = model(data.x, data.edge_index)
        else:
            z = model(data.x, data.edge_index)
    
    return z.cpu()


# Function to perform community detection on node embeddings
def detect_communities_from_embeddings(embeddings: torch.Tensor, 
                                     n_clusters: Optional[int] = None, 
                                     method: str = 'kmeans') -> torch.Tensor:
    """
    Detect communities from node embeddings
    
    Parameters:
    -----------
    embeddings: torch.Tensor
        Node embeddings
    n_clusters: int
        Number of clusters (communities)
    method: str
        Clustering method ('kmeans' or 'spectral')
        
    Returns:
    --------
    communities: torch.Tensor
        Community assignments for each node
    """
    # Convert to numpy for scikit-learn
    embeddings_np = embeddings.numpy()
    
    if method.lower() == 'kmeans':
        if n_clusters is None:
            # If n_clusters not provided, use silhouette score to find optimal k
            from sklearn.metrics import silhouette_score
            best_score = -1
            best_k = 2  # Start with 2 clusters
            
            # Try different values of k
            for k in range(2, min(10, embeddings_np.shape[0] // 5 + 1)):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings_np)
                    
                    if len(set(labels)) <= 1:  # Skip if only one cluster found
                        continue
                        
                    score = silhouette_score(embeddings_np, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                except:
                    continue
            
            n_clusters = best_k
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        communities = kmeans.fit_predict(embeddings_np)
    
    elif method.lower() == 'spectral':
        from sklearn.cluster import SpectralClustering
        if n_clusters is None:
            n_clusters = 5  # Default to 5 clusters if not specified
            
        spectral = SpectralClustering(n_clusters=n_clusters, 
                                      random_state=42, 
                                      affinity='nearest_neighbors')
        communities = spectral.fit_predict(embeddings_np)
    
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    return torch.tensor(communities, dtype=torch.long)


# Function to visualize node embeddings colored by communities
def plot_embeddings(embeddings: torch.Tensor, communities: torch.Tensor, 
                  method: str = 'tsne', figsize: Tuple[int, int] = (10, 8), 
                  title: str = "Node Embeddings"):
    """
    Visualize node embeddings colored by community assignment
    
    Parameters:
    -----------
    embeddings: torch.Tensor
        Node embeddings
    communities: torch.Tensor
        Community assignments
    method: str
        Dimensionality reduction method ('tsne', 'pca', or 'none' if already 2D)
    figsize: tuple
        Figure size
    title: str
        Plot title
        
    Returns:
    --------
    None
    """
    # Convert to numpy
    embeddings_np = embeddings.numpy()
    communities_np = communities.numpy()
    
    if method.lower() == 'tsne':
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)-1))
        embeddings_2d = tsne.fit_transform(embeddings_np)
    elif method.lower() == 'pca':
        # Use PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings_np)
    elif method.lower() == 'none':
        # Assume embeddings are already 2D
        if embeddings_np.shape[1] != 2:
            raise ValueError(f"When method='none', embeddings must be 2D, but got {embeddings_np.shape[1]}D")
        embeddings_2d = embeddings_np
    else:
        raise ValueError(f"Unsupported visualization method: {method}. Use 'tsne', 'pca', or 'none'.")
    
    plt.figure(figsize=figsize)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=communities_np, cmap='rainbow', s=50, alpha=0.8)
    plt.colorbar(label='Community')
    
    # Don't add method name to title if method is 'none'
    if method.lower() == 'none':
        plt.title(title)
    else:
        plt.title(f'{title} ({method.upper()})')
        
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.show()


# Function to evaluate communities against ground truth
def evaluate_communities(communities: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate detected communities against ground truth
    
    Parameters:
    -----------
    communities: torch.Tensor
        Detected community assignments
    ground_truth: torch.Tensor
        Ground truth community assignments
        
    Returns:
    --------
    metrics: dict
        Evaluation metrics
    """
    # Convert to numpy for sklearn
    communities_np = communities.numpy()
    ground_truth_np = ground_truth.numpy()
    
    metrics = {}
    metrics['nmi'] = normalized_mutual_info_score(ground_truth_np, communities_np)
    metrics['ari'] = adjusted_rand_score(ground_truth_np, communities_np)
    
    return metrics


# Function to convert community assignments to a RustWorkX node attribute
def add_communities_to_graph(G: rx.PyGraph, communities: torch.Tensor, 
                           attr_name: str = 'detected_community') -> rx.PyGraph:
    """
    Add community assignments as a node attribute to RustWorkX graph
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
        Graph to add communities to
    communities: torch.Tensor
        Community assignments
    attr_name: str
        Name of the node attribute to add
        
    Returns:
    --------
    G: RustWorkX PyGraph
        Graph with community attributes
    """
    communities_np = communities.numpy()
    
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
        
        # Add community info if this node is in communities_np (check index bounds)
        if i < len(communities_np):
            node_data[attr_name] = int(communities_np[i])
        
        # Add node with updated data
        node_mapping[i] = new_G.add_node(node_data)
    
    # Add edges
    for edge in G.edge_list():
        source, target = edge[0], edge[1]
        edge_data = G.get_edge_data(source, target)
        new_G.add_edge(node_mapping[source], node_mapping[target], edge_data)
    
    return new_G


# Main function to run GNN-based community detection
def run_gnn_community_detection(G: rx.PyGraph, model_type: str = 'gcn', 
                             embedding_dim: int = 16, n_clusters: Optional[int] = None, 
                             epochs: int = 100, ground_truth_attr: str = 'community') -> Dict[str, Any]:
    """
    Run GNN-based community detection
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
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
    data = rwx_to_pyg(G)
    
    # If n_clusters not specified, try to infer from ground truth
    if n_clusters is None:
        has_communities = False
        for i in range(len(G)):
            node_data = G.get_node_data(i)
            if node_data and ground_truth_attr in node_data:
                has_communities = True
                break
                
        if has_communities:
            # Extract communities from graph
            communities = []
            for i in range(len(G)):
                node_data = G.get_node_data(i)
                if node_data and ground_truth_attr in node_data:
                    communities.append(node_data[ground_truth_attr])
                else:
                    communities.append(0)  # Default
            
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
    plot_embeddings(embeddings, communities, title=f"{model_type.upper()} Node Embeddings")
    
    # Add communities to graph
    G = add_communities_to_graph(G, communities, attr_name='detected_community')
    
    # Evaluate against ground truth if available
    results = {
        'model_type': model_type,
        'embeddings': embeddings,
        'communities': communities,
        'training_time': training_time
    }
    
    has_ground_truth = False
    for i in range(len(G)):
        node_data = G.get_node_data(i)
        if node_data and ground_truth_attr in node_data:
            has_ground_truth = True
            break
            
    if has_ground_truth:
        # Extract ground truth from graph
        ground_truth = []
        for i in range(len(G)):
            node_data = G.get_node_data(i)
            if node_data and ground_truth_attr in node_data:
                ground_truth.append(node_data[ground_truth_attr])
            else:
                ground_truth.append(0)  # Default
                
        ground_truth = torch.tensor(ground_truth, dtype=torch.long)
        metrics = evaluate_communities(communities, ground_truth)
        results['metrics'] = metrics
        
        print("\nEvaluation against ground truth:")
        print(f"NMI: {metrics['nmi']:.4f}")
        print(f"ARI: {metrics['ari']:.4f}")
    
    return results


# Function to compare different GNN models
def compare_gnn_models(G: rx.PyGraph, embedding_dim: int = 16, n_clusters: Optional[int] = None, 
                     epochs: int = 100, ground_truth_attr: str = 'community') -> pl.DataFrame:
    """
    Compare different GNN models for community detection
    
    Parameters:
    -----------
    G: RustWorkX PyGraph
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
                'Num Communities': len(torch.unique(results['communities']))
            }
            
            if 'metrics' in results:
                result_entry['NMI'] = results['metrics']['nmi']
                result_entry['ARI'] = results['metrics']['ari']
            
            results_list.append(result_entry)
            
        except Exception as e:
            print(f"Error running {model_type}: {e}")
    
    # Create DataFrame
    results_df = pl.DataFrame(results_list)
    
    # Visualize comparison
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    if 'NMI' in results_df.columns:
        plt.subplot(2, 1, 1)
        
        # Convert to pandas for plotting
        pd_df = results_df.to_pandas()
        pd_df.plot(x='Model', y=['NMI', 'ARI'], kind='bar', ax=plt.gca())
        plt.title('Community Detection Quality')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
    
    # Plot training time
    plt.subplot(2, 1, 2)
    pd_df = results_df.to_pandas()
    pd_df.plot(x='Model', y='Training Time (s)', kind='bar', ax=plt.gca(), color='green')
    plt.title('Training Time')
    plt.ylabel('Seconds')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df
                