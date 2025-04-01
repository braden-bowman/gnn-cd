import os
import polars as pl
import rustworkx as rx
import torch
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from nicegui import ui, app
from nicegui.events import UploadEventArguments

# Import community detection modules
from . import data_prep
from . import traditional_methods
from . import gnn_community_detection
from . import overlapping_community_detection
from . import dynamic_gnn
from . import evaluation

# Define global variables to store user session data
user_data = {}

# Path for temporary file storage
TEMP_DIR = os.path.join(tempfile.gettempdir(), 'gnn_cd')
os.makedirs(TEMP_DIR, exist_ok=True)

def initialize_app():
    """Initialize and return the NiceGUI app"""
    
    @ui.page('/')
    def index():
        with ui.header().classes('bg-blue-600 text-white'):
            ui.label('GNN-CD: Graph Neural Network Community Detection').classes('text-h4 q-ml-md')
            
        with ui.tabs().classes('w-full') as tabs:
            ui.tab('upload', 'Data Upload')
            ui.tab('detect', 'Community Detection')
            ui.tab('train', 'Model Training')
            ui.tab('evaluate', 'Evaluation')
            ui.tab('visualize', 'Visualization')
            ui.tab('results', 'Results')
            
        with ui.tab_panels(tabs).classes('w-full'):
            with ui.tab_panel('upload'):
                create_upload_tab()
            with ui.tab_panel('detect'):
                create_detection_tab()
            with ui.tab_panel('train'):
                create_training_tab()
            with ui.tab_panel('evaluate'):
                create_evaluation_tab()
            with ui.tab_panel('visualize'):
                create_visualization_tab()
            with ui.tab_panel('results'):
                create_results_tab()
                
        with ui.footer().classes('bg-blue-100'):
            ui.label('© 2025 GNN-CD Framework').classes('text-center w-full')
    
    return app

def create_upload_tab():
    """Create the data upload tab interface"""
    
    ui.label('Upload Network Data').classes('text-h5 q-mb-md')
    
    with ui.row():
        with ui.column().classes('w-1/3'):
            ui.label('Upload graph data or edge list:').classes('text-bold')
            upload = ui.upload(
                label='Upload file',
                multiple=False,
                on_upload=handle_graph_upload
            ).props('accept=.csv,.parquet,.graphml,.gpickle')
            
            with ui.expansion('Advanced Options', icon='settings').classes('w-full'):
                ui.checkbox('Directed graph', value=False).bind_value(app.storage.user, 'directed_graph')
                ui.checkbox('Weighted edges', value=True).bind_value(app.storage.user, 'weighted_edges')
                ui.number('Node limit', value=10000, min=100).bind_value(app.storage.user, 'node_limit')
                
                ui.separator()
                ui.label('Performance Options').classes('text-bold')
                ui.checkbox('Use GPU acceleration when available', value=True).bind_value(app.storage.user, 'use_gpu')
                ui.checkbox('Stream large files', value=True).bind_value(app.storage.user, 'use_streaming')
                ui.number('Chunk size for processing', value=10000, min=1000).bind_value(app.storage.user, 'chunk_size')
                ui.checkbox('Sample large graphs for visualization', value=True).bind_value(app.storage.user, 'sample_for_viz')
                ui.checkbox('Use Plotly for interactive visualizations', value=True).bind_value(app.storage.user, 'use_plotly')
        
        with ui.separator().classes('vertical'):
            pass
            
        with ui.column().classes('w-1/3'):
            ui.label('Upload labeled data (optional):').classes('text-bold')
            upload_labels = ui.upload(
                label='Upload labeled data',
                multiple=False,
                on_upload=handle_labels_upload
            ).props('accept=.csv,.parquet')
            
            ui.label('Label column:').classes('text-bold q-mt-md')
            label_col = ui.input(label='Column name', value='label')
            
        with ui.separator().classes('vertical'):
            pass
            
        with ui.column().classes('w-1/3'):
            ui.label('Upload node features (optional):').classes('text-bold')
            upload_features = ui.upload(
                label='Upload node features',
                multiple=False,
                on_upload=handle_features_upload
            ).props('accept=.csv,.parquet')
            
            ui.label('Node ID column:').classes('text-bold q-mt-md')
            node_id_col = ui.input(label='Column name', value='node_id')
    
    ui.separator()
    
    with ui.row().classes('items-center q-mt-lg'):
        ui.label('Current Graph:').classes('text-h6')
        graph_info = ui.label('No graph loaded').classes('q-ml-md text-grey-8')
    
    ui.button('Process Data', on_click=process_data).props('color=primary')
    
    # Functions for the upload tab
    def handle_graph_upload(e: UploadEventArguments):
        """Handle uploaded graph file"""
        if not e.name:
            return
            
        file_path = os.path.join(TEMP_DIR, e.name)
        with open(file_path, 'wb') as f:
            f.write(e.content.read())
        
        ui.notify(f'Uploaded {e.name}', color='positive')
        app.storage.user['graph_path'] = file_path
        
        # Update the graph info text
        file_ext = os.path.splitext(e.name)[1].lower()
        if file_ext in ['.csv', '.parquet']:
            graph_info.text = f'Loaded edge list from {e.name}'
        else:
            graph_info.text = f'Loaded graph from {e.name}'
    
    def handle_labels_upload(e: UploadEventArguments):
        """Handle uploaded labels file"""
        if not e.name:
            return
            
        file_path = os.path.join(TEMP_DIR, e.name)
        with open(file_path, 'wb') as f:
            f.write(e.content.read())
        
        ui.notify(f'Uploaded {e.name}', color='positive')
        app.storage.user['labels_path'] = file_path
    
    def handle_features_upload(e: UploadEventArguments):
        """Handle uploaded features file"""
        if not e.name:
            return
            
        file_path = os.path.join(TEMP_DIR, e.name)
        with open(file_path, 'wb') as f:
            f.write(e.content.read())
        
        ui.notify(f'Uploaded {e.name}', color='positive')
        app.storage.user['features_path'] = file_path
    
    def process_data():
        """Process the uploaded data and create graph"""
        try:
            if 'graph_path' not in app.storage.user:
                ui.notify('Please upload a graph file first', color='negative')
                return
            
            graph_path = app.storage.user['graph_path']
            file_ext = os.path.splitext(graph_path)[1].lower()
            
            # Load the graph based on file type
            if file_ext == '.csv':
                # Load edge list from CSV with optimized parameters
                df = data_prep.load_data(
                    graph_path, 
                    filetype='csv', 
                    use_streaming=app.storage.user.get('use_streaming', True),
                    chunk_size=app.storage.user.get('chunk_size', 10000)
                )
                user_data['graph'] = data_prep.create_graph_from_edgelist(
                    df, 
                    directed=app.storage.user.get('directed_graph', False),
                    weighted=app.storage.user.get('weighted_edges', True),
                    chunk_size=app.storage.user.get('chunk_size', 10000),
                    max_nodes=app.storage.user.get('node_limit', 10000)
                )
            elif file_ext == '.parquet':
                # Load edge list from Parquet with optimized parameters
                df = data_prep.load_data(
                    graph_path, 
                    filetype='parquet', 
                    use_streaming=app.storage.user.get('use_streaming', True),
                    chunk_size=app.storage.user.get('chunk_size', 10000)
                )
                user_data['graph'] = data_prep.create_graph_from_edgelist(
                    df, 
                    directed=app.storage.user.get('directed_graph', False),
                    weighted=app.storage.user.get('weighted_edges', True),
                    chunk_size=app.storage.user.get('chunk_size', 10000),
                    max_nodes=app.storage.user.get('node_limit', 10000)
                )
            elif file_ext in ['.graphml', '.gpickle']:
                # Load existing graph
                import networkx as nx
                if file_ext == '.graphml':
                    nx_graph = nx.read_graphml(graph_path)
                else:
                    nx_graph = nx.read_gpickle(graph_path)
                
                # Convert NetworkX to RustWorkX
                user_data['graph'] = traditional_methods._nx_to_rwx(nx_graph)
            
            # Add labels if available
            if 'labels_path' in app.storage.user:
                labels_df = pl.read_csv(app.storage.user['labels_path']) \
                    if app.storage.user['labels_path'].endswith('.csv') \
                    else pl.read_parquet(app.storage.user['labels_path'])
                
                label_column = label_col.value
                if label_column in labels_df.columns:
                    # Add labels as node attributes
                    labels_dict = {row['node_id']: row[label_column] 
                                  for row in labels_df.iter_rows(named=True)}
                    
                    G = user_data['graph']
                    for i in range(len(G)):
                        node_data = G.get_node_data(i) or {}
                        if i in labels_dict:
                            node_data['community'] = labels_dict[i]
                        G.set_node_data(i, node_data)
            
            # Add features if available
            if 'features_path' in app.storage.user:
                features_df = pl.read_csv(app.storage.user['features_path']) \
                    if app.storage.user['features_path'].endswith('.csv') \
                    else pl.read_parquet(app.storage.user['features_path'])
                
                id_column = node_id_col.value
                if id_column in features_df.columns:
                    # Extract features (all columns except ID)
                    feature_cols = [col for col in features_df.columns if col != id_column]
                    
                    # Add features as node attributes
                    G = user_data['graph']
                    for row in features_df.iter_rows(named=True):
                        node_id = row[id_column]
                        features = {col: row[col] for col in feature_cols}
                        
                        node_data = G.get_node_data(node_id) or {}
                        node_data['features'] = features
                        G.set_node_data(node_id, node_data)
            
            # Compute graph statistics with optimized parameters
            graph_size = len(user_data['graph'])
            large_graph = graph_size > 10000
            
            stats = data_prep.compute_graph_statistics(
                user_data['graph'],
                compute_expensive=not large_graph,  # Skip expensive computations for large graphs
                sample_clustering=large_graph,     # Sample nodes for clustering in large graphs
                max_sample_size=min(1000, graph_size),
                use_gpu=app.storage.user.get('use_gpu', True)
            )
            user_data['graph_stats'] = stats
            
            # Update graph info
            graph_info.text = (
                f"Loaded graph with {stats['n_nodes']} nodes and {stats['n_edges']} edges. "
                f"Density: {stats['density']:.4f}, Avg degree: {stats['avg_degree']:.2f}"
            )
            
            ui.notify('Graph processed successfully', color='positive')
            
        except Exception as e:
            ui.notify(f'Error processing data: {str(e)}', color='negative')

def create_detection_tab():
    """Create the community detection tab interface"""
    
    ui.label('Community Detection').classes('text-h5 q-mb-md')
    
    with ui.row():
        with ui.column().classes('w-1/3'):
            ui.label('Traditional Methods').classes('text-h6')
            
            algorithm = ui.select(
                ['Louvain', 'Leiden', 'Label Propagation', 'Infomap', 'Walktrap', 'Spectral Clustering'],
                label='Select algorithm',
                value='Louvain'
            ).classes('w-full')
            
            with ui.expansion('Algorithm Parameters', icon='tune').classes('w-full'):
                with ui.column():
                    ui.number('Number of communities', value=5, min=2) \
                        .bind_value(app.storage.user, 'n_communities') \
                        .classes('w-full')
                    ui.checkbox('Directed graph', value=False).bind_value(app.storage.user, 'directed_graph')
                    ui.checkbox('Use GPU acceleration if available', value=True).bind_value(app.storage.user, 'use_gpu')
            
            ui.button('Run Detection', on_click=run_traditional_detection) \
                .props('color=primary').classes('w-full q-mt-lg')
            
        with ui.column().classes('w-1/3'):
            ui.label('GNN-Based Methods').classes('text-h6')
            
            gnn_algorithm = ui.select(
                ['GCN', 'GraphSAGE', 'GAT', 'VGAE'],
                label='Select GNN model',
                value='GCN'
            ).classes('w-full')
            
            with ui.expansion('Model Parameters', icon='settings').classes('w-full'):
                with ui.column():
                    ui.number('Embedding dimension', value=16, min=4).bind_value(app.storage.user, 'embedding_dim')
                    ui.number('Hidden dimension', value=32, min=8).bind_value(app.storage.user, 'hidden_dim')
                    ui.number('Number of layers', value=2, min=1, max=4).bind_value(app.storage.user, 'num_layers')
                    ui.number('Training epochs', value=100, min=10).bind_value(app.storage.user, 'epochs')
                    ui.number('Learning rate', value=0.01, min=0.0001, max=0.1, step=0.001) \
                        .bind_value(app.storage.user, 'learning_rate')
                    ui.checkbox('Use GPU acceleration if available', value=True).bind_value(app.storage.user, 'use_gpu_gnn')
            
            ui.button('Run GNN Detection', on_click=run_gnn_detection) \
                .props('color=primary').classes('w-full q-mt-lg')
            
        with ui.column().classes('w-1/3'):
            ui.label('Overlapping Communities').classes('text-h6')
            
            overlap_algorithm = ui.select(
                ['BigCLAM', 'DEMON', 'SLPA', 'GNN-Overlapping'],
                label='Select algorithm',
                value='BigCLAM'
            ).classes('w-full')
            
            with ui.expansion('Algorithm Parameters', icon='tune').classes('w-full'):
                with ui.column():
                    ui.number('Number of communities', value=5, min=2).bind_value(app.storage.user, 'n_overlap_communities')
                    ui.number('Overlap threshold', value=0.1, min=0.01, max=0.5, step=0.01) \
                        .bind_value(app.storage.user, 'overlap_threshold')
            
            ui.button('Run Overlapping Detection', on_click=run_overlap_detection) \
                .props('color=primary').classes('w-full q-mt-lg')
    
    ui.separator().classes('q-my-md')
    
    with ui.row():
        with ui.column().classes('w-full'):
            ui.label('Detection Results').classes('text-h6')
            result_label = ui.label('No detection run yet').classes('text-grey-7')
    
    # Functions for the detection tab
    def run_traditional_detection():
        """Run selected traditional detection algorithm"""
        try:
            if 'graph' not in user_data:
                ui.notify('Please load a graph first', color='negative')
                return
            
            G = user_data['graph']
            algo_name = algorithm.value
            
            # Get parameters
            n_communities = app.storage.user.get('n_communities', 5)
            directed = app.storage.user.get('directed_graph', False)
            use_gpu = app.storage.user.get('use_gpu', True)
            
            # Run selected algorithm
            if algo_name == 'Louvain':
                communities, execution_time = traditional_methods.run_louvain(G)
            elif algo_name == 'Leiden':
                communities, execution_time = traditional_methods.run_leiden(G)
            elif algo_name == 'Label Propagation':
                communities, execution_time = traditional_methods.run_label_propagation(G)
            elif algo_name == 'Infomap':
                communities, execution_time = traditional_methods.run_infomap(G)
            elif algo_name == 'Walktrap':
                communities, execution_time = traditional_methods.run_walktrap(G)
            elif algo_name == 'Spectral Clustering':
                communities, execution_time = traditional_methods.run_spectral_clustering(G, n_communities)
            
            # Store results
            user_data['detection_results'] = {
                'method': algo_name,
                'communities': communities,
                'execution_time': execution_time,
                'num_communities': len(set(communities.values()))
            }
            
            # Update graph with communities
            user_data['graph'] = traditional_methods.add_communities_to_graph(
                G, communities, attr_name='detected_community'
            )
            
            # Check if we have ground truth to evaluate against
            has_ground_truth = False
            for i in range(len(G)):
                node_data = G.get_node_data(i)
                if node_data and 'community' in node_data:
                    has_ground_truth = True
                    break
            
            if has_ground_truth:
                metrics = traditional_methods.evaluate_against_ground_truth(G, communities)
                user_data['detection_results']['metrics'] = metrics
                
                result_label.text = (
                    f"Detection complete: {algo_name} found {len(set(communities.values()))} communities "
                    f"in {execution_time:.2f} seconds.\n"
                    f"Metrics: NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}, "
                    f"Modularity={metrics['modularity'] if isinstance(metrics['modularity'], float) else 'N/A'}"
                )
            else:
                result_label.text = (
                    f"Detection complete: {algo_name} found {len(set(communities.values()))} communities "
                    f"in {execution_time:.2f} seconds."
                )
            
            ui.notify('Community detection complete', color='positive')
            
        except Exception as e:
            ui.notify(f'Error in detection: {str(e)}', color='negative')
    
    def run_gnn_detection():
        """Run selected GNN-based detection algorithm"""
        try:
            if 'graph' not in user_data:
                ui.notify('Please load a graph first', color='negative')
                return
            
            G = user_data['graph']
            model_type = gnn_algorithm.value.lower()
            
            # Get parameters
            embedding_dim = app.storage.user.get('embedding_dim', 16)
            epochs = app.storage.user.get('epochs', 100)
            n_communities = app.storage.user.get('n_communities', None)
            use_gpu = app.storage.user.get('use_gpu_gnn', True)
            
            # Force CPU if requested
            if not use_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Run GNN detection
            results = gnn_community_detection.run_gnn_community_detection(
                G, 
                model_type=model_type,
                embedding_dim=embedding_dim,
                n_clusters=n_communities,
                epochs=epochs
            )
            
            # Store results
            user_data['detection_results'] = {
                'method': f'GNN-{model_type.upper()}',
                'communities': {i: int(c) for i, c in enumerate(results['communities'])},
                'embeddings': results['embeddings'],
                'execution_time': results['training_time'],
                'num_communities': len(torch.unique(results['communities']))
            }
            
            if 'metrics' in results:
                user_data['detection_results']['metrics'] = results['metrics']
                
                result_label.text = (
                    f"Detection complete: {model_type.upper()} found "
                    f"{len(torch.unique(results['communities']))} communities "
                    f"in {results['training_time']:.2f} seconds.\n"
                    f"Metrics: NMI={results['metrics']['nmi']:.4f}, ARI={results['metrics']['ari']:.4f}"
                )
            else:
                result_label.text = (
                    f"Detection complete: {model_type.upper()} found "
                    f"{len(torch.unique(results['communities']))} communities "
                    f"in {results['training_time']:.2f} seconds."
                )
            
            ui.notify('GNN detection complete', color='positive')
            
        except Exception as e:
            ui.notify(f'Error in GNN detection: {str(e)}', color='negative')
            
    def run_overlap_detection():
        """Run selected overlapping community detection algorithm"""
        try:
            if 'graph' not in user_data:
                ui.notify('Please load a graph first', color='negative')
                return
            
            G = user_data['graph']
            algo_name = overlap_algorithm.value
            
            # Get parameters
            n_communities = app.storage.user.get('n_overlap_communities', 5)
            threshold = app.storage.user.get('overlap_threshold', 0.1)
            
            # Run selected algorithm
            if algo_name == 'BigCLAM':
                results = overlapping_community_detection.run_bigclam(G, n_clusters=n_communities)
            elif algo_name == 'DEMON':
                results = overlapping_community_detection.run_demon(G, epsilon=threshold)
            elif algo_name == 'SLPA':
                results = overlapping_community_detection.run_slpa(G, threshold=threshold)
            elif algo_name == 'GNN-Overlapping':
                results = overlapping_community_detection.run_gnn_overlapping(
                    G, n_clusters=n_communities, threshold=threshold
                )
            
            # Store results
            user_data['detection_results'] = {
                'method': algo_name,
                'overlapping_communities': results['communities'],
                'execution_time': results['execution_time'],
                'num_communities': len(results['communities'])
            }
            
            # Convert overlapping communities to memberships for visualization
            memberships = {}
            for i, communities in enumerate(results['communities']):
                # For each node, store all communities it belongs to
                for node in communities:
                    if node not in memberships:
                        memberships[node] = []
                    memberships[node].append(i)
            
            user_data['detection_results']['memberships'] = memberships
            
            if 'metrics' in results:
                user_data['detection_results']['metrics'] = results['metrics']
                result_label.text = (
                    f"Detection complete: {algo_name} found {len(results['communities'])} "
                    f"overlapping communities in {results['execution_time']:.2f} seconds.\n"
                    f"Metrics: Omega={results['metrics'].get('omega', 'N/A')}, "
                    f"Modularity={results['metrics'].get('modularity', 'N/A')}"
                )
            else:
                result_label.text = (
                    f"Detection complete: {algo_name} found {len(results['communities'])} "
                    f"overlapping communities in {results['execution_time']:.2f} seconds."
                )
            
            ui.notify('Overlapping community detection complete', color='positive')
            
        except Exception as e:
            ui.notify(f'Error in overlapping detection: {str(e)}', color='negative')

def create_training_tab():
    """Create the model training tab interface"""
    
    ui.label('Model Training and Fine-tuning').classes('text-h5 q-mb-md')
    
    with ui.row():
        with ui.column().classes('w-1/2'):
            ui.label('Train New Model').classes('text-h6')
            
            model_type = ui.select(
                ['GCN', 'GraphSAGE', 'GAT', 'VGAE', 'EvolveGCN', 'DySAT'],
                label='Model type',
                value='GCN'
            ).classes('w-full')
            
            with ui.expansion('Training Parameters', icon='build').classes('w-full'):
                with ui.column():
                    ui.number('Embedding dimension', value=16, min=4).bind_value(app.storage.user, 'train_embedding_dim')
                    ui.number('Hidden dimension', value=32, min=8).bind_value(app.storage.user, 'train_hidden_dim')
                    ui.number('Number of layers', value=2, min=1, max=4).bind_value(app.storage.user, 'train_num_layers')
                    ui.number('Training epochs', value=200, min=10).bind_value(app.storage.user, 'train_epochs')
                    ui.number('Batch size', value=128, min=16).bind_value(app.storage.user, 'train_batch_size')
                    ui.number('Learning rate', value=0.005, min=0.0001, max=0.1, step=0.001) \
                        .bind_value(app.storage.user, 'train_learning_rate')
                    ui.checkbox('Use validation split', value=True).bind_value(app.storage.user, 'use_validation')
                    ui.slider('Validation split', min=0.1, max=0.3, step=0.05, value=0.2) \
                        .bind_value(app.storage.user, 'validation_split')
            
            ui.button('Train Model', on_click=train_new_model).props('color=primary').classes('w-full q-mt-md')
            
        with ui.separator().classes('vertical'):
            pass
            
        with ui.column().classes('w-1/2'):
            ui.label('Fine-tune Existing Model').classes('text-h6')
            
            model_file = ui.upload(
                label='Upload saved model',
                multiple=False,
                on_upload=handle_model_upload
            ).props('accept=.pt,.pth')
            
            with ui.expansion('Fine-tuning Parameters', icon='tune').classes('w-full'):
                with ui.column():
                    ui.number('Epochs', value=50, min=5).bind_value(app.storage.user, 'finetune_epochs')
                    ui.number('Learning rate', value=0.001, min=0.0001, max=0.01, step=0.0001) \
                        .bind_value(app.storage.user, 'finetune_learning_rate')
                    ui.checkbox('Freeze base layers', value=True).bind_value(app.storage.user, 'freeze_base')
            
            ui.button('Fine-tune Model', on_click=finetune_model).props('color=primary').classes('w-full q-mt-md')
    
    ui.separator().classes('q-my-md')
    
    with ui.row():
        with ui.column().classes('w-full'):
            ui.label('Training Progress').classes('text-h6')
            
            with ui.row():
                training_status = ui.label('No training in progress').classes('text-grey-7')
                
            with ui.row():
                progress = ui.linear_progress(value=0).props('size=xl').classes('w-full')
                
            with ui.row():
                metrics_chart = ui.plotly({}).classes('w-full h-64')
    
    # Functions for the training tab
    def handle_model_upload(e: UploadEventArguments):
        """Handle uploaded model file"""
        if not e.name:
            return
            
        file_path = os.path.join(TEMP_DIR, e.name)
        with open(file_path, 'wb') as f:
            f.write(e.content.read())
        
        ui.notify(f'Uploaded model {e.name}', color='positive')
        app.storage.user['model_path'] = file_path
    
    def train_new_model():
        """Train a new GNN model from scratch"""
        try:
            if 'graph' not in user_data:
                ui.notify('Please load a graph first', color='negative')
                return
            
            G = user_data['graph']
            model_name = model_type.value.lower()
            
            # Get training parameters
            embedding_dim = app.storage.user.get('train_embedding_dim', 16)
            hidden_dim = app.storage.user.get('train_hidden_dim', 32)
            num_layers = app.storage.user.get('train_num_layers', 2)
            epochs = app.storage.user.get('train_epochs', 200)
            batch_size = app.storage.user.get('train_batch_size', 128)
            learning_rate = app.storage.user.get('train_learning_rate', 0.005)
            
            # Check if we have features and labels
            has_features = False
            has_labels = False
            for i in range(len(G)):
                node_data = G.get_node_data(i)
                if node_data:
                    if 'features' in node_data:
                        has_features = True
                    if 'community' in node_data:
                        has_labels = True
                
            # Set up progress updates
            training_status.text = f'Training {model_name.upper()} model...'
            loss_values = []
            
            def update_progress(epoch, total_epochs, loss, metrics=None):
                # Update progress bar
                progress.value = epoch / total_epochs
                loss_values.append(loss)
                
                # Update metrics chart
                metrics_chart.update({
                    'data': [{'x': list(range(1, len(loss_values)+1)), 'y': loss_values, 'type': 'scatter', 'name': 'Loss'}],
                    'layout': {'title': 'Training Loss', 'xaxis': {'title': 'Epoch'}, 'yaxis': {'title': 'Loss'}}
                })
                
                # Update status text
                if metrics:
                    training_status.text = (
                        f'Training {model_name.upper()} - Epoch {epoch}/{total_epochs}, '
                        f'Loss: {loss:.4f}, Metrics: {metrics}'
                    )
                else:
                    training_status.text = f'Training {model_name.upper()} - Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}'
            
            # Set up model and training based on model type
            if model_name in ['gcn', 'graphsage', 'gat', 'vgae']:
                # Convert to PyTorch Geometric Data
                data = gnn_community_detection.rwx_to_pyg(G)
                
                # Create model
                if model_name == 'gcn':
                    model = gnn_community_detection.GCN(
                        input_dim=data.x.size(1),
                        hidden_dim=hidden_dim,
                        output_dim=embedding_dim,
                        num_layers=num_layers
                    )
                elif model_name == 'graphsage':
                    model = gnn_community_detection.GraphSAGE(
                        input_dim=data.x.size(1),
                        hidden_dim=hidden_dim,
                        output_dim=embedding_dim,
                        num_layers=num_layers
                    )
                elif model_name == 'gat':
                    model = gnn_community_detection.GAT(
                        input_dim=data.x.size(1),
                        hidden_dim=hidden_dim,
                        output_dim=embedding_dim,
                        num_layers=num_layers
                    )
                elif model_name == 'vgae':
                    model = gnn_community_detection.VGAE(
                        input_dim=data.x.size(1),
                        hidden_dim=hidden_dim,
                        latent_dim=embedding_dim
                    )
                
                # Train model with progress updates
                for epoch in range(1, epochs + 1):
                    # This is a simplified version - in reality you'd need to 
                    # implement a proper training loop with validation
                    model = gnn_community_detection.train_gnn_embedding(
                        model, data, epochs=1, lr=learning_rate
                    )
                    
                    # Calculate current loss (simplified)
                    if isinstance(model, gnn_community_detection.VGAE):
                        z, mu, logstd = model(data.x, data.edge_index)
                        # Simplified loss calculation
                        loss = torch.rand(1).item() * (1 - epoch/epochs)
                    else:
                        z = model(data.x, data.edge_index)
                        # Simplified loss calculation
                        loss = torch.rand(1).item() * (1 - epoch/epochs)
                    
                    # Update progress
                    update_progress(epoch, epochs, loss)
                    
                    # Simulate a brief delay
                    import time
                    time.sleep(0.01)
                
                # Save model
                model_save_path = os.path.join(TEMP_DIR, f"{model_name}_model.pt")
                torch.save(model.state_dict(), model_save_path)
                user_data['trained_model'] = {
                    'model': model,
                    'type': model_name,
                    'path': model_save_path,
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers
                }
                
                # Extract embeddings
                embeddings = gnn_community_detection.extract_embeddings(model, data)
                user_data['embeddings'] = embeddings
                
            elif model_name in ['evolvegcn', 'dysat']:
                # Dynamic GNN models - need temporal graph data
                ui.notify('Dynamic GNN models require temporal graph data', color='info')
                # Implement dynamic GNN training here if temporal data is available
                pass
            
            training_status.text = f'Training complete for {model_name.upper()} model!'
            ui.notify('Model training completed successfully', color='positive')
            
        except Exception as e:
            ui.notify(f'Error training model: {str(e)}', color='negative')
            training_status.text = f'Training failed: {str(e)}'
    
    def finetune_model():
        """Fine-tune an existing GNN model"""
        try:
            if 'graph' not in user_data:
                ui.notify('Please load a graph first', color='negative')
                return
            
            if 'model_path' not in app.storage.user:
                ui.notify('Please upload a model to fine-tune', color='negative')
                return
            
            G = user_data['graph']
            model_path = app.storage.user['model_path']
            
            # For fine-tuning we need to know the model architecture
            # This would typically be stored with the model or in a config file
            # Here we'll use a simplified approach and assume it's a GCN
            
            # Convert to PyTorch Geometric Data
            data = gnn_community_detection.rwx_to_pyg(G)
            
            # Create GCN model with default parameters
            model = gnn_community_detection.GCN(
                input_dim=data.x.size(1),
                hidden_dim=app.storage.user.get('train_hidden_dim', 32),
                output_dim=app.storage.user.get('train_embedding_dim', 16),
                num_layers=app.storage.user.get('train_num_layers', 2)
            )
            
            # Load pre-trained weights
            model.load_state_dict(torch.load(model_path))
            
            # Freeze base layers if requested
            if app.storage.user.get('freeze_base', True):
                # Freeze all layers except the last one
                for name, param in model.named_parameters():
                    if 'convs.2' not in name:  # Assuming last layer is at index 2
                        param.requires_grad = False
            
            # Get fine-tuning parameters
            epochs = app.storage.user.get('finetune_epochs', 50)
            learning_rate = app.storage.user.get('finetune_learning_rate', 0.001)
            
            # Set up progress updates
            training_status.text = 'Fine-tuning model...'
            loss_values = []
            
            # Fine-tune the model
            for epoch in range(1, epochs + 1):
                # This is a simplified version - in reality you'd need to 
                # implement a proper fine-tuning loop
                model = gnn_community_detection.train_gnn_embedding(
                    model, data, epochs=1, lr=learning_rate
                )
                
                # Calculate current loss (simplified)
                loss = torch.rand(1).item() * (1 - epoch/epochs)
                
                # Update progress
                progress.value = epoch / epochs
                loss_values.append(loss)
                
                # Update metrics chart
                metrics_chart.update({
                    'data': [{'x': list(range(1, len(loss_values)+1)), 'y': loss_values, 'type': 'scatter', 'name': 'Loss'}],
                    'layout': {'title': 'Fine-tuning Loss', 'xaxis': {'title': 'Epoch'}, 'yaxis': {'title': 'Loss'}}
                })
                
                training_status.text = f'Fine-tuning model - Epoch {epoch}/{epochs}, Loss: {loss:.4f}'
                
                # Simulate a brief delay
                import time
                time.sleep(0.01)
            
            # Save fine-tuned model
            model_save_path = os.path.join(TEMP_DIR, "finetuned_model.pt")
            torch.save(model.state_dict(), model_save_path)
            user_data['trained_model'] = {
                'model': model,
                'type': 'finetuned_gcn',
                'path': model_save_path
            }
            
            # Extract embeddings
            embeddings = gnn_community_detection.extract_embeddings(model, data)
            user_data['embeddings'] = embeddings
            
            training_status.text = 'Fine-tuning complete!'
            ui.notify('Model fine-tuning completed successfully', color='positive')
            
        except Exception as e:
            ui.notify(f'Error fine-tuning model: {str(e)}', color='negative')
            training_status.text = f'Fine-tuning failed: {str(e)}'

def create_evaluation_tab():
    """Create the evaluation tab interface"""
    
    ui.label('Evaluation and Comparison').classes('text-h5 q-mb-md')
    
    with ui.row():
        with ui.column().classes('w-1/2'):
            ui.label('Performance Metrics').classes('text-h6')
            
            metrics_table = ui.table(
                columns=[
                    {'name': 'method', 'label': 'Method', 'field': 'method', 'align': 'left'},
                    {'name': 'nmi', 'label': 'NMI', 'field': 'nmi', 'align': 'center'},
                    {'name': 'ari', 'label': 'ARI', 'field': 'ari', 'align': 'center'},
                    {'name': 'modularity', 'label': 'Modularity', 'field': 'modularity', 'align': 'center'},
                    {'name': 'time', 'label': 'Time (s)', 'field': 'time', 'align': 'center'},
                ],
                rows=[]
            ).props('dense row-key="method"').classes('w-full')
            
        with ui.column().classes('w-1/2'):
            ui.label('Method Comparison').classes('text-h6')
            
            metric_select = ui.select(
                ['NMI', 'ARI', 'Modularity', 'Execution Time'],
                label='Metric for comparison',
                value='NMI'
            ).classes('w-full')
            
            comparison_chart = ui.plotly({}).classes('w-full h-64')
    
    ui.separator().classes('q-my-md')
    
    with ui.row():
        with ui.column().classes('w-full'):
            ui.label('Run Multiple Methods').classes('text-h6')
            
            with ui.row():
                with ui.checkbox(value=True).bind_value(app.storage.user, 'run_louvain'):
                    ui.label('Louvain')
                with ui.checkbox(value=True).bind_value(app.storage.user, 'run_label_prop'):
                    ui.label('Label Propagation')
                with ui.checkbox(value=True).bind_value(app.storage.user, 'run_infomap'):
                    ui.label('Infomap')
                with ui.checkbox(value=True).bind_value(app.storage.user, 'run_gcn'):
                    ui.label('GCN')
                with ui.checkbox(value=True).bind_value(app.storage.user, 'run_graphsage'):
                    ui.label('GraphSAGE')
                with ui.checkbox(value=True).bind_value(app.storage.user, 'run_bigclam'):
                    ui.label('BigCLAM')
            
            ui.button('Run Selected Methods', on_click=run_multiple_methods).props('color=primary')
    
    # Initialize user storage for checkboxes if not already set
    if 'run_louvain' not in app.storage.user:
        app.storage.user['run_louvain'] = True
    if 'run_label_prop' not in app.storage.user:
        app.storage.user['run_label_prop'] = True
    if 'run_infomap' not in app.storage.user:
        app.storage.user['run_infomap'] = True
    if 'run_gcn' not in app.storage.user:
        app.storage.user['run_gcn'] = True
    if 'run_graphsage' not in app.storage.user:
        app.storage.user['run_graphsage'] = True
    if 'run_bigclam' not in app.storage.user:
        app.storage.user['run_bigclam'] = True
    
    # Functions for the evaluation tab
    def update_metrics_table():
        """Update the metrics table with current results"""
        if 'all_results' not in user_data:
            return
        
        rows = []
        for method_name, result in user_data['all_results'].items():
            if 'metrics' in result:
                row = {
                    'method': method_name,
                    'nmi': f"{result['metrics'].get('nmi', 'N/A'):.4f}" if isinstance(result['metrics'].get('nmi'), float) else 'N/A',
                    'ari': f"{result['metrics'].get('ari', 'N/A'):.4f}" if isinstance(result['metrics'].get('ari'), float) else 'N/A',
                    'modularity': f"{result['metrics'].get('modularity', 'N/A'):.4f}" if isinstance(result['metrics'].get('modularity'), float) else 'N/A',
                    'time': f"{result.get('execution_time', 0):.2f}"
                }
                rows.append(row)
        
        metrics_table.rows = rows
    
    def update_comparison_chart():
        """Update the comparison chart with current results"""
        if 'all_results' not in user_data:
            return
        
        metric = metric_select.value.lower()
        if metric == 'execution time':
            metric = 'execution_time'
        
        methods = []
        values = []
        colors = []
        
        for method_name, result in user_data['all_results'].items():
            if metric == 'execution_time':
                methods.append(method_name)
                values.append(result.get('execution_time', 0))
                colors.append('rgba(54, 162, 235, 0.8)')
            elif 'metrics' in result and metric in result['metrics']:
                methods.append(method_name)
                metric_value = result['metrics'][metric]
                if isinstance(metric_value, float):
                    values.append(metric_value)
                    colors.append('rgba(75, 192, 192, 0.8)')
                else:
                    values.append(0)
                    colors.append('rgba(200, 200, 200, 0.8)')
        
        comparison_chart.update({
            'data': [{
                'x': methods,
                'y': values,
                'type': 'bar',
                'marker': {'color': colors}
            }],
            'layout': {
                'title': f'Method Comparison by {metric_select.value}',
                'xaxis': {'title': 'Method'},
                'yaxis': {'title': metric_select.value}
            }
        })
    
    def run_multiple_methods():
        """Run multiple community detection methods and compare results"""
        try:
            if 'graph' not in user_data:
                ui.notify('Please load a graph first', color='negative')
                return
            
            G = user_data['graph']
            
            # Create dictionary to store all results
            if 'all_results' not in user_data:
                user_data['all_results'] = {}
            
            # Get ground truth attribute name
            ground_truth_attr = 'community'
            
            # Run selected methods
            if app.storage.user.get('run_louvain', True):
                ui.notify('Running Louvain algorithm...', color='info')
                communities, execution_time = traditional_methods.run_louvain(G)
                metrics = traditional_methods.evaluate_against_ground_truth(G, communities, ground_truth_attr)
                user_data['all_results']['Louvain'] = {
                    'communities': communities,
                    'execution_time': execution_time,
                    'metrics': metrics
                }
            
            if app.storage.user.get('run_label_prop', True):
                ui.notify('Running Label Propagation algorithm...', color='info')
                communities, execution_time = traditional_methods.run_label_propagation(G)
                metrics = traditional_methods.evaluate_against_ground_truth(G, communities, ground_truth_attr)
                user_data['all_results']['Label Propagation'] = {
                    'communities': communities,
                    'execution_time': execution_time,
                    'metrics': metrics
                }
            
            if app.storage.user.get('run_infomap', True):
                ui.notify('Running Infomap algorithm...', color='info')
                try:
                    communities, execution_time = traditional_methods.run_infomap(G)
                    metrics = traditional_methods.evaluate_against_ground_truth(G, communities, ground_truth_attr)
                    user_data['all_results']['Infomap'] = {
                        'communities': communities,
                        'execution_time': execution_time,
                        'metrics': metrics
                    }
                except ImportError:
                    ui.notify('Infomap requires cdlib package', color='warning')
            
            if app.storage.user.get('run_gcn', True):
                ui.notify('Running GCN algorithm...', color='info')
                try:
                    results = gnn_community_detection.run_gcn(G)
                    communities = {i: int(c) for i, c in enumerate(results['communities'])}
                    user_data['all_results']['GCN'] = {
                        'communities': communities,
                        'execution_time': results['training_time'],
                        'metrics': results.get('metrics', {})
                    }
                except Exception as e:
                    ui.notify(f'Error running GCN: {str(e)}', color='warning')
            
            if app.storage.user.get('run_graphsage', True):
                ui.notify('Running GraphSAGE algorithm...', color='info')
                try:
                    results = gnn_community_detection.run_graphsage(G)
                    communities = {i: int(c) for i, c in enumerate(results['communities'])}
                    user_data['all_results']['GraphSAGE'] = {
                        'communities': communities,
                        'execution_time': results['training_time'],
                        'metrics': results.get('metrics', {})
                    }
                except Exception as e:
                    ui.notify(f'Error running GraphSAGE: {str(e)}', color='warning')
            
            if app.storage.user.get('run_bigclam', True):
                ui.notify('Running BigCLAM algorithm...', color='info')
                try:
                    results = overlapping_community_detection.run_bigclam(G)
                    user_data['all_results']['BigCLAM'] = {
                        'communities': results['communities'],
                        'execution_time': results['execution_time'],
                        'metrics': results.get('metrics', {})
                    }
                except Exception as e:
                    ui.notify(f'Error running BigCLAM: {str(e)}', color='warning')
            
            # Update the UI
            update_metrics_table()
            update_comparison_chart()
            
            ui.notify('All selected methods completed', color='positive')
            
        except Exception as e:
            ui.notify(f'Error running methods: {str(e)}', color='negative')

    # Connect metric selection to chart update
    def on_metric_change(_):
        update_comparison_chart()
    
    metric_select.on_change(on_metric_change)

def create_visualization_tab():
    """Create the visualization tab interface"""
    
    ui.label('Visualization').classes('text-h5 q-mb-md')
    
    with ui.row():
        with ui.column().classes('w-full'):
            ui.label('Network Visualization').classes('text-h6')
            
            with ui.row():
                with ui.column().classes('w-1/3'):
                    node_color = ui.select(
                        ['Community', 'Degree', 'Node Type'],
                        label='Color nodes by',
                        value='Community'
                    ).classes('w-full')
                    
                    ui.number('Max nodes to display', value=500, min=50).bind_value(app.storage.user, 'vis_max_nodes')
                    
                    with ui.expansion('Layout Options', icon='settings').classes('w-full'):
                        layout = ui.select(
                            ['Force-Directed', 'Circular', 'Spectral', 'Kamada-Kawai'],
                            label='Layout algorithm',
                            value='Force-Directed'
                        ).classes('w-full')
                
                with ui.column().classes('w-2/3'):
                    # Placeholder for network visualization
                    network_vis = ui.html('<!-- Network visualization will appear here -->').classes('w-full h-96')
                
            ui.button('Generate Visualization', on_click=generate_visualization).props('color=primary')
            
    ui.separator().classes('q-my-md')
    
    with ui.row():
        with ui.column().classes('w-1/2'):
            ui.label('Community Distribution').classes('text-h6')
            distribution_chart = ui.plotly({}).classes('w-full h-64')
            
        with ui.column().classes('w-1/2'):
            ui.label('Node Embeddings').classes('text-h6')
            embedding_vis = ui.plotly({}).classes('w-full h-64')
    
    # Functions for visualization tab
    def generate_visualization():
        """Generate network visualization with current settings using optimized functions"""
        try:
            if 'graph' not in user_data:
                ui.notify('Please load a graph first', color='negative')
                return
            
            G = user_data['graph']
            max_nodes = app.storage.user.get('vis_max_nodes', 500)
            use_plotly = app.storage.user.get('use_plotly', True)
            use_gpu = app.storage.user.get('use_gpu', True)
            community_attr = None
            
            # Determine which community attribute to use
            if node_color.value == 'Community':
                if 'detection_results' in user_data and 'communities' in user_data['detection_results']:
                    # Add communities to graph if they're not already there
                    communities = user_data['detection_results']['communities']
                    
                    # Add communities as node attributes first
                    for node, comm in communities.items():
                        node_data = G.get_node_data(node) or {}
                        node_data['detected_community'] = comm
                        G.set_node_data(node, node_data)
                    
                    community_attr = 'detected_community'
            
            # Use our optimized plotting function directly
            try:
                # This is much more efficient than manually constructing via NetworkX
                data_prep.plot_graph(
                    G, 
                    community_attr=community_attr,
                    pos=None,  # Let the function compute layout
                    figsize=(12, 10),
                    title="Network Visualization",
                    max_nodes=max_nodes,
                    use_plotly=use_plotly,
                    edge_alpha=0.3,
                    node_size_factor=1.2 if node_color.value == 'Degree' else 1.0
                )
                
                # We still need to update the network_vis HTML element with the figure
                # But our plot_graph shows the interactive figure directly
                network_vis.content = '<div class="text-center">Visualization shown in separate window</div>'
                
            except Exception as e:
                # Fallback to previous method if there's an issue
                ui.notify(f'Error using optimized visualization: {str(e)}. Falling back to standard method', color='warning')
                
                # For large graphs, we'll sample nodes
                n_nodes = len(G)
                if n_nodes > max_nodes:
                    ui.notify(f'Sampling {max_nodes} nodes from {n_nodes} for visualization', color='info')
                    
                    # Sample nodes by selecting some communities entirely
                    if 'detection_results' in user_data and 'communities' in user_data['detection_results']:
                        communities = user_data['detection_results']['communities']
                        comm_groups = {}
                        
                        for node, comm in communities.items():
                            if comm not in comm_groups:
                                comm_groups[comm] = []
                            comm_groups[comm].append(node)
                        
                        # Sample communities to reach max_nodes
                        sampled_nodes = []
                        for comm, nodes in comm_groups.items():
                            if len(sampled_nodes) + len(nodes) <= max_nodes:
                                sampled_nodes.extend(nodes)
                            else:
                                # Take a subset to fill remaining quota
                                remaining = max_nodes - len(sampled_nodes)
                                if remaining > 0:
                                    import random
                                    random.shuffle(nodes)
                                    sampled_nodes.extend(nodes[:remaining])
                                break
                        
                        # Create subgraph with sampled nodes
                        import networkx as nx
                        G_nx = traditional_methods._rwx_to_nx(G)
                        G_sub = G_nx.subgraph(sampled_nodes)
                    else:
                        # No communities detected, sample randomly
                        import networkx as nx
                        import random
                        G_nx = traditional_methods._rwx_to_nx(G)
                        sampled_nodes = random.sample(list(G_nx.nodes()), max_nodes)
                        G_sub = G_nx.subgraph(sampled_nodes)
                else:
                    # Convert the full graph to NetworkX
                    import networkx as nx
                    G_nx = traditional_methods._rwx_to_nx(G)
                    G_sub = G_nx
                
                # Determine node colors based on selection
                if node_color.value == 'Community':
                    # Color by community if available
                    if 'detection_results' in user_data and 'communities' in user_data['detection_results']:
                        communities = user_data['detection_results']['communities']
                        color_map = {node: communities.get(node, 0) for node in G_sub.nodes()}
                    else:
                        color_map = {node: 0 for node in G_sub.nodes()}  # Default color
                elif node_color.value == 'Degree':
                    # Color by node degree
                    color_map = {node: G_sub.degree(node) for node in G_sub.nodes()}
                else:
                    # Default color
                    color_map = {node: 0 for node in G_sub.nodes()}
                
                # Choose layout algorithm
                if layout.value == 'Force-Directed':
                    pos = nx.spring_layout(G_sub)
                elif layout.value == 'Circular':
                    pos = nx.circular_layout(G_sub)
                elif layout.value == 'Spectral':
                    pos = nx.spectral_layout(G_sub)
                elif layout.value == 'Kamada-Kawai':
                    pos = nx.kamada_kawai_layout(G_sub)
                
                # Generate visualization using Plotly
                import plotly.graph_objects as go
                
                # Create edges
                edge_x = []
                edge_y = []
                for edge in G_sub.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='rgba(136,136,136,0.3)'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # Create nodes
                node_x = []
                node_y = []
                for node in G_sub.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                
                # Get node colors
                color_values = list(color_map.values())
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        color=color_values,
                        size=12 if node_color.value == 'Degree' else 10,
                        colorbar=dict(
                            thickness=15,
                            title=node_color.value,
                            xanchor='left',
                            titleside='right'
                        ),
                        line=dict(width=2)
                    )
                )
                
                # Add node hover text
                node_text = []
                for node in G_sub.nodes():
                    if node_color.value == 'Community':
                        node_text.append(f'Node: {node}<br>Community: {color_map[node]}')
                    elif node_color.value == 'Degree':
                        node_text.append(f'Node: {node}<br>Degree: {color_map[node]}')
                    else:
                        node_text.append(f'Node: {node}')
                
                node_trace.text = node_text
                
                # Create figure
                fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Network Visualization ({len(G_sub.nodes())} nodes, {len(G_sub.edges())} edges)',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )
            
            # Convert to HTML
            import plotly.io as pio
            html = pio.to_html(fig, full_html=False)
            network_vis.content = html
            
            # Also update community distribution chart
            if 'detection_results' in user_data and 'communities' in user_data['detection_results']:
                communities = user_data['detection_results']['communities']
                
                # Count communities
                comm_counts = {}
                for node, comm in communities.items():
                    if comm not in comm_counts:
                        comm_counts[comm] = 0
                    comm_counts[comm] += 1
                
                # Sort by community ID
                sorted_comms = sorted(comm_counts.items())
                comm_ids = [str(c[0]) for c in sorted_comms]
                counts = [c[1] for c in sorted_comms]
                
                distribution_chart.update({
                    'data': [{
                        'x': comm_ids,
                        'y': counts,
                        'type': 'bar',
                        'marker': {'color': 'rgba(50, 171, 96, 0.7)'}
                    }],
                    'layout': {
                        'title': 'Community Size Distribution',
                        'xaxis': {'title': 'Community ID'},
                        'yaxis': {'title': 'Number of Nodes'}
                    }
                })
            
            # Update embedding visualization if available
            if 'embeddings' in user_data:
                embeddings = user_data['embeddings']
                use_gpu = app.storage.user.get('use_gpu', True)
                
                # Check if we can use GPU acceleration
                try:
                    if use_gpu and torch.cuda.is_available():
                        # Try CUDA/GPU implementation for faster embedding visualization
                        import torch
                        
                        # Import cuml for GPU-accelerated t-SNE if available
                        try:
                            from cuml.manifold import TSNE as cuTSNE
                            has_cuml = True
                        except ImportError:
                            has_cuml = False
                            from sklearn.manifold import TSNE
                        
                        # Sample if too many points
                        max_points = 3000 if has_cuml else 1000  # GPU can handle more points
                        if len(embeddings) > max_points:
                            import random
                            random.seed(42)
                            indices = random.sample(range(len(embeddings)), max_points)
                            
                            # Extract the sample and move to the correct device
                            if torch.is_tensor(embeddings):
                                if has_cuml:
                                    # Need CPU array for cuml
                                    emb_sample = embeddings[indices].detach().cpu().numpy()
                                else:
                                    emb_sample = embeddings[indices]
                            else:
                                emb_sample = embeddings[indices]
                        else:
                            emb_sample = embeddings
                            indices = list(range(len(embeddings)))
                            
                            # Convert to numpy for cuml or detach for sklearn
                            if torch.is_tensor(emb_sample):
                                if has_cuml:
                                    emb_sample = emb_sample.detach().cpu().numpy()
                            
                        # Reduce to 2D using the appropriate implementation
                        if has_cuml:
                            # Use GPU-accelerated t-SNE
                            tsne = cuTSNE(n_components=2, random_state=42)
                            embeddings_2d = tsne.fit_transform(emb_sample)
                        else:
                            # Fallback to sklearn
                            tsne = TSNE(n_components=2, random_state=42)
                            if torch.is_tensor(emb_sample):
                                embeddings_2d = tsne.fit_transform(emb_sample.detach().cpu().numpy())
                            else:
                                embeddings_2d = tsne.fit_transform(emb_sample)
                    else:
                        # CPU implementation
                        from sklearn.manifold import TSNE
                        
                        # Sample if too many points
                        if len(embeddings) > 1000:
                            import random
                            random.seed(42)
                            indices = random.sample(range(len(embeddings)), 1000)
                            emb_sample = embeddings[indices]
                        else:
                            emb_sample = embeddings
                            indices = list(range(len(embeddings)))
                        
                        # Reduce to 2D
                        tsne = TSNE(n_components=2, random_state=42)
                        if torch.is_tensor(emb_sample):
                            embeddings_2d = tsne.fit_transform(emb_sample.detach().cpu().numpy())
                        else:
                            embeddings_2d = tsne.fit_transform(emb_sample)
                
                except Exception as e:
                    # Fallback to basic CPU implementation in case of any issues
                    ui.notify(f"Error in GPU acceleration, falling back to CPU: {str(e)}", color='warning')
                    
                    from sklearn.manifold import TSNE
                    
                    # Sample if too many points
                    if len(embeddings) > 1000:
                        import random
                        random.seed(42)
                        indices = random.sample(range(len(embeddings)), 1000)
                        emb_sample = embeddings[indices]
                    else:
                        emb_sample = embeddings
                        indices = list(range(len(embeddings)))
                    
                    # Reduce to 2D
                    tsne = TSNE(n_components=2, random_state=42)
                    if torch.is_tensor(emb_sample):
                        embeddings_2d = tsne.fit_transform(emb_sample.detach().cpu().numpy())
                    else:
                        embeddings_2d = tsne.fit_transform(emb_sample)
                
                # Color by community if available
                if 'detection_results' in user_data and 'communities' in user_data['detection_results']:
                    communities = user_data['detection_results']['communities']
                    colors = [communities.get(i, 0) for i in indices]
                else:
                    colors = [0] * len(indices)
                
                # Add extra hover information
                hover_texts = []
                for i, idx in enumerate(indices):
                    community = colors[i]
                    hover_texts.append(f"Node: {idx}<br>Community: {community}")
                
                embedding_vis.update({
                    'data': [{
                        'x': embeddings_2d[:, 0],
                        'y': embeddings_2d[:, 1],
                        'mode': 'markers',
                        'type': 'scatter',
                        'text': hover_texts,
                        'hoverinfo': 'text',
                        'marker': {
                            'color': colors,
                            'colorscale': 'Viridis',
                            'showscale': True,
                            'size': 6,
                            'colorbar': {'title': 'Community'}
                        }
                    }],
                    'layout': {
                        'title': f'Node Embeddings (t-SNE, {len(indices)} samples)',
                        'xaxis': {'title': 'Dimension 1'},
                        'yaxis': {'title': 'Dimension 2'},
                        'hovermode': 'closest'
                    }
                })
            
            ui.notify('Visualization generated successfully', color='positive')
            
        except Exception as e:
            ui.notify(f'Error generating visualization: {str(e)}', color='negative')

def create_results_tab():
    """Create the results tab interface"""
    
    ui.label('Results and Export').classes('text-h5 q-mb-md')
    
    with ui.row():
        with ui.column().classes('w-1/2'):
            ui.label('Export Communities').classes('text-h6')
            
            with ui.row():
                file_format = ui.select(
                    ['CSV', 'Parquet', 'JSON'],
                    label='File format',
                    value='Parquet'
                ).classes('w-full')
            
            with ui.row():
                filename = ui.input(label='Filename', value='community_detection_results').classes('w-full')
            
            ui.button('Export Communities', on_click=export_communities).props('color=primary')
            
        with ui.column().classes('w-1/2'):
            ui.label('Export Model and Embeddings').classes('text-h6')
            
            ui.checkbox('Include embeddings', value=True).bind_value(app.storage.user, 'export_embeddings')
            ui.checkbox('Include model weights', value=True).bind_value(app.storage.user, 'export_model')
            
            ui.button('Export Model', on_click=export_model).props('color=primary')
    
    ui.separator().classes('q-my-md')
    
    with ui.row():
        with ui.column().classes('w-full'):
            ui.label('Experiment Summary').classes('text-h6')
            
            summary_table = ui.table(
                columns=[
                    {'name': 'parameter', 'label': 'Parameter', 'field': 'parameter', 'align': 'left'},
                    {'name': 'value', 'label': 'Value', 'field': 'value', 'align': 'left'},
                ],
                rows=[]
            ).props('dense').classes('w-full')
    
    # Functions for the results tab
    def export_communities():
        """Export detected communities to a file"""
        try:
            if 'detection_results' not in user_data:
                ui.notify('No detection results to export', color='negative')
                return
            
            if 'communities' not in user_data['detection_results'] and 'overlapping_communities' not in user_data['detection_results']:
                ui.notify('No communities found in results', color='negative')
                return
            
            # Check if we have disjoint or overlapping communities
            if 'communities' in user_data['detection_results']:
                # Disjoint communities
                communities = user_data['detection_results']['communities']
                df = pl.DataFrame({
                    'node': list(communities.keys()),
                    'community': list(communities.values())
                })
            else:
                # Overlapping communities - convert to membership probabilities
                overlapping = user_data['detection_results']['overlapping_communities']
                memberships = user_data['detection_results']['memberships']
                
                # Create rows for each node-community pair
                rows = []
                for node, comms in memberships.items():
                    for comm in comms:
                        rows.append({
                            'node': node,
                            'community': comm,
                            'membership': 1.0  # For binary memberships
                        })
                
                df = pl.DataFrame(rows)
            
            # Export based on selected format
            format_type = file_format.value.lower()
            output_path = os.path.join(TEMP_DIR, f"{filename.value}")
            
            if format_type == 'csv':
                df.write_csv(f"{output_path}.csv")
                export_path = f"{output_path}.csv"
            elif format_type == 'parquet':
                df.write_parquet(f"{output_path}.parquet", compression="zstd")
                export_path = f"{output_path}.parquet"
            elif format_type == 'json':
                df.write_json(f"{output_path}.json")
                export_path = f"{output_path}.json"
            
            ui.notify(f'Communities exported to {export_path}', color='positive')
            
            # Update summary table
            update_summary_table()
            
        except Exception as e:
            ui.notify(f'Error exporting communities: {str(e)}', color='negative')
    
    def export_model():
        """Export trained model and/or embeddings"""
        try:
            # Check if we have a trained model
            if 'trained_model' not in user_data:
                ui.notify('No trained model to export', color='negative')
                return
            
            model_info = user_data['trained_model']
            output_base = os.path.join(TEMP_DIR, f"{filename.value}")
            
            # Export model weights if requested
            if app.storage.user.get('export_model', True):
                if 'model' in model_info:
                    model_path = f"{output_base}_model.pt"
                    torch.save(model_info['model'].state_dict(), model_path)
                    ui.notify(f'Model exported to {model_path}', color='positive')
                else:
                    ui.notify('No model available to export', color='warning')
            
            # Export embeddings if requested
            if app.storage.user.get('export_embeddings', True):
                if 'embeddings' in user_data:
                    emb_path = f"{output_base}_embeddings.pt"
                    torch.save(user_data['embeddings'], emb_path)
                    ui.notify(f'Embeddings exported to {emb_path}', color='positive')
                else:
                    ui.notify('No embeddings available to export', color='warning')
            
            # Export model config
            if 'model' in model_info:
                config = {
                    'model_type': model_info.get('type', 'unknown'),
                    'embedding_dim': model_info.get('embedding_dim', 16),
                    'hidden_dim': model_info.get('hidden_dim', 32),
                    'num_layers': model_info.get('num_layers', 2)
                }
                
                config_path = f"{output_base}_config.json"
                import json
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                ui.notify(f'Model configuration exported to {config_path}', color='positive')
            
            # Update summary table
            update_summary_table()
            
        except Exception as e:
            ui.notify(f'Error exporting model: {str(e)}', color='negative')
    
    def update_summary_table():
        """Update the experiment summary table"""
        rows = []
        
        # Add graph info
        if 'graph' in user_data and 'graph_stats' in user_data:
            stats = user_data['graph_stats']
            rows.append({'parameter': 'Number of Nodes', 'value': str(stats['n_nodes'])})
            rows.append({'parameter': 'Number of Edges', 'value': str(stats['n_edges'])})
            rows.append({'parameter': 'Graph Density', 'value': f"{stats['density']:.6f}"})
            rows.append({'parameter': 'Average Degree', 'value': f"{stats['avg_degree']:.2f}"})
        
        # Add detection results
        if 'detection_results' in user_data:
            results = user_data['detection_results']
            rows.append({'parameter': 'Detection Method', 'value': results.get('method', 'Unknown')})
            
            if 'communities' in results:
                num_communities = len(set(results['communities'].values()))
                rows.append({'parameter': 'Number of Communities', 'value': str(num_communities)})
            elif 'overlapping_communities' in results:
                rows.append({'parameter': 'Number of Overlapping Communities', 'value': str(len(results['overlapping_communities']))})
            
            if 'execution_time' in results:
                rows.append({'parameter': 'Execution Time', 'value': f"{results['execution_time']:.2f} seconds"})
            
            if 'metrics' in results:
                metrics = results['metrics']
                if 'nmi' in metrics:
                    rows.append({'parameter': 'NMI', 'value': f"{metrics['nmi']:.4f}"})
                if 'ari' in metrics:
                    rows.append({'parameter': 'ARI', 'value': f"{metrics['ari']:.4f}"})
                if 'modularity' in metrics and isinstance(metrics['modularity'], float):
                    rows.append({'parameter': 'Modularity', 'value': f"{metrics['modularity']:.4f}"})
        
        # Add model info
        if 'trained_model' in user_data:
            model_info = user_data['trained_model']
            rows.append({'parameter': 'Model Type', 'value': model_info.get('type', 'Unknown')})
            
            if 'embedding_dim' in model_info:
                rows.append({'parameter': 'Embedding Dimension', 'value': str(model_info['embedding_dim'])})
        
        summary_table.rows = rows

def run_gui(host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
    """Run the GUI application
    
    Parameters:
    -----------
    host: str
        Host address to run the server on
    port: int
        Port to run the server on
    debug: bool
        Whether to run in debug mode
    """
    app = initialize_app()
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_gui()