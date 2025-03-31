Customizing the Application
Using Your Own Data
Replace the mock data in app.py with your actual network data by modifying the load_data() function. Your data should provide:

edge_index: PyTorch tensor of shape [2, num_edges] representing connections
num_nodes: Total number of nodes in the network
x: Node features
true_labels: Ground truth community labels (if available)
model: Your trained GNN model

Adding New Clustering Algorithms
To add a new clustering algorithm:

Add it to the clustering_methods dictionary in the code
Update the UI select options
Implement the algorithm in the run_clustering function

License
[Your License Here]
Contact
[Your Contact Information]