# UNSW-NB15 Cybersecurity Analysis with Community Detection

This document describes the addition of cybersecurity analysis capabilities to the GNN Community Detection framework, specifically focusing on the UNSW-NB15 dataset with optimized data processing using Polars and GPU acceleration.

## Overview

We've extended the GNN-CD framework with optimized data handling for cybersecurity use cases by:

1. Using Polars for efficient data processing and RustworkX for graph operations
2. Implementing intelligent caching to avoid repeated processing
3. Creating GPU-ready data formats for faster training and evaluation
4. Adding memory monitoring utilities for handling large datasets
5. Implementing specialized evaluation metrics for attack detection
6. Creating a comprehensive analytical pipeline for cybersecurity applications

## Key Features

### Optimized Data Processing with Polars

The implementation uses Polars, a lightning-fast DataFrame library, for efficient data processing:
- Up to 10x faster CSV parsing compared to pandas
- Memory-efficient operations through lazy evaluation
- Native Parquet support for fast data storage and retrieval
- Parallel execution of operations for improved throughput

```python
# Example of Polars usage for efficient filtering
ip_data = data.filter((pl.col('srcip') == ip) | (pl.col('dstip') == ip))
```

### GPU Acceleration for GNNs

The implementation includes GPU-ready data formats for optimal performance:
- Automatic detection of GPU availability
- PyTorch Geometric compatible data structures
- Proper tensor device placement for GPU training
- Memory-efficient sparse tensor representations

```python
# Example of GPU data preparation
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(DEVICE)
```

### Intelligent Caching System

To avoid repeated processing of large datasets, a comprehensive caching system is implemented:

1. **Dataset Caching**: Raw dataset is cached as optimized Parquet files
2. **Feature Selection Caching**: Selected features are cached for faster access
3. **Graph Caching**: The constructed graph is cached with versioning
4. **Results Caching**: All community detection results are cached for comparison

```python
# Example of cache checking logic
if os.path.exists(FEATURES_CACHE):
    features_df = pl.read_parquet(FEATURES_CACHE)
    # Use cached features
else:
    # Perform feature selection and cache results
    features_df = perform_feature_selection(data)
    features_df.write_parquet(FEATURES_CACHE)
```

### Memory Monitoring Utilities

For handling large datasets, memory monitoring utilities are provided:
- Decorator for tracking memory usage in functions
- Context manager for monitoring code blocks
- Automatic memory cleanup and optimization
- GPU memory tracking for PyTorch operations

```python
@memory_monitor
def process_large_dataset():
    # Processing code here

with MemoryTracker("loading data"):
    data = load_dataset(...)
```

## Implementation Details

### Processing Pipeline

The cybersecurity analysis implementation follows this pipeline:

1. **Data Loading and Preprocessing**
   - Efficiently load data with Polars and caching
   - Perform feature selection with F-test scoring
   - Cache preprocessing results in Parquet format

2. **Graph Construction**
   - Build network graph with devices as nodes
   - Add edges based on network traffic
   - Process data in chunks for memory efficiency
   - Create GPU-ready data formats

3. **Community Detection**
   - Apply traditional, GNN-based, and overlapping methods
   - Use GPU acceleration for neural models
   - Cache results for comparison
   - Evaluate performance with cybersecurity metrics

4. **Attack Analysis**
   - Identify communities with high attack concentrations
   - Compare methods on attack detection capabilities
   - Visualize attack patterns within communities

### Cybersecurity-Specific Evaluation

The implementation includes metrics specific to cybersecurity:
- **Attack Concentration**: Measure of attack traffic density in communities
- **Community Purity**: Homogeneity of communities by traffic type
- **Attack Community Ratio**: Proportion of attack-related communities

```python
# Example of attack concentration calculation
attack_count = sum(G.get_node_data(node)['label'] for node in community)
concentration = attack_count / len(community) if community else 0
```

## Usage Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   ```bash
   python data/unsw/process_unsw.py
   ```

3. **Run Evaluation**:
   ```bash
   python data/unsw/evaluate_community_detection.py
   ```

4. **Explore Notebook**:
   ```bash
   jupyter notebook notebooks/7_UNSW_Cybersecurity_Analysis.ipynb
   ```

## Results and Analysis

The implementation enables comprehensive analysis of the UNSW-NB15 dataset:

1. **Feature Importance**: Identification of key features for attack detection
2. **Community Structure**: Visualization of network traffic patterns
3. **Method Comparison**: Evaluation of detection methods by various metrics
4. **Attack Communities**: Analysis of communities with high attack concentration

These results provide valuable insights for cybersecurity practitioners:
- Which network features are most indicative of attacks
- Which community detection methods are best for attack identification
- How attack traffic forms distinguishable patterns in networks

## Future Extensions

The framework can be extended in several ways:

1. **Real-time Analysis**: Adapt for streaming data with online community detection
2. **Multi-dataset Support**: Add support for other cybersecurity datasets
3. **Advanced Visualization**: Create interactive dashboards for security operations
4. **Custom GNN Models**: Develop specialized GNN architectures for attack detection