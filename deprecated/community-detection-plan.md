# Community Detection in Graph Neural Networks: A Comparison Framework

## Overview
This document outlines a framework for comparing different community detection approaches in graph neural networks. We'll create a series of notebooks that apply various methods to the same dataset and compare their ability to recover ground truth community structures.

## Data Sources
- **DER cyber**: Distributed Energy Resource cybersecurity data
- **AIS**: Automatic Identification System data
- **PAI**: Additional data to be requested
- **Benchmark datasets**: For validation (Zachary's Karate Club, Facebook networks, etc.)

## Notebook Series

### 1. Data Preparation and Exploration
- Data loading and preprocessing
- Feature selection/engineering
- Graph construction from raw data
- Exploratory analysis of graph properties
- Visualization of initial graph structure

### 2. Traditional Community Detection (Baseline)
- Louvain method
- Leiden algorithm 
- Infomap
- Spectral clustering
- Hierarchical clustering
- Evaluation against ground truth

### 3. Node Embedding + Clustering Approaches
- Node2Vec + K-means
- DeepWalk + K-means
- SDNE + K-means
- LINE + K-means
- Evaluation against ground truth

### 4. Graph Neural Network Approaches
- **GCN (Graph Convolutional Networks)**
  - Semi-supervised approach with community labels
  - Unsupervised approach with clustering decoder
  
- **GraphSAGE**
  - Various aggregation functions
  - Supervised and unsupervised variants
  
- **GAT (Graph Attention Networks)**
  - Attention mechanism for community detection
  - Multi-head attention variations
  
- **Graph Autoencoders**
  - VGAE (Variational Graph Auto-Encoder)
  - Adversarial approaches

### 5. Temporal/Dynamic Graph Approaches
- **EvolveGCN**
  - Tracking community evolution
  - RNN-based parameter updates
  
- **DySAT**
  - Self-attention based temporal GNN
  
- **CTDNE (Continuous-Time Dynamic Network Embeddings)**
  - Temporal random walks

### 6. Overlapping Community Detection
- **BigCLAM**
  - Overlapping community detection
  
- **Graph Neural Networks for Overlapping Communities**
  - Modified objectives for soft assignments
  
- **DMGI (Deep Multiplex Graph Infomax)**
  - For multiplex networks

### 7. Comprehensive Evaluation and Visualization
- Comparative analysis of all methods
- Performance metrics:
  - NMI (Normalized Mutual Information)
  - ARI (Adjusted Rand Index) 
  - F1 score
  - Modularity
  - Conductance
- Visualization of communities using the approaches from Vehlow et al.
- Time complexity and scalability analysis
- Robustness to noise and parameter sensitivity

## Graph Types to Explore
- Real-world graphs from data sources
- Synthetic graphs with known community structure:
  - Stochastic Block Models
  - LFR benchmark graphs
  - Newman-Watts-Strogatz small-world networks
  - Barabási–Albert scale-free networks

## Implementation Requirements
- PyTorch Geometric or DGL for GNN implementations
- NetworkX for graph manipulation and traditional methods
- Scikit-learn for evaluation metrics
- Matplotlib, Plotly, and other visualization libraries
- Karate Club library for benchmarking

## Evaluation Strategy
1. For each dataset, establish ground truth communities
2. Apply each method with optimal hyperparameters
3. Compare detected communities against ground truth
4. Analyze strengths and weaknesses of each approach
5. Identify which methods work best for different types of graphs
