# UNSW-NB15 Dataset for Cybersecurity Community Detection

This directory contains scripts for processing and analyzing the UNSW-NB15 cybersecurity dataset using community detection techniques.

## Features

- **Efficient Processing**: Uses Polars for fast data processing and RustworkX for graph operations
- **GPU Acceleration**: GPU-ready data formats for faster model training
- **Caching**: Intelligent caching to avoid reprocessing data multiple times
- **Memory Optimization**: Tools to monitor and optimize memory usage for large datasets
- **Parallel Execution**: Support for batch processing to speed up operations

## Dataset Description

The UNSW-NB15 dataset is a comprehensive network traffic dataset that contains normal and attack traffic. It includes:
- Over 2 million records of network traffic 
- 49 features for each record
- Labels for 9 different attack types and normal traffic
- A variety of attack scenarios including DoS, worms, backdoors, and more

## Download Instructions

1. Download the dataset from the official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

2. Download the following files and place them in this directory:
   - `UNSW-NB15_features.csv` - Feature descriptions
   - `UNSW-NB15_1.csv` - First part of the dataset (can also download parts 2-4 for more data)
   - `UNSW-NB15_GT.csv` - Ground truth labels (optional)

## Directory Structure

```
/data/unsw/
  ├── process_unsw.py        - Process data and construct graph
  ├── evaluate_community_detection.py - Apply community detection methods
  ├── memory_utils.py        - Memory monitoring utilities
  ├── processed/             - Cached processed data
  │   ├── all_data.parquet   - Processed dataset in Parquet format
  │   ├── selected_features.parquet - Selected features cache
  │   ├── unsw_graph.pt      - RustworkX graph file
  │   ├── gpu_data/          - GPU-ready data formats
  │   └── results/           - Evaluation results
  └── README.md              - This file
```

## Usage

### Quick Start

For a quick start, download the dataset and run:

```bash
python process_unsw.py         # Process data and build graph
python evaluate_community_detection.py  # Run evaluation
```

### Processing Pipeline

The processing pipeline automatically uses caching to avoid repeated work:

1. **Data Loading**: First loads dataset with Polars for fast processing
   ```bash
   python process_unsw.py
   ```

2. **Feature Selection**: Performs feature selection and caches results
   - Selected features are saved to `processed/selected_features.parquet`

3. **Graph Construction**: Builds network graph with devices as nodes
   - The constructed graph is saved to `processed/unsw_graph.pt`
   - GPU-ready data is saved to `processed/gpu_data/gpu_ready_data.pt`

4. **Community Detection**: Applies various detection methods with caching
   ```bash
   python evaluate_community_detection.py
   ```
   - Results are saved to `processed/results/community_detection_results.pkl`
   - Visualizations are saved to `processed/results/`

5. **Attack Analysis**: Analyzes communities with high attack concentrations
   - Identifies communities with high concentrations of attack traffic
   - Compares methods by attack detection capabilities

### Using GPU Acceleration

The implementation automatically detects GPU availability:

```python
# The code checks for GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU data is loaded to the appropriate device
X = gpu_data['x'].to(DEVICE)
y = gpu_data['y'].to(DEVICE)
```

### Memory Monitoring

For large datasets, you can use the memory monitoring utilities:

```python
from memory_utils import MemoryTracker, memory_monitor

# Using the context manager
with MemoryTracker("loading data"):
    data = load_dataset(...)

# Using the decorator
@memory_monitor
def process_large_dataset():
    # Processing code here
```

## Extending the Implementation

To adapt this for different datasets:
1. Modify the data loading functions in `process_unsw.py`
2. Adjust the feature selection parameters for your dataset
3. Update the graph construction logic if needed

For adding new community detection methods:
1. Import the method in `evaluate_community_detection.py`
2. Add it to the `methods` dictionary in `evaluate_all_methods()`