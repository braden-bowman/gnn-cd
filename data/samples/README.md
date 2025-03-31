# Sample Data Files

This directory contains small sample data files that can be used for testing and demonstration purposes.

## UNSW-NB15 Samples

### unsw_features_sample.csv
Contains the feature descriptions for the UNSW-NB15 dataset. This includes information about:
- Feature ID
- Feature name
- Data type (nominal, integer, float, etc.)
- Feature description

### unsw_sample.csv
Contains 10 sample records from the UNSW-NB15 dataset:
- 5 normal network traffic records (label=0)
- 5 attack records (label=1) of different types (exploit, brute-force, dos, reconnaissance)

This sample data is intended for testing purposes and doesn't represent the full complexity of the actual UNSW-NB15 dataset.

## Using Sample Data

These samples can be used to test the processing and analysis scripts:

```python
# Example usage with sample data
from community_detection.data_prep import load_data
import pandas as pd

# Load sample data
sample_data = pd.read_csv('data/samples/unsw_sample.csv')
features_info = pd.read_csv('data/samples/unsw_features_sample.csv')

# Use the sample data with processing functions
# ...
```

## Getting Full Datasets

For the complete UNSW-NB15 dataset, visit the official website:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

Other datasets used in this project:
- Synthetic graph data in `notebooks/data/` directory
- Custom generated datasets through the data generation functions