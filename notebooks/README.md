# Community Detection Notebooks

This directory contains Jupyter notebooks demonstrating different community detection approaches:

1. `1_Data_Preperation.ipynb`: Examples of data loading and synthetic graph generation
2. `2_Traditional_Methods.ipynb`: Traditional community detection algorithms
3. `3_GNN_Based_Methods.ipynb`: Graph Neural Network approaches
4. `4_Dynamic_GNN_Methods.ipynb`: Methods for dynamic, evolving networks
5. `5_Overlapping_Communities.ipynb`: Detecting overlapping community structures
6. `6_Comprehensive_Evaluation.ipynb`: Comparative evaluation of all methods
7. `7_UNSW_Cybersecurity_Analysis.ipynb`: Application of community detection to cybersecurity data

## Usage

To run these notebooks, first ensure all dependencies are installed:

```bash
pip install -r ../requirements.txt
pip install -e ..
```

Then start Jupyter and open the notebooks in order:

```bash
jupyter notebook
```

## Data

Notebook data should be saved to the `data/` directory. This directory is excluded from version control (via `.gitignore`), but its structure is maintained with a `.gitkeep` file.

When running the notebooks, the data is generated in the first notebook (`1_Data_Preperation.ipynb`) and then used by subsequent notebooks. If you're starting from a notebook other than the first, you may need to run the data preparation notebook first.