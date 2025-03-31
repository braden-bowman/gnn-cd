# GNN-CD Framework Guidelines

## Setup & Commands
```bash
# Installation
pip install -r requirements.txt
pip install -e ".[all]"  # Install with all extras

# Running notebooks
jupyter notebook notebooks/X_Notebook.ipynb

# Testing
pytest                    # Run all tests
pytest test_file.py       # Run specific test file
```

## Code Style Guidelines
- **Imports**: Group imports by category; use parenthesized multi-line imports
- **Formatting**: 4-space indentation; line continuation with parentheses
- **Types**: Use typing annotations (Dict, List, Optional, Tuple, Union, Any)
- **Error Handling**: Use try/except blocks for optional dependencies
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Documentation**: Document functions with parameters and return values
- **Code Structure**: Modular organization with clear separation of concerns
- **Package Structure**: Use proper package structure with __init__.py exports

## Module Standards
- Manage optional dependencies with try/except blocks
- Check for hardware acceleration where appropriate (GPU/CUDA)
- Include descriptive docstrings with type information
- Return values should have consistent formats across related functions

## Technology Stack
- **Data Processing**: Use Polars (not pandas) for DataFrame operations
- **Machine Learning**: PyTorch and TensorFlow for neural networks
- **Graph Processing**: RustworkX for graph operations
- **Serialization**: Parquet for structured data storage
- **Visualization**: Matplotlib, Seaborn, and Plotly for visualizations