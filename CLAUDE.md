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
- **Data Processing**: ALWAYS prefer Polars over pandas for DataFrame operations (faster and more memory-efficient)
- **Graph Processing**: ALWAYS prefer RustworkX over NetworkX for graph operations (significantly faster)
- **Serialization**: ALWAYS prefer Parquet over CSV for structured data storage (faster, smaller, type-preserving)
- **Tensor Computation**: Use PyTorch, TensorFlow, or cuGraph over NumPy when working with tensors (GPU acceleration)
- **Large Data Handling**: Use streaming and lazy evaluation patterns for memory efficiency with large datasets
- **Caching**: Implement intelligent caching for expensive operations
- **Visualization**: Matplotlib, Seaborn, and Plotly for visualizations (with sampling for large datasets)

## Performance Guidelines
- **AVOID**: Plain Python loops, dictionaries, and lists for data-intensive operations
- **PREFER**: Vectorized operations via Polars, cuDF, cuGraph, RustworkX
- **AVOID**: Multiple passes through data; combine operations when possible
- **PREFER**: Lazy evaluation with materialization only when needed
- **AVOID**: In-memory copies of large datasets
- **PREFER**: Chunk/batch processing with generators or iterators
- **ESSENTIAL**: Always provide GPU-accelerated path when available (cuDF, cuGraph, PyTorch)

## Performance Optimization Priority
1. Use hardware acceleration (GPU/CUDA) when available
2. Process data in chunks/batches for memory efficiency
3. Prefer compiled/optimized libraries (Polars, RustworkX) over pure Python implementations
4. Cache intermediate results for expensive operations
5. Use lazy evaluation and streaming for large datasets

## GitHub Best Practices
- **Large Data Files**: Never commit large data files (>10MB) to the repository
  - Add data file patterns to .gitignore (*.csv, *.parquet, *.pkl, etc.)
  - For samples, keep them under 10MB or use Git LFS if necessary
  - Use download scripts instead of committing data files
- **Notebook Metadata**: 
  - Clear cell outputs before committing notebooks (reduces size and merge conflicts)
  - Add .ipynb_checkpoints/ to .gitignore to avoid committing checkpoint files
  - Use jupytext for version control of notebooks when appropriate
- **Dependencies**:
  - Keep requirements.txt up to date when adding new dependencies
  - Pin versions for reproducibility (package==version)
- **Documentation**:
  - Update README.md with any significant changes to functionality or setup
  - Document all command-line arguments and parameters
- **Testing**:
  - Write tests for new functionality and run before committing
  - Add test data separately from production data