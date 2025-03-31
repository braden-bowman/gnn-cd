# Contributing to GNN Community Detection Framework

Thank you for your interest in contributing to this project! Here are some guidelines to help you get started.

## Code of Conduct

Please be respectful and inclusive in your interactions with other contributors.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature/bugfix
3. Make your changes
4. Write or update tests if necessary
5. Ensure the code lints correctly
6. Submit a pull request

## Data Management

**IMPORTANT**: This project deals with potentially large datasets. Please follow these guidelines for data management:

### DO NOT Commit:
- Large data files (CSV, Parquet, etc.)
- Model checkpoints or saved models (.pt files)
- Processed cache files
- Large visualization outputs
- UNSW-NB15 or other large dataset files
- Any file larger than 10MB

### What to Commit:
- Code changes (Python scripts, notebooks)
- Documentation updates
- Small sample data for testing in `data/samples/` directory
- Small example synthetic graphs in `notebooks/data/` directory
- Configuration files

### Handling Data in PRs:
- Use the sample data files for tests and examples
- Document how to obtain and process the real datasets
- Use `.gitignore` to prevent accidental commits of large files

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

3. Run tests:
   ```bash
   pytest
   ```

## Style Guidelines

- Follow PEP 8 for Python code
- Use descriptive variable names
- Document public functions and classes
- Use type hints where appropriate
- Keep line length to a maximum of 100 characters

## Submitting Changes

1. Push your changes to your fork
2. Submit a pull request against the `main` branch
3. Describe your changes in detail:
   - What problem does it solve?
   - How does it solve the problem?
   - Are there any side effects?
   - Include screenshot/examples if applicable

## Questions?

Feel free to open an issue if you have questions about contributing.

Thank you for helping improve this project!