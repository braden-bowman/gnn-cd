#!/bin/bash
# Run the complete UNSW-NB15 analysis pipeline

# Get the directory of this script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR"

echo "=== UNSW-NB15 Cybersecurity Analysis ==="
echo ""

# Check for dataset files
if [ ! -f "UNSW-NB15_1.csv" ] || [ ! -f "UNSW-NB15_features.csv" ]; then
    echo "Dataset files not found. Creating synthetic data for testing..."
    python download_unsw.py --synthetic
fi

# Process the dataset
echo ""
echo "Step 1/3: Processing dataset and constructing graph..."
python process_unsw.py

# Run community detection evaluation
echo ""
echo "Step 2/3: Evaluating community detection methods..."
python evaluate_community_detection.py

# Open notebook (if running interactively)
echo ""
echo "Step 3/3: Analysis complete!"
echo "Results are saved in the 'results' directory."
echo ""
echo "To view the analysis in a notebook, run:"
echo "jupyter notebook ../notebooks/7_UNSW_Cybersecurity_Analysis.ipynb"