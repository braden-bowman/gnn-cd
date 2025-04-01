#!/bin/bash

# Install required libraries
echo "Installing required libraries..."
pip install -e ".[all]"

# Install the karateclub library for BigCLAM implementation
echo "Installing karateclub..."
pip install karateclub

# Install NiceGUI for the web frontend
echo "Installing NiceGUI..."
pip install "nicegui>=1.3.0"

# Check for correct parameter names in cdlib
echo "Checking cdlib parameters..."
python -c "
import inspect
from cdlib.algorithms import demon
print('DEMON parameters:', inspect.signature(demon))
"

echo "Making run_gui.py executable..."
chmod +x run_gui.py

echo "Installation complete. You can now run the notebooks and GUI:"
echo "- For notebooks: jupyter notebook notebooks/"
echo "- For GUI: python run_gui.py"
echo "  or: ./run_gui.py"