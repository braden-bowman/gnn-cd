#\!/usr/bin/env python

import json
import sys

def update_notebook_cell(notebook_path, cell_num, new_content):
    """Update a specific cell in a Jupyter notebook."""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Update the specified cell
    notebook['cells'][cell_num]['source'] = new_content.splitlines(True)
    
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)

# Read the fixes
with open('/home/braden/gnn-cd/temp/bigclam_fix.py', 'r') as f:
    bigclam_fix = f.read()

with open('/home/braden/gnn-cd/temp/demon_fix.py', 'r') as f:
    demon_fix = f.read()

# Update the notebook
notebook_path = '/home/braden/gnn-cd/notebooks/5_Overlapping_Communities.ipynb'

print("Updating BigCLAM function in cell 5...")
# Find bigclam function in cell 5
update_notebook_cell(notebook_path, 5, bigclam_fix)

print("Updating DEMON function in cell 5...")
# Find demon function in cell 5 
update_notebook_cell(notebook_path, 5, demon_fix)

print("Notebook updated successfully\!")
