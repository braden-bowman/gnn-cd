#!/usr/bin/env python
"""
GNN-CD GUI Launcher

This script launches the NiceGUI-based frontend for the GNN-CD framework,
allowing users to load network data, run community detection algorithms,
and visualize the results through a web interface.

Usage:
    python run_gui.py [--host HOST] [--port PORT] [--debug]

Options:
    --host HOST     Host address to bind to (default: 0.0.0.0)
    --port PORT     Port to run the GUI on (default: 8080)
    --debug         Run in debug mode with additional logging

Example:
    python run_gui.py --host localhost --port 8088 --debug
"""

import argparse
from community_detection.gui import run_gui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the GNN-CD GUI")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, 
                        help="Port to run the GUI on (default: 8080)")
    parser.add_argument("--debug", action="store_true", 
                        help="Run in debug mode with additional logging")
    
    args = parser.parse_args()
    
    print(f"Starting GNN-CD GUI on {args.host}:{args.port} (debug mode: {args.debug})")
    print("Once started, access the GUI by opening your browser to:")
    print(f"  http://localhost:{args.port}/ (if running locally)")
    print(f"  http://<your-ip-address>:{args.port}/ (if connecting from another machine)")
    print("\nPress Ctrl+C to stop the server\n")
    
    run_gui(host=args.host, port=args.port, debug=args.debug)