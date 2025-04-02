#!/usr/bin/env python
"""
Script to fix the overlapping community detection issues in 5_Overlapping_Communities.ipynb
"""

import sys
import subprocess
import os

def print_header(msg):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {msg}")
    print("=" * 80)

def main():
    print_header("Checking for required packages")
    
    try:
        import pip
        print("✅ pip is available")
    except ImportError:
        print("❌ pip is not available. Please install pip first.")
        return
    
    # Install karateclub for BigCLAM implementation
    print("\nAttempting to install karateclub...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "karateclub"])
        print("✅ karateclub installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️ Failed to install karateclub - will use built-in implementation")
    
    # Test fixing the cdlib DEMON issue
    print("\nVerifying cdlib and its algorithms...")
    try:
        from cdlib import algorithms
        print("✅ cdlib is installed")
        
        import inspect
        try:
            demon_sig = inspect.signature(algorithms.demon)
            print(f"✅ DEMON function signature: {demon_sig}")
            params = list(demon_sig.parameters.keys())
            print(f"  Parameters: {params}")
            
            if 'min_com_size' in params:
                print("  - Will use 'min_com_size' parameter")
            elif 'min_comm_size' in params:
                print("  - Will use 'min_comm_size' parameter")
            else:
                print("  - Will use just 'epsilon' parameter")
        except Exception as e:
            print(f"⚠️ Unable to inspect DEMON function: {e}")
    except ImportError:
        print("⚠️ cdlib is not installed - some overlapping community detection algorithms won't work.")
        print("  You can install it with: pip install cdlib")
    
    # Check if we need to update the module
    print("\nChecking module implementations...")
    try:
        from community_detection.overlapping_community_detection import run_bigclam, run_demon, run_slpa
        print("✅ All required functions are present in the module")
    except ImportError as e:
        print(f"⚠️ Error importing from module: {e}")
    
    print_header("NEXT STEPS")
    print("""
1. Run the notebook 5_Overlapping_Communities.ipynb
2. If you encounter errors with the DEMON algorithm, it will now try multiple parameter
   combinations to find the one that works with your version of cdlib.
3. For BigCLAM, the implementation will use:
   a) The built-in module implementation first
   b) karateclub library implementation if available
   c) Direct implementation as a fallback

The fixes have been applied directly to the module. You don't need to
modify the notebook directly.

If you still encounter issues, please create a github issue with the error details.
""")

if __name__ == "__main__":
    main()