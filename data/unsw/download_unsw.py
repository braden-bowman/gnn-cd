#!/usr/bin/env python3
# Helper script to download and extract the UNSW-NB15 dataset

import os
import argparse
import requests
import zipfile
import io
import sys
from tqdm import tqdm

# Define paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# URLs for dataset files
# Note: UNSW-NB15 dataset requires registration, so direct download links aren't provided
# The script will guide users to the official download page
DOWNLOAD_PAGE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"

def print_download_instructions():
    """Print instructions for manual download"""
    print(f"\nUNSW-NB15 Dataset Download Instructions")
    print(f"=======================================")
    print(f"The UNSW-NB15 dataset requires registration and manual download.")
    print(f"Please follow these steps:")
    print(f"1. Visit: {DOWNLOAD_PAGE}")
    print(f"2. Fill out the registration form to request access")
    print(f"3. Download the following files:")
    print(f"   - UNSW-NB15_features.csv")
    print(f"   - UNSW-NB15_1.csv (and optionally parts 2-4)")
    print(f"   - UNSW-NB15_GT.csv (ground truth labels)")
    print(f"4. Place the downloaded files in this directory:")
    print(f"   {DATA_DIR}")
    print(f"\nAfter downloading, you can run the processing scripts:")
    print(f"python process_unsw.py")
    print(f"python evaluate_community_detection.py")

def check_existing_files():
    """Check if any dataset files already exist"""
    expected_files = [
        "UNSW-NB15_features.csv",
        "UNSW-NB15_1.csv",
        "UNSW-NB15_2.csv",
        "UNSW-NB15_3.csv", 
        "UNSW-NB15_4.csv",
        "UNSW-NB15_GT.csv"
    ]
    
    existing = [f for f in expected_files if os.path.exists(os.path.join(DATA_DIR, f))]
    
    if existing:
        print(f"\nFound {len(existing)} existing dataset files:")
        for f in existing:
            file_path = os.path.join(DATA_DIR, f)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")
    else:
        print(f"\nNo existing dataset files found in {DATA_DIR}")
    
    return existing

def create_synthetic_data(size=1000):
    """Create synthetic data for testing when real data isn't available"""
    import numpy as np
    import pandas as pd
    
    print(f"\nCreating synthetic data for testing ({size} records)...")
    
    # Generate synthetic features data
    feature_names = [
        "id", "name", "category", "type", "description"
    ]
    feature_rows = []
    for i in range(49):
        feature_rows.append({
            "id": i,
            "name": f"feature_{i}",
            "category": np.random.choice(["basic", "content", "time", "additional"]),
            "type": np.random.choice(["nominal", "integer", "float", "binary"]),
            "description": f"Synthetic feature {i} for testing"
        })
    
    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(os.path.join(DATA_DIR, "UNSW-NB15_features.csv"), index=False)
    print(f"  Created UNSW-NB15_features.csv with {len(features_df)} features")
    
    # Generate synthetic network traffic data
    np.random.seed(42)
    
    # Create columns similar to UNSW-NB15
    columns = [
        "srcip", "sport", "dstip", "dsport", "proto",
        "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
        "sloss", "dloss", "service", "sload", "dload",
        "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
        "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
        "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt",
        "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
        "ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd",
        "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "attack_cat", "label"
    ]
    
    # Generate data
    data = []
    for i in range(size):
        # Generate IPs with more repeats to create community structure
        src_ip = f"192.168.1.{np.random.randint(1, 50)}"
        dst_ip = f"10.0.0.{np.random.randint(1, 50)}"
        
        # Generate an attack with 20% probability
        is_attack = np.random.random() < 0.2
        attack_cat = np.random.choice(["DoS", "Exploits", "Reconnaissance", "Fuzzers", "Worms"]) if is_attack else "Normal"
        
        # Generate features with some correlation to attack type
        row = {
            "srcip": src_ip,
            "sport": np.random.randint(1024, 65535),
            "dstip": dst_ip,
            "dsport": np.random.randint(1, 1024),  # Lower ports for servers
            "proto": np.random.choice(["tcp", "udp", "icmp"]),
            "state": np.random.choice(["FIN", "CON", "REQ"]),
            "dur": np.random.exponential(10) * (3 if is_attack else 1),  # Attacks may have longer duration
            "sbytes": np.random.randint(10, 10000) * (5 if is_attack else 1),  # Attacks may send more bytes
            "dbytes": np.random.randint(10, 5000),
            "label": 1 if is_attack else 0,
            "attack_cat": attack_cat
        }
        
        # Fill in remaining columns with random values
        for col in columns:
            if col not in row:
                row[col] = np.random.random() * 100
        
        data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(DATA_DIR, "UNSW-NB15_1.csv"), index=False)
    print(f"  Created UNSW-NB15_1.csv with {len(df)} records")
    
    # Create simple ground truth
    gt_df = pd.DataFrame({
        "id": range(size),
        "label": df["label"],
        "attack_cat": df["attack_cat"]
    })
    gt_df.to_csv(os.path.join(DATA_DIR, "UNSW-NB15_GT.csv"), index=False)
    print(f"  Created UNSW-NB15_GT.csv with ground truth labels")
    
    print("Synthetic data creation complete. You can now run the processing scripts.")
    return ["UNSW-NB15_features.csv", "UNSW-NB15_1.csv", "UNSW-NB15_GT.csv"]

def main():
    parser = argparse.ArgumentParser(description="Helper script for UNSW-NB15 dataset")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic data for testing")
    parser.add_argument("--size", type=int, default=1000, help="Size of synthetic dataset (default: 1000)")
    args = parser.parse_args()
    
    # Check for existing files
    existing_files = check_existing_files()
    
    if args.synthetic:
        synthetic_files = create_synthetic_data(args.size)
    else:
        # Print instructions for manual download
        print_download_instructions()
    
    print("\nNext steps:")
    print("1. Run data processing:   python process_unsw.py")
    print("2. Run analysis:          python evaluate_community_detection.py")
    print("3. Open the notebook:     jupyter notebook ../notebooks/7_UNSW_Cybersecurity_Analysis.ipynb")

if __name__ == "__main__":
    main()