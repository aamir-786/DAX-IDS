import pandas as pd
import numpy as np
import os

def check_overlap():
    unsw_path = 'd:/Fiverr/AI/datasets/UNSW-NB15/UNSW_NB15_training-set.parquet'
    cic_path = 'd:/Fiverr/AI/datasets/CIC-IDS2017/Monday-WorkingHours.pcap_ISCX.csv'
    
    if not os.path.exists(unsw_path) or not os.path.exists(cic_path):
        print("Dataset files not found for overlap check.")
        return

    unsw_cols = pd.read_parquet(unsw_path).columns.tolist()
    cic_cols = pd.read_csv(cic_path, nrows=1).columns.tolist()
    cic_cols = [c.strip() for c in cic_cols] # Clean spaces
    
    intersection = set(unsw_cols).intersection(set(cic_cols))
    
    print(f"UNSW-NB15 Columns: {len(unsw_cols)}")
    print(f"CIC-IDS2017 Columns: {len(cic_cols)}")
    print(f"Exact Name Intersection: {intersection}")
    
    # Semantic mapping suggestion
    mapping = {
        'dur': 'Flow Duration',
        'spkts': 'Total Fwd Packets',
        'dpkts': 'Total Backward Packets',
        'sbytes': 'Total Length of Fwd Packets',
        'dbytes': 'Total Length of Bwd Packets',
        'rate': 'Flow Packets/s',
        'smean': 'Fwd Packet Length Mean',
        'dmean': 'Bwd Packet Length Mean'
    }
    
    common_semantics = [k for k in mapping.keys() if k in unsw_cols and mapping[k] in cic_cols]
    print(f"Semantic Overlap Found: {common_semantics}")
    
    if len(intersection) == 0 and len(common_semantics) > 0:
        print("\nDecision: Datasets have different column names but shared underlying semantics.")
        print("To perform cross-dataset testing, we would need to:")
        print("1. Map features using a common schema (e.g. use the 8 common semantic features).")
        print("2. Retrain models on only these common features.")
        print("3. Test across datasets.")
    elif len(intersection) > 0:
        print("\nDecision: Datasets share columns. Direct cross-testing may be possible on these columns.")
    else:
        print("\nDecision: No significant overlap found. Cross-dataset testing is not recommended without heavy feature engineering.")

if __name__ == "__main__":
    check_overlap()
