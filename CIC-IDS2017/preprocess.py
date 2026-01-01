import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import glob

def preprocess():
    print("Loading CIC-IDS2017 data...")
    csv_files = glob.glob('../datasets/CIC-IDS2017/*.csv')
    df_list = []
    for f in csv_files:
        print(f"Reading {f}...")
        # Some files might have different encodings or issues
        df_temp = pd.read_csv(f)
        df_list.append(df_temp)
    
    df = pd.concat(df_list, ignore_index=True)
    
    print("Cleaning column names...")
    df.columns = df.columns.str.strip()
    
    print("Handling NaNs and Infs...")
    # Replacing inf with NaN then dropping
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print("Converting labels...")
    # Benign -> 0, Others -> 1
    df['target'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    print("Dropping requested columns...")
    # Common columns in CIC-IDS2017 that fit the 'IDs, IPs, timestamps' rule
    cols_to_drop = ['Label', 'Destination Port']
    # Check if IP/Timestamp columns exist (some versions have them)
    potential_drops = ['Source IP', 'Destination IP', 'Timestamp', 'External IP']
    for p in potential_drops:
        if p in df.columns:
            cols_to_drop.append(p)
            
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    print("Encoding categorical features...")
    # CIC-IDS2017 is mostly numeric, but let's check for object types
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'target' in cat_cols: cat_cols.remove('target')
    if cat_cols:
        print(f"Categorical columns found: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols)
    
    print("Standardizing numerical features...")
    num_cols = df.columns.drop('target')
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Cast to float32 for efficiency
    X = df.drop(columns=['target']).values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    
    print(f"Final feature count: {X.shape[1]}")
    np.save('X.npy', X)
    np.save('y.npy', y)
    print(f"Preprocessed data saved. X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    preprocess()
