import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def preprocess():
    print("Loading data...")
    train_path = '../datasets/UNSW-NB15/UNSW_NB15_training-set.parquet'
    test_path = '../datasets/UNSW-NB15/UNSW_NB15_testing-set.parquet'
    
    df1 = pd.read_parquet(train_path)
    df2 = pd.read_parquet(test_path)
    df = pd.concat([df1, df2], ignore_index=True)
    
    print("Converting labels...")
    # Normal -> 0
    # DoS, Exploits, Worms, Backdoors -> 1
    # We map "Normal" to 0 and all others to 1 to maintain binary anomaly detection
    # following the hint that DoS, Exploits etc are 1.
    df['target'] = df['attack_cat'].apply(lambda x: 0 if x.strip() == 'Normal' else 1)
    
    # Drop IDs, IP addresses, timestamps
    # Based on column inspection, none of the columns in this version are IPs or Timestamps.
    # 'id' is often present in CSVs but not in this parquet.
    cols_to_drop = ['attack_cat', 'label']
    if 'id' in df.columns:
        cols_to_drop.append('id')
    
    df = df.drop(columns=cols_to_drop)
    
    print("Encoding categorical features...")
    cat_cols = ['proto', 'service', 'state']
    df = pd.get_dummies(df, columns=cat_cols)
    
    print("Standardizing numerical features...")
    num_cols = [c for c in df.columns if c not in ['target'] and not any(c.startswith(cat+'_') for cat in cat_cols)]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    X = df.drop(columns=['target']).values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    
    print(f"Final feature count: {X.shape[1]}")
    np.save('X.npy', X)
    np.save('y.npy', y)
    print(f"Preprocessed data saved. X shape: {X.shape}, y shape: {y.shape}")

if __name__ == "__main__":
    preprocess()
