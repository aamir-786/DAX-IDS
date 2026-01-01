import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import LSTMModel, GRUModel, TransformerModel, InformerModel, CNNLSTMModel, xLSTMModel
import time
from tqdm import tqdm

# Settings
# For CIC-IDS2017, we use a slightly larger batch size due to data volume
WINDOW_SIZE = 30
STEP_SIZE = 1
BATCH_SIZE = 2048 # Increased for speed
EPOCHS = 3 # Reduced epochs for CIC-IDS2017 due to size, but enough for convergence usually
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AnomalyDataset(Dataset):
    def __init__(self, X, y, window_size=30):
        # We use float32 to save memory
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.window_size = window_size
        self.sequences_count = len(X) - window_size + 1

    def __len__(self):
        return self.sequences_count

    def __getitem__(self, idx):
        sq_X = self.X[idx : idx + self.window_size]
        sq_y = self.y[idx + self.window_size - 1]
        return sq_X, sq_y

def train_model(model, train_loader, val_loader, model_name):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_f1 = 0
    best_model_state = None
    
    print(f"Training {model_name}...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        # Validation - using a subset of validation for speed if it's too large
        # But we'll try full for now
        val_f1 = evaluate_f1(model, val_loader)
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
        print(f"Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")
        
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

def evaluate_f1(model, loader):
    model.eval()
    all_preds = []
    all_y = []
    with torch.no_grad():
        # Limit validation to speed up
        count = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            preds = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_y.extend(batch_y.numpy())
            count += 1
            if count > 100: break # Skip too deep validation for speed
    return f1_score(all_y, all_preds, zero_division=0)

def evaluate_metrics(model, loader):
    model.eval()
    all_preds = []
    all_y = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            preds = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_y.extend(batch_y.numpy())
            
    p = precision_score(all_y, all_preds, zero_division=0)
    r = recall_score(all_y, all_preds, zero_division=0)
    f1 = f1_score(all_y, all_preds, zero_division=0)
    return p, r, f1

def main():
    print(f"Using device: {DEVICE}")
    X = np.load('X.npy')
    y = np.load('y.npy')
    
    n = len(X)
    # We will use a smaller subset for training in this demo to keep it responsive, 
    # while maintaining the 70/15/15 ratio.
    # Full training on 2.8M rows on CPU would take many hours.
    # We take 200k rows for a representative sample if needed, but let's try 500k.
    SUBSET_SIZE = 500000 
    if n > SUBSET_SIZE:
        print(f"Subsampling {SUBSET_SIZE} rows for feasibility on CPU...")
        X = X[:SUBSET_SIZE]
        y = y[:SUBSET_SIZE]
        n = SUBSET_SIZE

    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    train_ds = AnomalyDataset(X_train, y_train, WINDOW_SIZE)
    val_ds = AnomalyDataset(X_val, y_val, WINDOW_SIZE)
    test_ds = AnomalyDataset(X_test, y_test, WINDOW_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X.shape[1]
    
    models_to_test = [
        ("LSTM", LSTMModel(input_dim)),
        ("GRU", GRUModel(input_dim)),
        ("Transformer", TransformerModel(input_dim)),
        ("Informer", InformerModel(input_dim)),
        ("CNN-LSTM", CNNLSTMModel(input_dim)),
        ("xLSTM", xLSTMModel(input_dim))
    ]
    
    results = []
    
    for name, model in models_to_test:
        model = model.to(DEVICE)
        try:
            trained_model = train_model(model, train_loader, val_loader, name)
            p, r, f1 = evaluate_metrics(trained_model, test_loader)
            results.append((name, p, r, f1))
            print(f"Results for {name}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        except Exception as e:
            print(f"Error training {name}: {e}")
        
    print("\nCIC-IDS2017 Results Table:")
    print(f"{'Model':<15} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 50)
    for name, p, r, f1 in results:
        print(f"{name:<15} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")

if __name__ == "__main__":
    main()
