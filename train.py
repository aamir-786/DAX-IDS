import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from models import LSTMModel, GRUModel, TransformerModel, InformerModel, CNNLSTMModel, xLSTMModel
import time
from tqdm import tqdm

# Settings
WINDOW_SIZE = 30
STEP_SIZE = 1
BATCH_SIZE = 512
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AnomalyDataset(Dataset):
    def __init__(self, X, y, window_size=30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window_size = window_size
        self.sequences_count = len(X) - window_size + 1

    def __len__(self):
        return self.sequences_count

    def __getitem__(self, idx):
        # Sequence
        sq_X = self.X[idx : idx + self.window_size]
        # Label is the label of the last point in the sequence
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
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        val_f1 = evaluate_f1(model, val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            
        print(f"Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")
        
    model.load_state_dict(best_model_state)
    return model

def evaluate_f1(model, loader):
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
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    # Split the original tabular data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    # Datasets
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
        trained_model = train_model(model, train_loader, val_loader, name)
        p, r, f1 = evaluate_metrics(trained_model, test_loader)
        results.append((name, p, r, f1))
        print(f"Results for {name}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
        
    print("\nFinal Results Table:")
    print(f"{'Model':<15} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 50)
    for name, p, r, f1 in results:
        print(f"{name:<15} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")

if __name__ == "__main__":
    main()
