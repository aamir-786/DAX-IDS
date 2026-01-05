import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix
import sys
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import LSTMModel, GRUModel, TransformerModel, InformerModel, CNNLSTMModel, xLSTMModel, BiLSTMModel, TCNModel, AutoformerModel
import argparse

# Settings
WINDOW_SIZE = 30
STEP_SIZE = 1
BATCH_SIZE = 2048 # Larger for CIC-IDS2017
EPOCHS = 3 # Fewer epochs due to size
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AnomalyDataset(Dataset):
    def __init__(self, X, y, indices, window_size=30):
        # Use torch.from_numpy to share memory if possible
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.indices = indices
        self.window_size = window_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sq_X = self.X[real_idx : real_idx + self.window_size]
        sq_y = self.y[real_idx + self.window_size - 1]
        return sq_X, sq_y

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, val_loader, model_name, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_f1 = 0
    best_model_state = None
    
    start_time = time.time()
    print(f"Training {model_name}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        # Validation - limited for speed on CIC-IDS2017
        val_metrics = evaluate_metrics(model, val_loader, limit_batches=100)
        val_f1 = val_metrics['f1']
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
        print(f"Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")
        
    training_time = time.time() - start_time
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, training_time

def evaluate_metrics(model, loader, model_name_str="Model", save_plots=False, limit_batches=None):
    model.eval()
    all_preds = []
    all_probs = []
    all_y = []
    
    start_inf = time.time()
    batch_count = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            probs = outputs.cpu().numpy()
            preds = (outputs >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_y.extend(batch_y.numpy())
            batch_count += 1
            if limit_batches and batch_count >= limit_batches:
                break
    
    time_elapsed = time.time() - start_inf
    avg_inf_time_batch = time_elapsed / batch_count if batch_count > 0 else 0
    avg_inf_time_sample = time_elapsed / (batch_count * BATCH_SIZE) if batch_count > 0 else 0
    
    y_true = np.array(all_y)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.5
        
    pr_auc = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Save Visualizations
    if save_plots:
        save_dir = os.path.join(os.path.dirname(__file__), 'plots', model_name_str)
        os.makedirs(save_dir, exist_ok=True)
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob)
        plt.figure(); plt.plot(fpr_vals, tpr_vals, label=f'ROC (area = {roc_auc:.2f})'); plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC - {model_name_str}'); plt.legend(); plt.savefig(os.path.join(save_dir, 'roc_curve.png')); plt.close()
        
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(); plt.plot(rec_vals, prec_vals, label=f'PR (area = {pr_auc:.2f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR - {model_name_str}'); plt.legend(); plt.savefig(os.path.join(save_dir, 'pr_curve.png')); plt.close()
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'CM - {model_name_str}'); plt.savefig(os.path.join(save_dir, 'confusion_matrix.png')); plt.close()
    
    return {
        'precision': p, 'recall': r, 'f1': f1,
        'roc_auc': roc_auc, 'pr_auc': pr_auc, 'mcc': mcc, 'fpr': fpr,
        'inf_time_batch': avg_inf_time_batch,
        'inf_time_sample': avg_inf_time_sample
    }

def main():
    print(f"Using device: {DEVICE}")
    base_dir = os.path.dirname(__file__)
    X = np.load(os.path.join(base_dir, 'X.npy'))
    y = np.load(os.path.join(base_dir, 'y.npy'))
    
    n = len(X)
    # Using a subset for feasibility if it's too large, but keeping 60/20/20 split
    SUBSET_SIZE = 500000 
    if n > SUBSET_SIZE:
        print(f"Subsampling {SUBSET_SIZE} rows for feasibility...")
        X = X[:SUBSET_SIZE]
        y = y[:SUBSET_SIZE]
        n = SUBSET_SIZE

    # Create indices for all possible windows
    all_indices = np.arange(n - WINDOW_SIZE + 1)
    np.random.seed(42)
    np.random.shuffle(all_indices)
    
    num_windows = len(all_indices)
    train_end = int(0.6 * num_windows)
    val_end = int(0.8 * num_windows)
    
    train_indices = all_indices[:train_end]
    val_indices = all_indices[train_end:val_end]
    test_indices = all_indices[val_end:]
    
    print(f"Data Split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    train_ds = AnomalyDataset(X, y, train_indices, WINDOW_SIZE)
    val_ds = AnomalyDataset(X, y, val_indices, WINDOW_SIZE)
    test_ds = AnomalyDataset(X, y, test_indices, WINDOW_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X.shape[1]
    
    model_list = [
        ("xLSTM", xLSTMModel(input_dim)),
        ("LSTM", LSTMModel(input_dim)),
        ("GRU", GRUModel(input_dim)),
        ("BiLSTM", BiLSTMModel(input_dim)),
        ("Transformer", TransformerModel(input_dim)),
        ("Informer", InformerModel(input_dim)),
        ("Autoformer", AutoformerModel(input_dim)),
        ("CNN-LSTM", CNNLSTMModel(input_dim)),
        ("TCN", TCNModel(input_dim))
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all', help='Model to train (all, LSTM, GRU, etc.)')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    args = parser.parse_args()
    
    current_epochs = args.epochs
    
    if args.model != 'all':
        model_list = [m for m in model_list if m[0].lower() == args.model.lower()]
        if not model_list:
            print(f"Model {args.model} not found.")
            return

    benchmarks = []
    
    for name, model in model_list:
        model = model.to(DEVICE)
        params = count_parameters(model)
        
        try:
            # Train
            trained_model, train_time = train_model(model, train_loader, val_loader, name, current_epochs)
            
            # Evaluate
            metrics = evaluate_metrics(trained_model, test_loader, name, save_plots=True)
            
            benchmarks.append({
                'name': name,
                'params': params,
                'train_time': train_time,
                'metrics': metrics
            })
            print(f"Finished {name}. F1: {metrics['f1']:.4f}, Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, ROC: {metrics['roc_auc']:.4f}")
        except Exception as e:
            print(f"Error training {name}: {e}")

    # Tables for CIC-IDS2017
    print("\n" + "="*50)
    print("CIC-IDS2017 Results")
    print("="*50)
    
    print("\nTABLE 3: Efficiency Metrics (CIC-IDS2017)")
    print(f"{'Model':<15} | {'Params':<12} | {'Train Time (s)':<15} | {'Inf Time/Sample (ms)':<20}")
    print("-" * 70)
    for b in benchmarks:
        print(f"{b['name']:<15} | {b['params']:<12,} | {b['train_time']:<15.2f} | {b['metrics']['inf_time_sample']*1000:<20.4f}")

    print("\nTABLE 5: Statistical Metrics (CIC-IDS2017)")
    print(f"{'Model':<15} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'ROC-AUC':<10} | {'PR-AUC':<10} | {'MCC':<10} | {'FPR':<10} | {'Latency (ms)':<15}")
    print("-" * 110)
    for b in benchmarks:
        m = b['metrics']
        print(f"{b['name']:<15} | {m['precision']:<8.4f} | {m['recall']:<8.4f} | {m['f1']:<8.4f} | {m['roc_auc']:<10.4f} | {m['pr_auc']:<10.4f} | {m['mcc']:<10.4f} | {m['fpr']:<10.4f} | {m['inf_time_batch']*1000:<15.2f}")

    # Save results to JSON
    import json
    with open(os.path.join(base_dir, 'results.json'), 'w') as f:
        json.dump(benchmarks, f, indent=4)

if __name__ == "__main__":
    main()
