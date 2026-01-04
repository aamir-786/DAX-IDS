import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results():
    results_unsw = {}
    results_cic = {}
    
    if os.path.exists('UNSW-NB15/results.json'):
        with open('UNSW-NB15/results.json', 'r') as f:
            data = json.load(f)
            for item in data:
                results_unsw[item['name']] = item
                
    if os.path.exists('CIC-IDS2017/results.json'):
        with open('CIC-IDS2017/results.json', 'r') as f:
            data = json.load(f)
            for item in data:
                results_cic[item['name']] = item
    return results_unsw, results_cic

def plot_metric(results_unsw, results_cic, metric_path, title, ylabel, filename, is_nested=True):
    models = sorted(list(set(list(results_unsw.keys()) + list(results_cic.keys()))))
    
    def get_val(results, model):
        item = results.get(model)
        if not item: return 0
        if is_nested:
            parts = metric_path.split('.')
            val = item
            for p in parts:
                val = val.get(p, 0)
            return val
        return item.get(metric_path, 0)

    unsw_vals = [get_val(results_unsw, m) for m in models]
    cic_vals = [get_val(results_cic, m) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, unsw_vals, width, label='UNSW-NB15', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, cic_vals, width, label='CIC-IDS2017', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def main():
    unsw, cic = load_results()
    if not unsw and not cic:
        print("No results found.")
        return
        
    # 1. F1 Score Comparison
    plot_metric(unsw, cic, 'metrics.f1', 'Model Comparison: F1-Score', 'F1 Score', 'comparison_f1.png')
    
    # 2. ROC-AUC Comparison
    plot_metric(unsw, cic, 'metrics.roc_auc', 'Model Comparison: ROC-AUC', 'ROC-AUC', 'comparison_roc.png')
    
    # 3. Training Time Comparison
    plot_metric(unsw, cic, 'train_time', 'Model Comparison: Training Time (s)', 'Time (seconds)', 'comparison_time.png', is_nested=False)

if __name__ == "__main__":
    main()
