import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison_metrics():
    # Load results
    unsw_path = 'UNSW-NB15/results.json'
    cic_path = 'CIC-IDS2017/results.json'
    
    if not os.path.exists(unsw_path) or not os.path.exists(cic_path):
        print("Results files not found. Generate them by running train.py first.")
        return

    with open(unsw_path, 'r') as f:
        unsw_results = json.load(f)
    with open(cic_path, 'r') as f:
        cic_results = json.load(f)

    models = [r['name'] for r in unsw_results]
    
    # Metrics to plot
    metrics = ['precision', 'recall', 'f1']
    titles = ['Precision', 'Recall', 'F1-Score']
    filenames = ['comparison_precision.png', 'comparison_recall.png', 'comparison_f1.png']

    for metric, title, filename in zip(metrics, titles, filenames):
        unsw_vals = [r['metrics'][metric] for r in unsw_results]
        cic_vals = [r['metrics'][metric] for r in cic_results]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, unsw_vals, width, label='UNSW-NB15', color='#3498db')
        rects2 = ax.bar(x + width/2, cic_vals, width, label='CIC-IDS2017', color='#e74c3c')

        ax.set_ylabel('Score')
        ax.set_title(f'Model Comparison: {title}')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1.1)

        # Add value labels
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.close()

if __name__ == "__main__":
    plot_comparison_metrics()
