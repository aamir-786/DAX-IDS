import numpy as np
import os

base_dir = 'CIC-IDS2017'
X = np.load(os.path.join(base_dir, 'X.npy'))
y = np.load(os.path.join(base_dir, 'y.npy'))

n = len(X)
SUBSET_SIZE = 500000 
if n > SUBSET_SIZE:
    X = X[:SUBSET_SIZE]
    y = y[:SUBSET_SIZE]
    n = SUBSET_SIZE

WINDOW_SIZE = 30
all_indices = np.arange(n - WINDOW_SIZE + 1)
np.random.seed(42)
np.random.shuffle(all_indices)

num_windows = len(all_indices)
train_end = int(0.6 * num_windows)
val_end = int(0.8 * num_windows)

train_indices = all_indices[:train_end]
val_indices = all_indices[train_end:val_end]
test_indices = all_indices[val_end:]

def get_ratio(indices, y, ws):
    pos = 0
    for idx in indices:
        if y[idx + ws - 1] == 1:
            pos += 1
    return pos / len(indices)

print(f"New Train Pos Ratio: {get_ratio(train_indices, y, WINDOW_SIZE):.4%}")
print(f"New Val Pos Ratio: {get_ratio(val_indices, y, WINDOW_SIZE):.4%}")
print(f"New Test Pos Ratio: {get_ratio(test_indices, y, WINDOW_SIZE):.4%}")
