import subprocess
import os

models = ["LSTM", "GRU", "BiLSTM", "Transformer", "Informer", "Autoformer", "CNN-LSTM", "TCN", "xLSTM"]
dataset = "UNSW-NB15"

for model in models:
    print(f"Running {model} for {dataset}...")
    cmd = ["python", f"{dataset}/train.py", "--model", model, "--epochs", "1"]
    try:
        subprocess.run(cmd, check=True)
        print(f"Success: {model}")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {model} with error {e}")

print("All individual tests finished for UNSW-NB15.")
