import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '.')))
from models import LSTMModel, GRUModel, TransformerModel, InformerModel, CNNLSTMModel, xLSTMModel, BiLSTMModel, TCNModel, AutoformerModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

input_dim = 188
models = [
    ("LSTM", LSTMModel(input_dim)),
    ("GRU", GRUModel(input_dim)),
    ("BiLSTM", BiLSTMModel(input_dim)),
    ("Transformer", TransformerModel(input_dim)),
    ("Informer", InformerModel(input_dim)),
    ("Autoformer", AutoformerModel(input_dim)),
    ("CNN-LSTM", CNNLSTMModel(input_dim)),
    ("TCN", TCNModel(input_dim)),
    ("xLSTM", xLSTMModel(input_dim))
]

for name, model in models:
    print(f"{name}: {count_parameters(model):,}")
