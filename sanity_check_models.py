import torch
from models import (LSTMModel, GRUModel, TransformerModel, InformerModel, 
                    CNNLSTMModel, xLSTMModel, BiLSTMModel, TCNModel, AutoformerModel)

def test_models():
    batch_size = 8
    seq_len = 30
    input_dim = 42 # Example for UNSW-NB15
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    models = [
        ("LSTM", LSTMModel(input_dim)),
        ("GRU", GRUModel(input_dim)),
        ("Transformer", TransformerModel(input_dim)),
        ("Informer", InformerModel(input_dim)),
        ("CNNLSTM", CNNLSTMModel(input_dim)),
        ("xLSTM", xLSTMModel(input_dim)),
        ("BiLSTM", BiLSTMModel(input_dim)),
        ("TCN", TCNModel(input_dim)),
        ("Autoformer", AutoformerModel(input_dim))
    ]
    
    for name, model in models:
        print(f"Testing {name}...")
        try:
            out = model(x)
            print(f"  Output shape: {out.shape}")
            if out.shape != (batch_size,):
                print(f"  [WARNING] Expected output shape ({batch_size},), got {out.shape}")
        except Exception as e:
            print(f"  [ERROR] {name} failed: {e}")

if __name__ == "__main__":
    test_models()
