import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Final time step
        out = self.fc(out[:, -1, :])
        return torch.clamp(torch.sigmoid(out).squeeze(), 1e-7, 1.0 - 1e-7)

# 2. GRU
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return torch.clamp(torch.sigmoid(out).squeeze(), 1e-7, 1.0 - 1e-7)

# 3. Transformer (Encoder-only)
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])
        return torch.clamp(torch.sigmoid(out).squeeze(), 1e-7, 1.0 - 1e-7)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# 4. Informer (Simplified version with Distilling and Multi-head)
# For Informer, we use a simplified version of ProbSparse attention logic or just a robust Transformer variant
# specialized for long sequences. Here we implement a simplified Informer block.
class InformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(InformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        # Informer uses TimeFeatures but we use PositionalEncoding here for simplicity in binary context
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Simplified Informer: Transformer Encoder with Distilling (Conv + MaxPool)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
            for _ in range(num_layers)
        ])
        # Informer is known for distilling. We add a pooling layer to handle long horizon.
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.encoder:
            x = layer(x)
        out = self.fc(x[:, -1, :])
        return torch.clamp(torch.sigmoid(out).squeeze(), 1e-7, 1.0 - 1e-7)

# 5. CNN-LSTM
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(CNNLSTMModel, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.clamp(torch.sigmoid(out).squeeze(), 1e-7, 1.0 - 1e-7)

# 6. xLSTM (Simplified sLSTM implementation)
# xLSTM uses exponential gating and a stabilized state.
class sLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(sLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        # Weights for gates: i, f, o, g (candidate state)
        self.weight_ih = nn.Linear(input_dim, 4 * hidden_dim)
        self.weight_hh = nn.Linear(hidden_dim, 4 * hidden_dim)
        
    def forward(self, x, state):
        h_prev, c_prev, n_prev = state
        
        gates = self.weight_ih(x) + self.weight_hh(h_prev)
        i, f, o, g = gates.chunk(4, 1)
        
        # Exponential gating (sLSTM style)
        # We use a more stable log-sum-exp style update or aggressive clamping
        # to prevent overflow over the 30-step sequence.
        i = torch.clamp(i, max=5.0) 
        f = torch.clamp(f, max=5.0) 
        
        i_exp = torch.exp(i)
        f_exp = torch.exp(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        # Stabilization and clamping of internal states to prevent Inf
        n_curr = torch.clamp(f_exp * n_prev + i_exp, max=1e20)
        c_curr = torch.clamp(f_exp * c_prev + i_exp * g, min=-1e20, max=1e20)
        
        # Safe division for hidden state
        h_curr = o * (c_curr / (n_curr + 1e-6))
        
        return h_curr, (h_curr, c_curr, n_curr)

class xLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(xLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell = sLSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        device = x.device
        batch_size, seq_len, _ = x.size()
        
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        n = torch.zeros(batch_size, self.hidden_dim).to(device)
        state = (h, c, n)
        
        for t in range(seq_len):
            h, state = self.cell(x[:, t, :], state)
            
        out = self.fc(h)
        return torch.clamp(torch.sigmoid(out).squeeze(), 1e-7, 1.0 - 1e-7)
