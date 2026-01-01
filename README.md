# UNSW-NB15 Anomaly Detection with Long-Horizon Sequences

This project implements and evaluates multiple deep learning models for binary
network anomaly detection using the UNSW-NB15 dataset. The goal is to compare
standard architectures like LSTM, GRU, Transformers, and CNN-LSTMs with the
**xLSTM (extended memory LSTM)** model.

## Dataset

- **Name**: UNSW-NB15
- **Task**: Binary Anomaly Detection (Normal vs. Attack)
- **Labeling**:
  - Normal → 0
  - Attacks (DoS, Exploits, Worms, Backdoors, etc.) → 1
- **Sequence Length (Window Size)**: 30
- **Step Size**: 1

## Feature Preprocessing

- Dropped IDs, IP addresses, and timestamps.
- One-hot encoded categorical features (`proto`, `service`, `state`).
- Standardized numerical features using `StandardScaler`.
- Train / Validation / Test split: 70% / 15% / 15%.

## Models Evaluated

- **LSTM**: Standard Long Short-Term Memory.
- **GRU**: Gated Recurrent Unit.
- **Transformer**: Encoder-only architecture.
- **Informer**: Specialized Transformer for long sequences (Simplified).
- **CNN-LSTM**: Convolutional layer for local spatial grouping followed by LSTM.
- **xLSTM**: Extended memory LSTM with exponential gating (sLSTM variant).

## Final Evaluation Metrics (Test Set)

| Model       | Precision  | Recall     | F1-score   |
| ----------- | ---------- | ---------- | ---------- |
| LSTM        | 0.9713     | 0.9958     | 0.9834     |
| GRU         | 0.9968     | 0.9752     | 0.9859     |
| Transformer | 0.9855     | 0.9952     | 0.9903     |
| Informer    | 0.9930     | 0.9884     | 0.9907     |
| CNN-LSTM    | 0.9826     | 0.9924     | 0.9875     |
| **xLSTM**   | **0.9931** | **0.9826** | **0.9878** |

## Conclusion

The xLSTM model demonstrated high precision and competitive F1-score compared to
standard RNN architectures. While the Transformer-based models (Transformer and
Informer) achieved higher F1 scores in this specific setup, xLSTM showed robust
performance with its extended memory mechanism and sequential inductive bias.

## How to Run

1. Preprocess the data: `python preprocess.py`
2. Train models: `python train.py` (Note: `train.py` can be edited to select
   specific models)
