# Network Anomaly Detection with Long-Horizon Sequences

This project implements and evaluates multiple deep learning models for binary
network anomaly detection using two major datasets: **UNSW-NB15** and
**CIC-IDS2017**. The goal is to compare standard architectures with the **xLSTM
(extended memory LSTM)** model across different network environments.

## Project Structure

- `models.py`: Unified implementation of all deep learning architectures.
- `datasets/`: Raw data storage (ignored by git).
- `UNSW-NB15/`: Preprocessing and training scripts for the UNSW-NB15 dataset.
- `CIC-IDS2017/`: Preprocessing and training scripts for the CIC-IDS2017
  dataset.

## Datasets & Labeling

- **Binary Classification**: Normal (0) vs. Attack (1).
- **Sequence Length**: 30 steps.
- **Sliding Window**: Step size 1.
- **Preprocessing**:
  - Standardized features, one-hot encoding for categorical data.
  - Drop IDs, IP addresses, and timestamps.

## Models Evaluated

- **LSTM / GRU**: Standard Recurrent Neural Networks.
- **Transformer**: Encoder-only Transformer.
- **Informer**: Simplified Informer variant for long sequences.
- **CNN-LSTM**: Combined spatial and temporal modeling.
- **xLSTM**: Extended memory LSTM with exponential gating (sLSTM variant).

## Final Evaluation Results (Test Set)

### 1. UNSW-NB15

| Model       | Precision  | Recall     | F1-score   |
| ----------- | ---------- | ---------- | ---------- |
| LSTM        | 0.9713     | 0.9958     | 0.9834     |
| GRU         | 0.9968     | 0.9752     | 0.9859     |
| Transformer | 0.9855     | 0.9952     | 0.9903     |
| Informer    | 0.9930     | 0.9884     | 0.9907     |
| CNN-LSTM    | 0.9826     | 0.9924     | 0.9875     |
| **xLSTM**   | **0.9931** | **0.9826** | **0.9878** |

### 2. CIC-IDS2017 (500k Sample)

| Model       | Precision  | Recall     | F1-score   |
| ----------- | ---------- | ---------- | ---------- |
| LSTM        | 0.9823     | 0.9813     | 0.9818     |
| GRU         | 0.9941     | 0.5318     | 0.6930     |
| Transformer | 0.9989     | 0.9972     | 0.9981     |
| Informer    | 0.9989     | 0.3547     | 0.5235     |
| CNN-LSTM    | 0.9888     | 0.9868     | 0.9878     |
| **xLSTM**   | **0.9824** | **0.9830** | **0.9827** |

## Conclusion

The **xLSTM** model consistently shows high precision and robust F1 scores
across both datasets. In **UNSW-NB15**, it competed closely with Transformer
models, while in **CIC-IDS2017**, it maintained strong balanced performance
($0.9827$ F1), significantly outperforming standard RNNs like GRU which
struggled with recall in this configuration.

## How to Run

Every dataset has its own folder with `preprocess.py` and `train.py`:

1. Navigate to the dataset folder (e.g., `cd UNSW-NB15`).
2. Run preprocessing: `python preprocess.py`.
3. Run training: `python train.py`.
