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
- **BiLSTM**: Bidirectional LSTM for forward and backward temporal context.
- **Transformer**: Encoder-only Transformer for global dependencies.
- **Informer**: Optimized Transformer for long sequences (Simplified).
- **Autoformer**: Series-decomposition based Transformer for seasonal/trend
  patterns (Simplified).
- **CNN-LSTM**: Combined spatial (Conv1D) and temporal (LSTM) modeling.
- **TCN**: Temporal Convolutional Network with dilated convolutions.
- **xLSTM**: Extended memory LSTM with exponential gating (sLSTM variant).

## Final Evaluation Results (Test Set)

### 1. UNSW-NB15 Results

#### TABLE 2: Efficiency Metrics (UNSW-NB15)

| Model       | Params    | Train Time (s) | Inf Time/Sample (ms) |
| ----------- | --------- | -------------- | -------------------- |
| LSTM        | [Pending] | [Pending]      | [Pending]            |
| GRU         | [Pending] | [Pending]      | [Pending]            |
| BiLSTM      | [Pending] | [Pending]      | [Pending]            |
| Transformer | [Pending] | [Pending]      | [Pending]            |
| Informer    | [Pending] | [Pending]      | [Pending]            |
| Autoformer  | [Pending] | [Pending]      | [Pending]            |
| CNN-LSTM    | [Pending] | [Pending]      | [Pending]            |
| TCN         | [Pending] | [Pending]      | [Pending]            |
| xLSTM       | [Pending] | [Pending]      | [Pending]            |

#### TABLE 4: Statistical Metrics (UNSW-NB15)

| Model       | ROC-AUC   | PR-AUC    | MCC       | FPR       | Latency/Batch (ms) |
| ----------- | --------- | --------- | --------- | --------- | ------------------ |
| LSTM        | [Pending] | [Pending] | [Pending] | [Pending] | [Pending]          |
| GRU         | [Pending] | [Pending] | [Pending] | [Pending] | [Pending]          |
| BiLSTM      | [Pending] | [Pending] | [Pending] | [Pending] | [Pending]          |
| Transformer | [Pending] | [Pending] | [Pending] | [Pending] | [Pending]          |
| Informer    | [Pending] | [Pending] | [Pending] | [Pending] | [Pending]          |
| Autoformer  | [Pending] | [Pending] | [Pending] | [Pending] | [Pending]          |
| CNN-LSTM    | [Pending] | [Pending] | [Pending] | [Pending] | [Pending]          |
| TCN         | [Pending] | [Pending] | [Pending] | [Pending] | [Pending]          |
| xLSTM       | [Pending] | [Pending] | [Pending] | [Pending] | [Pending]          |

### 2. CIC-IDS2017 Results (500k Sample)

_Results tables follow the same structure as above._

## Conclusion

[To be updated after full benchmarking]

## How to Run

Every dataset has its own folder with `preprocess.py` and `train.py`:

1. Navigate to the dataset folder (e.g., `cd UNSW-NB15`).
2. Run preprocessing: `python preprocess.py`.
3. Run training: `python train.py --epochs 5`.
4. To run a specific model: `python train.py --model LSTM --epochs 5`.
