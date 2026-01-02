# Stock Price Prediction with CNN+LSTM and Transformer

This project trains deep learning models to predict the **next day closing price** (normalized using Min–Max scaling per feature) of a stock using historical time-series data.

Two model architectures are supported:
- **CNN + LSTM**
- **Transformer (Encoder-based)**

The code automatically uses **GPU (CUDA)** if available, otherwise falls back to CPU.

## Quick Start

```bash
git clone <repo-url>
cd Stock-prediction
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python scripts/train_aapl.py --model transformer
python scripts/eval_aapl.py --model transformer
```

## Project Structure

```text
.
├── src/
│   ├── config.py          # Global configuration
│   ├── data.py            # Data loading & preprocessing
│   ├── train.py           # Training loop (model-agnostic)
│   ├── eval.py            # Evaluation utilities
│   └── models/
│       ├── cnn_lstm.py
│       ├── transformer.py
│       ├── factory.py     # Model selector
│       └── __init__.py
├── scripts/
│   ├── train_aapl.py      # Train on AAPL
│   └── eval_aapl.py       # Evaluate on AAPL
├── artifacts/             # Saved model weights (ignored by git)
├── requirements.txt
├── .gitignore
└── README.md
```

## Models

### CNN + LSTM
- CNN extracts local temporal patterns
- LSTM models long-term dependencies
- Strong baseline for time-series prediction

### Transformer
- Uses self-attention to model temporal relationships
- Parallelizable and scalable
- Better suited for longer sequences


## Requirements
- Python 3.9+
- PyTorch (CPU or CUDA-enabled)
- Internet connection (stock data is downloaded via yfinance)

## Installation

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Training a Model
All scripts must be run from the project root directory

### Train CNN + LSTM (default)
```bash
python scripts/train_aapl.py
```
or explicitly:
```bash
python scripts/train_aapl.py --model cnn_lstm
```

### Train Transformer
```bash
python scripts/train_aapl.py --model transformer
```
With custom Transformer hyperparameters:
```bash
python scripts/train_aapl.py \
  --model transformer \
  --d_model 128 \
  --nhead 4 \
  --num_layers 3 \
  --dim_feedforward 256
```

### During training

- AAPL stock data is downloaded using yfinance
- Features are normalized independently
- Sliding windows of length seq_length are created
- The selected model is trained with early stopping
- Best model weights are saved to:
```text
artifacts/
  cnn_lstm_model_weights.pth
  transformer_model_weights.pth
```

## Evaluating a Model
Evaluation must match the model type that was trained.

### Evaluate CNN + LSTM
```bash
python scripts/eval_aapl.py --model cnn_lstm
```

### Evaluate Transformer
```bash
python scripts/eval_aapl.py --model transformer
```

### Evaluation output
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² score
- Plot of actual vs predicted normalized closing price


## Reproducibility

- Fixed random seeds for Python, NumPy, and PyTorch
- Optional deterministic CUDA execution
- Cached data downloads to avoid data drift
- Model checkpoints store configuration metadata

