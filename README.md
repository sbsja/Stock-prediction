# Stock Price Prediction with CNN+LSTM and Transformer

This project trains deep learning models to predict the next-day **normalized closing price** of a stock using historical time-series data.

Two model architectures are supported:
- **CNN + LSTM**
- **Transformer (Encoder-based)**

The code automatically uses **GPU (CUDA)** if available, otherwise falls back to CPU.

---

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


## Requirements
- Python 3.9+
- PyTorch (CPU or CUDA-enabled)
- Internet connection (stock data is downloaded via yfinance)

## Installation

### 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate         # Windows

### 2. Install dependencies
pip install -r requirements.txt


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
- model transformer \
- d_model 128 \
- nhead 4 \
- num_layers 3 \
- dim_feedforward 256
```
### During training

- AAPL stock data is downloaded using yfinance
- Features are normalized independently
- Sliding windows of length seq_length are created
- The selected model is trained with early stopping
- Best model weights are saved to:
artifacts/
  cnn_lstm_model_weights.pth
  transformer_model_weights.pth

## Evaluating a Model
Evaluation must match the model type that was trained.

### Evaluate CNN + LSTM
python scripts/eval_aapl.py --model cnn_lstm

### Evaluate Transformer
python scripts/eval_aapl.py --model transformer

### Evaluation output
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² score
- Plot of actual vs predicted normalized closing price

