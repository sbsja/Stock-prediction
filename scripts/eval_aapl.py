import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from src.config import Config
from src.data import load_process_data, create_train_test_set
from src.eval import load_model, predict, evaluate, plot_prediction

def main():
    cfg = Config()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    stock_symbol = "AAPL"
    X, y = load_process_data(
        stock_symbol,
        cfg.start_date, cfg.end_date,
        cfg.seq_length,
        cfg.feature_columns,
        cfg.channels,
    )

    X_train, y_train, X_test, y_test = create_train_test_set(X, y, cfg.train_ratio)

    model = load_model(cfg, device)
    y_pred = predict(model, X_test)
    y_true = y_test[:, 0]

    metrics = evaluate(y_true, y_pred)
    print(f"Evaluation for {stock_symbol}: {metrics}")

    plot_prediction(y_true, y_pred)

if __name__ == "__main__":
    main()
