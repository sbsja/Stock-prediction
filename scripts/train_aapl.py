import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


import torch
from src.config import Config
from src.data import load_process_data, create_train_test_set, create_dataloader
from src.train import train_model

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

    train_loader = create_dataloader(X_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = create_dataloader(X_test, y_test, cfg.batch_size, shuffle=False)

    train_model(train_loader, val_loader, cfg, device)

if __name__ == "__main__":
    main()
