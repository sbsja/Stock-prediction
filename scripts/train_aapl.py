import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import argparse
import torch
from src.config import Config
from src.data import load_process_data, create_train_test_set, create_dataloader
from src.train import train_model
from src.utils.repro import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn_lstm", "transformer"], default="cnn_lstm")
    # Optional transformer knobs (only used if --model transformer)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dim_feedforward", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()

    if args.model == "transformer":
        cfg = Config(
            model_name="transformer",
            model_kwargs={
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
            }
        )
        
        set_seed(cfg.seed, cfg.deterministic)
    else:
        cfg = Config(model_name="cnn_lstm")

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
