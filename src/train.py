from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from .model import CNN_LSTM

def train_model(train_loader, val_loader, cfg, device):
    model = CNN_LSTM(cfg.channels, cfg.seq_length).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    artifacts_dir = Path(cfg.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifacts_dir / cfg.weights_name

    patience = cfg.patience
    lowest_val_loss = float("inf")
    epochs_since_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(cfg.num_epochs):
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            pred = model(data)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_data = val_data.to(device)
                val_target = val_target.to(device)
                val_pred = model(val_data)
                val_loss += criterion(val_pred, val_target).item()

        val_losses.append(val_loss)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), weights_path)
            print(f"[epoch {epoch}] val_loss={val_loss:.6f}  -> saved {weights_path}")
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print("Early stopping.")
            break

    return model, train_losses, val_losses, weights_path
