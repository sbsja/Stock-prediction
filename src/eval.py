from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .model import CNN_LSTM

def load_model(cfg, device, weights_path=None):
    model = CNN_LSTM(cfg.channels, cfg.seq_length).to(device)
    if weights_path is None:
        weights_path = Path(cfg.artifacts_dir) / cfg.weights_name
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def predict(model, X):
    X = torch.tensor(X, dtype=torch.float32, device=next(model.parameters()).device)
    y = model(X).view(-1).detach().cpu().numpy()
    return y

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def plot_prediction(y_true, y_pred):
    plt.plot(y_true, label="Actual Close")
    plt.plot(y_pred, label="Predicted Close")
    plt.xlabel("Day")
    plt.ylabel("Close (normalized)")
    plt.legend()
    plt.show()
