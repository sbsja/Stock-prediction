from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.models import get_model


def _default_weights_path(cfg):
    # match train.py behavior if you do model-specific naming
    return Path(cfg.artifacts_dir) / f"{cfg.model_name}_{cfg.weights_name}"


def load_model(cfg, device, weights_path=None):
    if weights_path is None:
        weights_path = _default_weights_path(cfg)

    ckpt = torch.load(weights_path, map_location=device)

    # Case A: new format (dict with metadata)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # optional: rebuild cfg from checkpoint metadata for safety
        model_cfg = cfg
        if "model_name" in ckpt:
            # If you want to enforce loading exactly what was saved:
            from dataclasses import replace
            model_cfg = replace(
                cfg,
                model_name=ckpt.get("model_name", cfg.model_name),
                model_kwargs=ckpt.get("model_kwargs", cfg.model_kwargs),
                channels=ckpt.get("channels", cfg.channels),
                seq_length=ckpt.get("seq_length", cfg.seq_length),
            )

        model = get_model(model_cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])

    # Case B: old format (raw state_dict)
    else:
        model = get_model(cfg).to(device)
        model.load_state_dict(ckpt)

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
