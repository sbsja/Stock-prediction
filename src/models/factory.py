# src/models/factory.py
from .cnn_lstm import CNN_LSTM
from .transformer import TransformerModel

def get_model(cfg):
    name = cfg.model_name.lower()

    if name == "cnn_lstm":
        return CNN_LSTM(in_channels=cfg.channels, seq_length=cfg.seq_length)

    if name == "transformer":
        return TransformerModel(
            n_features=cfg.channels,
            seq_length=cfg.seq_length,
            **cfg.model_kwargs
        )

    raise ValueError(f"Unknown model_name: {cfg.model_name}")
