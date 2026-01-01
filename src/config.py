from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # device
    device: str = "cuda:0"   # will be overridden at runtime if no cuda

    # data
    start_date: str = "2015-01-01"
    end_date: str = "2024-01-01"
    seq_length: int = 20
    batch_size: int = 50
    train_ratio: float = 0.8
    feature_columns: tuple = ("Close", "Open", "High", "Low", "Volume")

    # training
    num_epochs: int = 1000
    learning_rate: float = 1e-4
    patience: int = 50

    # model
    channels: int = 5

    # paths
    artifacts_dir: str = "artifacts"
    weights_name: str = "model_weights.pth"
