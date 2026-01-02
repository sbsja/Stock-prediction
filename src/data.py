import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_process_data(stock_symbol, start_date, end_date, seq_length, feature_columns, channels):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[list(feature_columns)]
    data = np.array(data)

    data_normalized = np.empty_like(data, dtype=float)

    for feature in range(channels):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_feature = scaler.fit_transform(data[:, feature:feature+1])
        data_normalized[:, feature] = scaled_feature[:, 0]

    X, y = [], []
    for i in range(seq_length, len(data_normalized)):
        X.append(data_normalized[i-seq_length:i])
        y.append(data_normalized[i])

    return np.array(X), np.array(y)

def create_train_test_set(X, y, train_ratio, only_X_test=False):
    split_idx = int(len(X) * train_ratio)

    if only_X_test:
        X_test = X[split_idx:]
        return np.swapaxes(X_test, 1, 2)

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    y_train = y_train[:, 0:1]
    y_test = y_test[:, 0:1]

    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)

    return X_train, y_train, X_test, y_test

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

def create_dataloader(X, y, batch_size, shuffle=False, seed=42):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        pin_memory=torch.cuda.is_available(), 
        num_workers=0
    )
