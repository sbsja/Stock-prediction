import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

""" 
Same pipeline, but model is a pure LSTM.
"""

# Use GPU if possible
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

# Stock data parameters 
start_date = '2020-01-01'
end_date = '2024-01-01'

# Data preperation parameters
seq_length = 7
batch_size = 20
train_ratio = 0.8

# Training parameters
num_epochs = 1000
learning_rate = 0.0001

channels = 5  # number of features


def load_process_data(stock_symbol, start_date, end_date, seq_length,
                      feature_columns=['Close', 'Open', 'High', 'Low', 'Volume']):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[feature_columns]
    data = np.array(data)
    
    data_normalized = np.empty_like(data, dtype=float)
    
    for feature in range(channels):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_feature = scaler.fit_transform(data[:, feature:feature+1])
        for row in range(len(data)):
            data_normalized[row, feature] = scaled_feature[row, 0]
    
    X, y = [], []
    for idx in range(seq_length, len(data_normalized)):
        X.append(data_normalized[idx-seq_length:idx])
        y.append(data_normalized[idx])
        
    return np.array(X), np.array(y)


def create_train_test_set(X, y, train_ratio, only_X_test=False):
    split_idx = int(len(X) * train_ratio)

    if only_X_test:
        X_test = X[split_idx:]
        X_test_reshaped = np.swapaxes(X_test, 1, 2)
        return X_test_reshaped
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    y_train = y_train[:, 0:1]
    y_test = y_test[:, 0:1]

    X_train_reshaped = np.swapaxes(X_train, 1, 2)
    X_test_reshaped = np.swapaxes(X_test, 1, 2)
    
    return X_train_reshaped, y_train, X_test_reshaped, y_test


def create_dataloaders(X, y, batch_size, shuffle=False):
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# ========= LSTM-ONLY MODEL =========
class LSTM_Only(nn.Module):
    """
    Pure LSTM model.
    Input comes in as [B, C, L]; we transpose to [B, L, C] for LSTM.
    """
    def __init__(self, in_channels, seq_length):
        super(LSTM_Only, self).__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.hidden_size = 100
        self.num_layers = 2

        self.lstm = nn.LSTM(input_size=self.in_channels,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size, 1)  # predict next day's close

    def forward(self, x):
        # x: [B, C, L] -> [B, L, C]
        x = x.transpose(1, 2)

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        x, _ = self.lstm(x, (h0, c0))  # [B, L, H]
        x = self.relu(x)
        x = x[:, -1, :]                # last time step: [B, H]
        x = self.fc(x)                 # [B, 1]
        return x


def train(train_loader, test_loader):
    model = LSTM_Only(channels, seq_length).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    patience = 50
    running_loss = float('inf')
    lowest_val_loss = float('inf')
    epochs_since_improvement = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            y_prediction = model(data)
            loss = criterion(y_prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, val_target in test_loader:
                val_data = val_data.to(device)
                val_target = val_target.to(device)
                val_y_prediction = model(val_data)
                val_loss += criterion(val_y_prediction, val_target).item()

        val_losses.append(val_loss)

        if val_loss < running_loss:
            running_loss = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            
        if val_loss < lowest_val_loss:
            print(epoch, val_loss)
            print(lowest_val_loss)
            lowest_val_loss = val_loss
            torch.save(model.state_dict(), 'model_weights_lstm.pth')
        
        if epochs_since_improvement == patience:
            print('early break')
            break    

    return train_losses, val_losses


def predict(X_test):
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    model = LSTM_Only(channels, seq_length).to(device)
    model.load_state_dict(torch.load('model_weights_lstm.pth', map_location=device))
    model.eval()
    
    with torch.no_grad():
        y_predicted = model(X_test)

    y_predicted = y_predicted.view(-1)
    return y_predicted.cpu().numpy()


def plot_prediction(y_test, y_predicted):
    plt.plot(y_test, label='Actual Close')
    plt.plot(y_predicted, label='Predicted Close')
    plt.title('LSTM-Only Model: Actual vs Predicted Close Prices')
    plt.xlabel('Day')
    plt.ylabel('Close (normalized)')
    plt.legend()
    plt.show()


def train_model(stock_symbol):
    X, y = load_process_data(stock_symbol, start_date, end_date, seq_length)
    X_train, y_train, X_test, y_test = create_train_test_set(X, y, train_ratio)
    
    train_loader = create_dataloaders(X_train, y_train, batch_size, True)
    test_loader = create_dataloaders(X_test, y_test, batch_size)
    
    train_losses, val_losses = train(train_loader, test_loader)
    """
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.legend()
    plt.title(f"Loss Curves for {stock_symbol} (LSTM only)")
    plt.show()
    """


def evaluate_saved_model(stock_symbol):
    X, y = load_process_data(stock_symbol, start_date, end_date, seq_length)
    X_train, y_train, X_test, y_test = create_train_test_set(X, y, train_ratio)
    y_true = y_test[:, 0]

    y_pred = predict(X_test)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("\nEvaluation of saved LSTM model on test set:")
    print(f"  MAE  (L1)     : {mae:.4f}")
    print(f"  RMSE (L2)     : {rmse:.4f}")
    print(f"  R² Score      : {r2:.4f}")

    plot_prediction(y_true, y_pred)


def stock_prediction(stock_symbol):
    X, y = load_process_data(stock_symbol, start_date, end_date, seq_length)
    X_test = create_train_test_set(X, y, train_ratio, True)
    y_prediction = predict(X_test)
    return y_prediction
    
    
def main():
    stock_symbol = 'AAPL'
    train_model(stock_symbol)
    evaluate_saved_model(stock_symbol)
    preds = stock_prediction(stock_symbol)
    print("Number of predictions:", len(preds))


if __name__ == '__main__':
    main()
