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
1. restore model with lower validation loss
2. include more features in data
3. look at train loop
4. plot X_train to see if shape is correct
5. scale each feature by itself, är denna korrekt?
6. change shape of y_test
7. plot monte carlo
8. compare to other sets of weights for example even distribution or random distribution
"""

# Use GPU if possible
device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

# Stock data parameters 
start_date = '2020-01-01'
end_date = '2024-01-01'

# Data preperation parameters
seq_length = 7  # Use the last 7 days to predict the next day
batch_size = 20
train_ratio = 0.8

# Training parameters
num_epochs = 1000
learning_rate = 0.0001

channels = 5  # number of features

# Data Preparation Functions
def load_process_data(stock_symbol, start_date, end_date, seq_length,
                      feature_columns=['Close', 'Open', 'High', 'Low', 'Volume']):
    """Downloads and preprocesses stock data for a specified symbol and date range. Normalizes features, 
    creates sequences with a given sequence length, and splits data into inputs (X) and target (y, close prices)."""
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[feature_columns]
    data = np.array(data)
    
    # Normalize the data 
    data_normalized = np.empty_like(data, dtype=float)
    
    for feature in range(channels):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_feature = scaler.fit_transform(data[:, feature:feature+1])  # normalize a single feature
        for row in range(len(data)):
            data_normalized[row, feature] = scaled_feature[row, 0]
    
    # Create sequence
    X, y = [], []
    for idx in range(seq_length, len(data_normalized)):
        X.append(data_normalized[idx-seq_length:idx])
        y.append(data_normalized[idx])
        
    # output shape [samples, seq_length, features]
    return np.array(X), np.array(y)


def create_train_test_set(X, y, train_ratio, only_X_test=False):
    """Splits data into training and test sets, reshapes for model compatibility."""
    split_idx = int(len(X) * train_ratio)

    if only_X_test:
        X_test = X[split_idx:]
        X_test_reshaped = np.swapaxes(X_test, 1, 2)
        return X_test_reshaped
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    y_train = y_train[:, 0:1]
    y_test = y_test[:, 0:1]

    # Tensor shape: [batch_size, channels, length]
    X_train_reshaped = np.swapaxes(X_train, 1, 2)
    X_test_reshaped = np.swapaxes(X_test, 1, 2)
    
    return X_train_reshaped, y_train, X_test_reshaped, y_test


def create_dataloaders(X, y, batch_size, shuffle=False):
    """Converts feature and target arrays into a DataLoader for batch processing, optionally shuffling."""
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# ========= CNN-ONLY MODEL =========
class CNN_Only(nn.Module):
    """
    Pure 1D CNN model.
    Input shape: [batch_size, channels, seq_length]
    """
    def __init__(self, in_channels, seq_length):
        super(CNN_Only, self).__init__()
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.out_channels = 150

        self.conv = nn.Conv1d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=5,
                              stride=1,
                              padding=2)
        self.relu = nn.ReLU()

        # We will average over the time dimension -> [batch, out_channels]
        self.fc = nn.Linear(self.out_channels, 1)  # predict next day's close (normalized)

    def forward(self, x):
        # x: [B, C, L]
        x = self.conv(x)     # [B, out_channels, L]
        x = self.relu(x)
        # Global average pooling over time dimension
        x = x.mean(dim=2)    # [B, out_channels]
        x = self.fc(x)       # [B, 1]
        return x


def train(train_loader, test_loader):
    """"""
    # Loss and optimizer
    model = CNN_Only(channels, seq_length).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping parameters
    patience = 50
    running_loss = float('inf')
    lowest_val_loss = float('inf')
    epochs_since_improvement = 0
    
    # Store losses
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

        # Validation
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
            torch.save(model.state_dict(), 'model_weights_cnn.pth')
        
        if epochs_since_improvement == patience:
            print('early break')
            break    

    return train_losses, val_losses


def predict(X_test):
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    model = CNN_Only(channels, seq_length).to(device)
    model.load_state_dict(torch.load('model_weights_cnn.pth', map_location=device))
    model.eval()
    
    with torch.no_grad():
        y_predicted = model(X_test)

    y_predicted = y_predicted.view(-1)

    return y_predicted.cpu().numpy()


def plot_prediction(y_test, y_predicted):
    plt.plot(y_test, label='Actual Close')
    plt.plot(y_predicted, label='Predicted Close')
    plt.title('CNN-Only Model: Actual vs Predicted Close Prices')
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
    plt.title(f"Loss Curves for {stock_symbol} (CNN only)")
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

    print("\nEvaluation of saved CNN model on test set:")
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
    #train_model(stock_symbol)
    evaluate_saved_model(stock_symbol)
    #preds = stock_prediction(stock_symbol)
    #print("Number of predictions:", len(preds))


if __name__ == '__main__':
    main()
