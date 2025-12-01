import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize  # Not actually needed anymore, but left if you want to remove yourself
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
seq_length = 7 # Use the last 20 days to predict the next day
batch_size = 20
train_ratio = 0.8

# Training parameters
num_epochs = 1000
learning_rate = 0.0001

channels = 5 # number of features
num_companies = 10  # no longer used by model part, but left untouched

# Risk free rate
R = 4.36/100  # no longer used by model part, but left untouched


# Data Preparation Functions
def load_process_data(stock_symbol, start_date, end_date, seq_length, feature_columns=['Close', 'Open', 'High', 'Low', 'Volume']):
    """Downloads and preprocesses stock data for a specified symbol and date range. Normalizes features, 
    creates sequences with a given sequence length, and splits data into inputs (X) and target (y, close prices)."""
    # Download stock data
    #data = pd.read_csv("Stock data.csv")
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[feature_columns]
    data = np.array(data)
    
    # Normalize the data 
    #scaler_list = [None]*channels
    data_normalized = np.empty_like(data, dtype=float)
    
    for feature in range(channels):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_feature = scaler.fit_transform(data[:,feature:feature+1]) # nomalize a single feature

        for row in range(len(data)):
            data_normalized[row, feature] = scaled_feature[row, 0] #
            
        #scaler_list[feature] = scaler
    
    # Create sequence
    X, y = [], []
    for feature in range(seq_length, len(data_normalized)):
        X.append(data_normalized[feature-seq_length:feature])
        y.append(data_normalized[feature])
        
    # output shape [samlpes, seq_length, features]
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
    # Create datasets and dataloader
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# 1) Define the 1D CNN and LSTM model
class CNN_LSTM(nn.Module):
    def __init__(self, in_channels, seq_length):
        # CNN parameters
        self.in_channels = in_channels
        self.seq_length = seq_length
        self.out_channels = 150
        
        # LSTM parameters
        self.hidden_size = 100
        self.num_stacked_layers = 2
        
        # Dense layer parameters
        self.hidden_features = 25
        self.output = 1
        
        super(CNN_LSTM, self).__init__()

        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5, stride=1, padding=2)
        self.lstm = nn.LSTM(input_size=self.out_channels, hidden_size=self.hidden_size, num_layers=self.num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output) # Output is the predicted next day's closing price
        self.relu = nn.ReLU()

    def forward(self, x):
        
        # CNN, input shape [Batch Size, Features, seq_length]
        #print(x.shape, 'before CNN1')
        x = self.relu(self.conv(x))
        #print(x.shape, 'after CNN1 and before transpose')
        
        x = x.transpose(2, 1)
        #print(x.shape, 'after transpose and before LSTM')
        
        # LSTM
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        # LSTM input shape [Batch Size, Time Steps, Features]
        x, _ = self.lstm(x, (h0, c0))
        #print(x.shape, 'after lstm and before pick')
        
        x = self.relu(x)
        
        x = x[:, -1, :]
        #print(x.shape, 'after pick and before fc')
                
        x = self.fc(x)
        #print(x.shape, 'after fc and is output')
        
        return x
    

def train(train_loader, test_loader):
    """"""
    # Loss and optimizer
    model = CNN_LSTM(channels ,seq_length)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping paramerters
    patience = 50
    running_loss = float('inf')
    lowest_val_loss = float('inf')
    epochs_since_improvement = 0
    
    # Store losses
    train_losses = []
    val_losses = []

    # Batch training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # forward pass and loss
            #print(data.shape, 'data shape')
            #print(target.shape, 'target shape')
            y_prediction = model(data)
            
            """
            normalize y_prediction and target by last data point in data
            """
            
            loss = criterion(y_prediction, target)
            
            #zero gradients
            optimizer.zero_grad() 
            
            #backward pass
            loss.backward()
            
            #update weights
            optimizer.step()    
            
            train_losses.append(loss.item())
        
        #validation
        val_loss = 0
        with torch.no_grad():
            for val_batch_idx, (val_data, val_target) in enumerate(test_loader):
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
            torch.save(model.state_dict(), 'model_weights.pth')
        
        if epochs_since_improvement == patience:
            print('early break')
            break    
        
        #if epoch % 10 == 0:
            #print(f'Epoch {epoch}, Loss: {loss.item()}')
    return train_losses, val_losses

def predict(X_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    model = CNN_LSTM(channels ,seq_length)
    model.load_state_dict(torch.load('model_weights.pth'))
    
    with torch.no_grad():
        y_predicted = model(X_test)

    y_predicted = y_predicted.view(-1) # size from [200, 1] to [200]

    return y_predicted.cpu().numpy()


def plot_prediction(y_test, y_predicted):
    """"""
    # Plot actual vs predicted prices
    plt.plot(y_test, label='Actual Close')
    plt.plot(y_predicted, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()


def train_model(stock_symbol):
    """"""
    # Download and prepare stock data
    X, y = load_process_data(stock_symbol, start_date, end_date, seq_length)
    
    # Split data into training test sets (80/20)
    X_train, y_train, X_test, y_test = create_train_test_set(X, y, train_ratio)
    
    # Create train and test loader
    train_loader = create_dataloaders(X_train, y_train, batch_size, True)
    test_loader = create_dataloaders(X_test, y_test, batch_size)
    
    # Training model
    train_losses, val_losses = train(train_loader, test_loader)
    
    """
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.legend()
    plt.title(f"Loss Curves for {stock_symbol}")
    plt.show()
    """

def evaluate_and_plot(stock_symbol):
    # 1. Load data
    X, y = load_process_data(stock_symbol, start_date, end_date, seq_length)

    # 2. Create train/test split
    X_train, y_train, X_test, y_test = create_train_test_set(X, y, train_ratio)

    # 3. Make predictions on the test set
    y_predicted = predict(X_test)      # shape: [N,]
    y_test_flat = y_test[:, 0]         # true close prices (normalized)

    # 4. Compute evaluation metrics (all in normalized units)
    mae = mean_absolute_error(y_test_flat, y_predicted)
    mse = mean_squared_error(y_test_flat, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_flat, y_predicted)

    print(f"Evaluation on test set for {stock_symbol}:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2: {r2:.4f}")

    # 5. Plot predictions vs. true values
    plot_prediction(y_test_flat, y_predicted)

    # (optional) return metrics if you want to log/use them later
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }    
    
def stock_prediction(stock_symbol):
    """"""
    # Download and prepare stock data
    X, y = load_process_data(stock_symbol, start_date, end_date, seq_length)
    
    # Split data into training test sets (80/20)
    X_test = create_train_test_set(X, y, train_ratio, True)
    
    # predictions
    y_prediction = predict(X_test)

    # plot
    #plot_prediction(y_test, y_prediction)
    
    return y_prediction
    
    
def main():
    stock_symbol = ['AAPL']  # original variable kept, though only one symbol used
    # Example: train model for AAPL
    train_model('AAPL')
    
    evaluate_and_plot('AAPL')  # Plot predictions

    # Example: get predictions
    #preds = stock_prediction('AAPL')
    #print("Number of predictions:", len(preds))
        
    # 3. Autoregressive multi-step forecast over the whole test set
    #y_true_ar, y_pred_ar = autoregressive_forecast(stock_symbol)



if __name__ == '__main__':
    main()