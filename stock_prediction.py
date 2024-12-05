import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize


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
num_companies = 10

# Risk free rate
R = 4.36/100


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
        
        #validation
        val_loss = 0
        with torch.no_grad():
            for val_batch_idx, (val_data, val_target) in enumerate(test_loader):
                val_y_prediction = model(val_data)
                val_loss += criterion(val_y_prediction, val_target).item()
                
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
            

def predict(X_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    model = CNN_LSTM(channels ,seq_length)
    model.load_state_dict(torch.load('model_weights.pth'))
    
    with torch.no_grad():
        y_predicted = model(X_test)

    y_predicted = y_predicted.view(-1) # size from [200, 1] to [200]

    return np.array(y_predicted)


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
    train(train_loader, test_loader)
    
    
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
    
    
def Markowitz(stock_symbol_list=['AAPL', 'IBM', 'SPOT', 'GOOG', 'AMZN', 'META', 'ORCL', 'TSLA', 'INTC', 'SAP']):
    stocks_pred_prices = []
    for stock_symbol in stock_symbol_list:
        # make prediction for a stock
        prediction = stock_prediction(stock_symbol)
        
        series_pred = pd.Series(prediction)
        stocks_pred_prices.append(series_pred)
        
    stocks_data = pd.concat(stocks_pred_prices, axis=1)
    stocks_data.columns = stock_symbol_list
    
    # Returns log(rt) = log(pt / p(t-1))
    log_return = np.log(1+stocks_data.pct_change())
    log_return_mean = log_return.mean()
    sigma = log_return.cov()

    monte_calro_markowitz(log_return_mean, sigma)
    find_optimal_weights(log_return_mean, sigma)
    even_distributed_weights(log_return_mean, sigma)
    plot_markowitz_frontier(log_return_mean, sigma)
    
    
def check_sum_one(weights):
        return np.sum(weights) - 1


def find_optimal_weights(log_return_mean, sigma):
    """"""
    def negative_sharpe_ratio(weights):
        """
        Sharpe ratio, SR = (Rp - Rf) / sigmap
        Rp = return of the portfolio
        Rf = risk-free rate 
        sigmap = standard deviation of the portfolio's excess return.
        """
        # Annual expected return
        expected_return = np.sum((log_return_mean * weights) * 252)

        # Annual expected volatility
        expected_vol = np.sqrt(np.dot(weights.T, np.dot((sigma * 252), weights)))
        
        # Sharpe ratio
        sharpe_ratio = (expected_return - R) / expected_vol

        return -1*sharpe_ratio
    
    
    weights = np.array([1/num_companies for i in range(num_companies)])
    
    # Bounds, makes sure no asset is more than 100% of the portfolio
    bounds = tuple((0,1) for i in range(num_companies))

    # Constraints, makes sure the weights do not exceed 100%
    constraints = ({'type':'eq', 'fun':check_sum_one})
    
    optimal_weights = minimize(negative_sharpe_ratio, weights, method='SLSQP', bounds=bounds, constraints=constraints)
    #print(optimal_weights, 'weights scipy')
    
    #calculation
    optimal_return = np.sum((log_return_mean * optimal_weights.x) * 252)
    optimal_volatility = np.sqrt(np.dot(optimal_weights.x.T, np.dot((sigma * 252), optimal_weights.x)))
    maximized_sharpe_ratio = (optimal_return - R) / optimal_volatility
    #print(optimal_return, 'scipy optimal expected return')
    #print(optimal_volatility, 'scipy optimal expected volatility')
    #print(maximized_sharpe_ratio, 'scipy maximum sharpe ratio')
    plt.scatter(optimal_volatility, optimal_return, color='red', label='optimally distributed weights')


def even_distributed_weights(log_return_mean, sigma):
    weight = np.array([1/num_companies for i in range(num_companies)])
    
    even_return = np.sum((log_return_mean * weight) * 252)
    even_volatility = np.sqrt(np.dot(weight.T, np.dot((sigma * 252), weight)))
    even_sharpe_ratio = (even_return - R) / even_volatility
    #print(even_return, 'even expected return')
    #print(even_volatility, 'even expected volatility')
    #print(even_sharpe_ratio, 'even sharpe ratio')
    plt.scatter(even_volatility, even_return, color='black', label='evenly distributed weights')
    
    
def monte_calro_markowitz(log_return_mean, sigma):
    num_portfolio = 10000
    weights_list = np.zeros((num_portfolio, num_companies))
    
    expected_return_list = np.zeros(num_portfolio)
    expected_vol_list = np.zeros(num_portfolio)
    sharpe_ratio_list = np.zeros(num_portfolio)
    
    for k in range(num_portfolio):
        # generate random weights vector
        weight = np.array(np.random.random(num_companies)) # array med 10 random nummer
        weight = weight / np.sum(weight) # normalisera
        weights_list[k, :] = weight
            
        # Annual expected log return
        expected_return_list[k] = np.sum((log_return_mean * weight) * 252)
        
        # Annual expected volatility
        expected_vol_list[k] = np.sqrt(np.dot(weight.T, np.dot(sigma * 252, weight)))
        
        # Sharpe ratio
        sharpe_ratio_list[k] = (expected_return_list[k] - R) / expected_vol_list[k]

    max_sharpe = sharpe_ratio_list.argmax()
    approximated_weights = weights_list[max_sharpe, :]
    #print(weights_list[max_sharpe, :], 'weights monte carlo')
    
    monte_return = expected_return_list[max_sharpe]
    monte_vol = expected_vol_list[max_sharpe]
    monte_sharpe = sharpe_ratio_list[max_sharpe]
    #print(monte_return, 'monte carlo method expected return')
    #print(monte_vol, 'monte carlo method expected volatility')
    #print(monte_sharpe, 'monte carlo maximum shape ratio')
    plt.figure(figsize=(12,5))
    plt.scatter(expected_vol_list, expected_return_list, c=sharpe_ratio_list)
    plt.xlabel('Expected volatility')
    plt.ylabel('Expected returns')
    plt.colorbar(label='Sharpe Ratio')

def plot_markowitz_frontier(log_return_mean, sigma):
    """"""
    def minimize_volatility(w):
        return np.sqrt(np.dot(w.T, np.dot((sigma * 252), w)))
    
    def get_return(w):
        return np.sum((log_return_mean * w) * 252)
        
    expected_returns = np.linspace(0.2,2.2,100)
    expected_volatility = []
    weight = np.array([1/num_companies for i in range(num_companies)])
    
    # Bounds, makes sure no asset is more than 100% of the portfolio
    bounds = tuple((0,1) for i in range(num_companies))
    
    for ret in expected_returns:
        # Constraints, makes sure the weights do not exceed 100%
        constraints = ({'type':'eq', 'fun':check_sum_one},
                       {'type':'eq', 'fun': lambda w: get_return(w) - ret})
        # find best volatility
        optimal_volatility = minimize(minimize_volatility, weight, method='SLSQP', bounds=bounds, constraints=constraints)
        expected_volatility.append(optimal_volatility['fun'])
    
    plt.plot(expected_volatility, expected_returns, label='Efficient markowitz frontier')
    plt.legend()
    plt.show()
        
def main():
    
    stock_symbol = ['AAPL']
    
    #for stock in stock_symbol_list:
    #y_prediction = stock_prediction(stock_symbol)

    #stock_symbol_list=['IBM', 'SPOT', 'AMZN', 'META', 'INTC']
    Markowitz()
    #print("optimal wegths:", M)


if __name__ == '__main__':
    main()