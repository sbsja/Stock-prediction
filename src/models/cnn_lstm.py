import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, in_channels, seq_length):
        super().__init__()
        self.out_channels = 150
        self.hidden_size = 100
        self.num_stacked_layers = 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=5,
            padding=2
        )

        self.lstm = nn.LSTM(
            input_size=self.out_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_stacked_layers,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))      # [B, C, T]
        x = x.transpose(2, 1)            # [B, T, C]

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=x.device)

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        return self.fc(x)
