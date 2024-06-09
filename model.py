import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5,
                               kernel_size=(1, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=5, out_channels=10, kernel_size=(1, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))

        # Third convolutional layer
        self.conv3 = nn.Conv2d(
            in_channels=10, out_channels=5, kernel_size=(1, 2), stride=(1, 1))

        # Fully connected layer
        self.fc1 = nn.Linear(5 * 6 * 3, 90)
        self.dropout = nn.Dropout(0.5)

        # Output layer
        self.output = nn.Linear(90, 5)

    def forward(self, x):
        # Applying first conv layer
        x = self.pool1(F.relu(self.conv1(x)))

        # Applying second conv layer
        x = self.pool2(F.relu(self.conv2(x)))

        # Applying third conv layer
        x = F.relu(self.conv3(x))

        # Flattening the output for the fully connected layer
        x = x.view(-1, 5 * 6 * 3)

        # Fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer with softmax activation
        x = self.output(x)
        return F.log_softmax(x, dim=1)


class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=6, hidden_size=10, batch_first=True)
        self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(input_size=10, hidden_size=60, batch_first=True)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(60, 5)

    def forward(self, x):
        # First LSTM layer
        x, _ = self.lstm1(x)  # x shape: (batch_size, seq_length, 10)
        x = self.relu1(x)
        # Second LSTM layer
        x, _ = self.lstm2(x)  # x shape: (batch_size, seq_length, 60)
        # Take only the last output for the second LSTM
        x = self.relu2(x[:, -1, :])
        # Dropout
        x = self.dropout(x)
        # Output layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 4]):
        super(TCNBlock, self).__init__()
        self.convolutions = nn.ModuleList()
        def padding(d): return (kernel_size - 1) * d

        for dilation in dilations:
            self.convolutions.append(nn.Conv1d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               padding=padding(dilation),
                                               dilation=dilation))
            in_channels = out_channels  # Output of one layer is input to the next

    def forward(self, x):
        for conv in self.convolutions:
            x = F.relu(conv(x))
        return x


class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.tcn1 = TCNBlock(
            in_channels=6, out_channels=10, dilations=[1, 2, 4])
        self.tcn2 = TCNBlock(in_channels=10, out_channels=60, dilations=[
                             1])  # No dilation provided, assuming 1
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(60, 5)

    def forward(self, x):
        # x expected shape: (batch_size, 6, seq_length)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x[:, :, -1]  # Take the last timestep
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
