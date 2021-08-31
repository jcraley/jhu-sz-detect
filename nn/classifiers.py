import torch
from torch import nn

from nn.layers import new_dim


class Fc2(nn.Module):
    """A series of fully connected layers"""

    def __init__(self, d_in, dropout=0.0, hidden_sizes=[200],
                 apply_layer_norm=True, average_input=False):
        super(Fc2, self).__init__()

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.apply_layer_norm = apply_layer_norm
        self.average_input = average_input

        # Create the first layer
        if self.average_input:
            l_in = d_in[0]
        else:
            l_in = int(torch.prod(torch.tensor(d_in)))
        ln1 = nn.LayerNorm(l_in)
        layer1 = nn.Linear(l_in, hidden_sizes[0])

        # Create subsequent layers
        self.layer_norms = nn.ModuleList([ln1])
        self.layers = nn.ModuleList([layer1])
        for ii in range(1, len(hidden_sizes)-1):
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[ii-1]))
            self.layers.append(nn.Linear(hidden_sizes[ii-1],
                                         hidden_sizes[ii]))

        # Create the last layer
        self.layer_norms.append(nn.LayerNorm(hidden_sizes[-1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], 2))
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        if self.average_input:
            x = torch.mean(x, dim=2)
        else:
            x = x.view(x.size(0), -1)
        for ln, layer in zip(self.layer_norms, self.layers):
            if self.apply_layer_norm:
                x = ln(x)
            x = self.relu(layer(self.dropout(x)))
        return x


class Fc3(nn.Module):
    """A series of fully connected layers with three state output"""

    def __init__(self, d_in, dropout=0.0, hidden_sizes=[200],
                 apply_layer_norm=True, average_input=False):
        super(Fc3, self).__init__()

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.apply_layer_norm = apply_layer_norm
        self.average_input = average_input

        # Create the first layer
        if self.average_input:
            l_in = d_in[0]
        else:
            l_in = int(torch.prod(torch.tensor(d_in)))
        ln1 = nn.LayerNorm(l_in)
        layer1 = nn.Linear(l_in, hidden_sizes[0])

        # Create subsequent layers
        self.layer_norms = nn.ModuleList([ln1])
        self.layers = nn.ModuleList([layer1])
        for ii in range(1, len(hidden_sizes)-1):
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[ii-1]))
            self.layers.append(nn.Linear(hidden_sizes[ii-1],
                                         hidden_sizes[ii]))

        # Create the last layer
        self.layer_norms.append(nn.LayerNorm(hidden_sizes[-1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], 3))
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        if self.average_input:
            x = torch.mean(x, dim=2)
        else:
            x = x.view(x.size(0), -1)
        for ln, layer in zip(self.layer_norms, self.layers):
            if self.apply_layer_norm:
                x = ln(x)
            x = self.relu(layer(self.dropout(x)))
        return x


class RnnChannelClassifier(nn.Module):
    """docstring for RnnChannelClassifier."""

    def __init__(self, d_in, hidden_size=20, p=0.0, bidirectional=True,
                 **kwargs):
        super(RnnChannelClassifier, self).__init__()

        input_size = d_in[1]
        self.rnn = torch.nn.LSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 bidirectional=bidirectional)
        if bidirectional:
            self.linear = torch.nn.Linear(2 * hidden_size, 2)
        else:
            self.linear = torch.nn.Linear(hidden_size, 2)
        self.final_layer = torch.nn.Linear(2*d_in[0], 2)

    def forward(self, x):
        output, h_n = self.rnn(x)
        return self.final_layer(self.linear(output).view(x.size(0), -1))


class LstmClassifier(nn.Module):
    """docstring for RnnClassifier."""

    def __init__(self, d_in, hidden_size=20, nlayers=2,
                 bidirectional=True, dropout=0.1, average_input=True):
        super(LstmClassifier, self).__init__()

        self.average_input = average_input

        # Create the blstm
        if self.average_input:
            l_in = d_in[0]
        else:
            l_in = int(torch.prod(torch.tensor(d_in)))
        self.lstm = torch.nn.LSTM(input_size=l_in,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  num_layers=nlayers,
                                  dropout=dropout,
                                  batch_first=True)
        if bidirectional:
            self.linear = torch.nn.Linear(2 * hidden_size, 2)
        else:
            self.linear = torch.nn.Linear(hidden_size, 2)

    def forward(self, x):
        if self.average_input:
            x = torch.mean(x, dim=3)
        else:
            x = x.view(x.size(0), -1)
        output, h_n = self.lstm(x)
        return self.linear(output)


class WeiBaseline(nn.Module):
    """Use convolution blocks"""

    def __init__(self, d_in):
        super(WeiBaseline, self).__init__()

        # Activation, dropout, and pooling
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5, inplace=False)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3, padding=0,
                                 dilation=1, return_indices=False)

        # Convolution layers
        self.conv1 = nn.Conv1d(d_in[0], out_channels=32,
                               kernel_size=21, stride=1, padding=11)
        self.conv2 = nn.Conv1d(32, out_channels=64,
                               kernel_size=11, stride=1, padding=6)
        self.conv3 = nn.Conv1d(64, out_channels=128,
                               kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv1d(128, out_channels=128,
                               kernel_size=3, stride=1, padding=2)
        self.conv5 = nn.Conv1d(128, out_channels=128,
                               kernel_size=3, stride=1, padding=2)

        # Computed post CNN dimensions
        conv = new_dim(d_in[1], 21, stride=1, padding=10, dilation=1)
        pool = new_dim(conv, 3, stride=3, padding=0, dilation=1)
        conv = new_dim(pool, 11, stride=1, padding=5, dilation=1)
        pool = new_dim(conv, 3, stride=3, padding=0, dilation=1)
        conv = new_dim(pool, 3, stride=1, padding=1, dilation=1)
        pool = new_dim(conv, 3, stride=3, padding=0, dilation=1)
        conv = new_dim(pool, 3, stride=1, padding=1, dilation=1)
        pool = new_dim(conv, 3, stride=3, padding=0, dilation=1)
        conv = new_dim(pool, 3, stride=1, padding=1, dilation=1)
        pool = new_dim(conv, 3, stride=3, padding=0, dilation=1)

        # Fully connected layers
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, x):
        # Convolution layers
        x = self.drop(self.pool(self.relu(self.conv1(x))))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.drop(self.pool(self.relu(self.conv5(x))))

        # Global average pool
        x = torch.mean(x, dim=2)

        # Fully connected layers
        x = self.drop(self.relu(self.linear1(x)))
        return self.linear2(x)


class TwoDBaseline(nn.Module):
    """Use two d convolutions on features
    """

    def __init__(self, d_in):
        super(TwoDBaseline, self).__init__()

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5, inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                                 dilation=1, return_indices=False)

        self.conv1 = nn.Conv2d(1, out_channels=2,
                               kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, out_channels=4,
                               kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, out_channels=8,
                               kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, out_channels=16,
                               kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(16)

        self.linear = nn.Linear(16, 2)

    def forward(self, x):
        N, H, W = x.size()
        h = self.relu(self.conv1(x.view(N, 1, H, W)))
        h = self.pool(self.relu(self.conv2(h)))
        h = self.relu(self.conv3(h))
        h = self.pool(self.relu(self.conv4(h)))
        h = torch.mean(h, dim=(2, 3))
        return self.linear(h)
