import numpy as np
import torch
from torch import nn


def new_dim(L_in, kernel_size, stride=1, padding=0, dilation=1):
    """Calculate the length of a convolution's output.

    Parameters
    ----------
    L_in : integer
        Length of the input
    kernel_size : integer
        kernel size
    stride : integer
        stride
    padding : integer
        zero padding
    dilation : integer
        kernel dilation

    Returns
    -------
    integer
        Length of the layer output

    """
    return np.floor(1 +
                    (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
                    )


class CnnDnnLeakyRelu(nn.Module):
    """A series of 1d convolutions
    """

    def __init__(self, T, nchns, stride=5, out_stride=5,
                 in_kernel_size=5, out_kernel_size=5,
                 out_channels=5, padding=0, dilation=2, output_padding=0,
                 p=0.5, **kwargs):
        super(CnnDnnLeakyRelu, self).__init__()

        self.nchns = nchns
        self.has_history = False
        self.has_val_history = False

        # Convolutional layers
        self.conv1 = nn.Conv1d(self.nchns, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv3 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv4 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)

        # Encoder dimensions
        conv_out1 = new_dim(T, kernel_size=5, stride=1, padding=2,)
        pool_out1 = new_dim(conv_out1, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out2 = new_dim(pool_out1, kernel_size=5, stride=1, padding=2,)
        pool_out2 = new_dim(conv_out2, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out3 = new_dim(pool_out2, kernel_size=5, stride=1, padding=2,)
        pool_out3 = new_dim(conv_out3, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out4 = new_dim(pool_out3, kernel_size=5, stride=1, padding=2,)
        pool_out4 = new_dim(conv_out4, kernel_size=2, stride=2, padding=0,
                            dilation=1)

        # Fully connected layers
        l1 = int(pool_out4 * out_channels)
        self.linear1 = nn.Linear(l1, 200)
        self.linear2 = nn.Linear(200, 2)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=p, inplace=False)

    def forward(self, x):
        h1, _ = self.pool1(self.relu(self.conv1(x)))
        h2, _ = self.pool2(self.relu(self.conv2(h1)))
        h3, _ = self.pool3(self.relu(self.conv3(h2)))
        cnn_output, _ = self.pool4(self.relu(self.conv4(h3)))
        code = self.dropout(cnn_output.view(cnn_output.size(0), -1))
        h4 = self.dropout(self.relu(self.linear1(code)))
        return self.linear2(h4)

    def predict_proba(self, x):
        return self.softmax(self.forward(x))


class Cnn3Fc(nn.Module):
    """A series of 1d convolutions
    """

    def __init__(self, T, nchns, stride=5, out_stride=5,
                 in_kernel_size=5, out_kernel_size=5,
                 out_channels=5, padding=0, dilation=2, output_padding=0,
                 p=0.5, **kwargs):
        super(Cnn3Fc, self).__init__()

        self.nchns = nchns
        self.has_history = False
        self.has_val_history = False

        # Convolutional layers
        self.conv1 = nn.Conv1d(self.nchns, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv3 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv4 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)

        # Encoder dimensions
        conv_out1 = new_dim(T, kernel_size=5, stride=1, padding=2,)
        pool_out1 = new_dim(conv_out1, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out2 = new_dim(pool_out1, kernel_size=5, stride=1, padding=2,)
        pool_out2 = new_dim(conv_out2, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out3 = new_dim(pool_out2, kernel_size=5, stride=1, padding=2,)
        pool_out3 = new_dim(conv_out3, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out4 = new_dim(pool_out3, kernel_size=5, stride=1, padding=2,)
        pool_out4 = new_dim(conv_out4, kernel_size=2, stride=2, padding=0,
                            dilation=1)

        # Fully connected layers
        l1 = int(pool_out4 * out_channels)
        self.linear1 = nn.Linear(l1, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 2)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=p, inplace=False)

    def forward(self, x):
        h1, _ = self.pool1(self.relu(self.conv1(x)))
        h2, _ = self.pool2(self.relu(self.conv2(h1)))
        h3, _ = self.pool3(self.relu(self.conv3(h2)))
        cnn_output, _ = self.pool4(self.relu(self.conv4(h3)))
        code = self.dropout(cnn_output.view(cnn_output.size(0), -1))
        h4 = self.dropout(self.relu(self.linear1(code)))
        h5 = self.dropout(self.relu(self.linear2(h4)))
        return self.linear3(h5)

    def predict_proba(self, x):
        return self.softmax(self.forward(x))


class CnnSeparate(nn.Module):
    """docstring for CnnSeparate."""

    def __init__(self, T, nchns, stride=5, out_stride=5,
                 in_kernel_size=5, out_kernel_size=5,
                 out_channels=5, padding=0, dilation=2, output_padding=0,
                 p=0.5, **kwargs):
        super(CnnSeparate, self).__init__()

        self.nchns = nchns
        self.out_channels = out_channels

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv3 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv4 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)

        # Encoder dimensions
        conv_out1 = new_dim(T, kernel_size=5, stride=1, padding=2,)
        pool_out1 = new_dim(conv_out1, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out2 = new_dim(pool_out1, kernel_size=5, stride=1, padding=2,)
        pool_out2 = new_dim(conv_out2, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out3 = new_dim(pool_out2, kernel_size=5, stride=1, padding=2,)
        pool_out3 = new_dim(conv_out3, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out4 = new_dim(pool_out3, kernel_size=5, stride=1, padding=2,)
        pool_out4 = new_dim(conv_out4, kernel_size=2, stride=2, padding=0,
                            dilation=1)

        # Fully connected layers
        l1 = int(pool_out4 * out_channels * self.nchns)
        self.linear1 = nn.Linear(l1, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 2)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=p, inplace=False)

    def forward(self, x):
        N, C_in, l = list(x.shape)
        h1, _ = self.pool1(self.relu(self.conv1(x.view(N * C_in, 1, l))))
        h2, _ = self.pool2(self.relu(self.conv2(h1)))
        h3, _ = self.pool3(self.relu(self.conv3(h2)))
        cnn_output, _ = self.pool4(self.relu(self.conv4(h3)))
        code = self.dropout(cnn_output.view(N, -1))
        h4 = self.dropout(self.relu(self.linear1(code)))
        h5 = self.dropout(self.relu(self.linear2(h4)))
        return self.linear3(h5)

    def predict_proba(self, x):
        return self.softmax(self.forward(x))


class CnnGraphFuse(nn.Module):
    """docstring for CnnSeparate."""

    def __init__(self, T, nchns, Abar, stride=5, out_stride=5,
                 in_kernel_size=5, out_kernel_size=5,
                 out_channels=5, padding=0, dilation=2, output_padding=0,
                 p=0.5, **kwargs):
        super(CnnGraphFuse, self).__init__()

        self.nchns = nchns
        self.out_channels = out_channels
        self.Abar = Abar

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv3 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)
        self.conv4 = nn.Conv1d(out_channels, out_channels=out_channels,
                               kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                  dilation=1, return_indices=True)

        # Encoder dimensions
        conv_out1 = new_dim(T, kernel_size=5, stride=1, padding=2,)
        pool_out1 = new_dim(conv_out1, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out2 = new_dim(pool_out1, kernel_size=5, stride=1, padding=2,)
        pool_out2 = new_dim(conv_out2, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out3 = new_dim(pool_out2, kernel_size=5, stride=1, padding=2,)
        pool_out3 = new_dim(conv_out3, kernel_size=2, stride=2, padding=0,
                            dilation=1)
        conv_out4 = new_dim(pool_out3, kernel_size=5, stride=1, padding=2,)
        pool_out4 = new_dim(conv_out4, kernel_size=2, stride=2, padding=0,
                            dilation=1)

        # Fully connected layers
        l1 = int(pool_out4 * out_channels * self.nchns) * 2
        self.linear1 = nn.Linear(l1, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 2)
        self.relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=p, inplace=False)

    def forward(self, x):
        N, C_in, l = list(x.shape)
        h1, _ = self.pool1(self.relu(self.conv1(x.view(N * C_in, 1, l))))
        h2, _ = self.pool2(self.relu(self.conv2(h1)))
        h3, _ = self.pool3(self.relu(self.conv3(h2)))
        cnn_output, _ = self.pool4(self.relu(self.conv4(h3)))
        # Average over the neighbors according to the adjacency matrix
        neighbor_average = torch.matmul(self.Abar, cnn_output.view(N, C_in, -1))
        # Concatenate
        code = self.dropout(
            torch.cat((cnn_output.view(N, -1), neighbor_average.view(N, -1)), dim=1)
        )
        h4 = self.dropout(self.relu(self.linear1(code)))
        h5 = self.dropout(self.relu(self.linear2(h4)))
        return self.linear3(h5)

    def predict_proba(self, x):
        return self.softmax(self.forward(x))
