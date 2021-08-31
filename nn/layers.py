import copy
import torch
from torch import nn
import numpy as np


def new_dim(L_in, kernel_size, stride=1, padding=0, dilation=1):
    return np.floor(1 + (L_in + 2 * padding - dilation *
                            (kernel_size - 1) - 1) / stride
    )


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Conv1dPool(nn.Module):
    """A single convolution and pooling later with leaky relu activation"""

    def __init__(self, d_in, out_channels):
        super(Conv1dPool, self).__init__()

        # Define the components of a convolution and pool layer
        self.conv = nn.Conv1d(d_in[0], out_channels=out_channels,
                              kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                 dilation=1, return_indices=False)
        self.relu = nn.LeakyReLU()

        # Calculate length of the output
        conv_out = self.new_dim(d_in[1], kernel_size=5, stride=1, padding=2)
        pool_out = self.new_dim(conv_out, kernel_size=2, stride=2, padding=0)
        self.d_out = [out_channels, int(pool_out)]

    def new_dim(self, L_in, kernel_size, stride=1, padding=0, dilation=1):
        return np.floor(1 + (L_in + 2 * padding - dilation *
                             (kernel_size - 1) - 1) / stride
                        )

    def forward(self, x):
        return self.relu(self.pool(self.conv(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def ConvPoolCascade(nchns, T, out_channels_list):
    """Produce a series of Convolution and Pool Layers"""
    layers = nn.ModuleList([])
    layers.append(Conv1dPool([nchns, T], out_channels_list[0]))
    for out_channels in out_channels_list[1:]:
        layers.append(Conv1dPool(layers[-1].d_out, out_channels))
    return layers


class ConvBlock(nn.Module):
    """A single block for convolution"""

    def __init__(self, d_in, out_channels=5, nlayers=1, apply_batchnorm=True):
        super(ConvBlock, self).__init__()

        self.nlayers = nlayers
        self.apply_batchnorm = apply_batchnorm
        
        # Define the components of a convolution and pool layer
        self.first_conv = nn.Conv1d(d_in[0], out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.first_bn = nn.BatchNorm1d(out_channels)
        repeat_conv = nn.Conv1d(out_channels, out_channels=out_channels,
                                kernel_size=3, stride=1, padding=1)
        repeat_bn = nn.BatchNorm1d(out_channels)
        self.layers = clones(repeat_conv, nlayers-1)
        self.bns = clones(repeat_bn, nlayers-1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                 dilation=1, return_indices=False)
        self.relu = nn.LeakyReLU()

        # Calulate the output dimension
        pool_out = new_dim(d_in[1], kernel_size=2, stride=2, padding=0)
        self.d_out = [out_channels, int(pool_out)]

    def forward(self, x):
        x = self.relu(self.first_conv(x))
        if self.apply_batchnorm:
            x = self.first_bn(x)
        for ii, layer in enumerate(self.layers):
            x = self.relu(layer(x))
            if self.apply_batchnorm:
                x = self.bns[ii](x)
        x = self.pool(x)
        return x


class SeparableConvBlock(nn.Module):
    """A single block for convolution"""

    def __init__(self, d_in, out_channels=5, nlayers=1, apply_batchnorm=True):
        super(SeparableConvBlock, self).__init__()

        self.nlayers = nlayers
        self.apply_batchnorm = apply_batchnorm

        # Define the components of a convolution and pool layer
        depth_conv = nn.Conv1d(d_in[0], out_channels=out_channels * d_in[0],
                              kernel_size=3, stride=1, padding=1,
                              groups=d_in[0])
        point_conv = nn.Conv1d(out_channels * d_in[0],
                               out_channels=out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.depth_convs = clones(depth_conv, nlayers)
        self.point_convs = clones(point_conv, nlayers)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,
                                 dilation=1, return_indices=False)
        self.relu = nn.LeakyReLU()

        # Calulate the output dimension
        pool_out = self.new_dim(d_in[1], kernel_size=2, stride=1, padding=0)
        self.d_out = [out_channels, int(pool_out)]

    def forward(self, x):
        for depth, point in zip(self.depth_convs, self.point_convs):
            x = depth(x)
            x = point(x)
            x = self.relu(x)
        x = self.pool(x)
        return x





