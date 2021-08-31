import torch
from torch import nn

from nn.layers import *


class CnnConv1dPoolEncoder(nn.Module):
    """A series of 1d convolutions for encoding the EEG signal"""

    def __init__(self, T, nchns, out_channels=5, **kwargs):
        super(CnnConv1dPoolEncoder, self).__init__()

        nlayers = 4
        out_channels_list = [out_channels] * nlayers
        self.layers = ConvPoolCascade(nchns, T, out_channels_list)
        self.d_out = self.layers[-1].d_out

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CnnSeparateEncoder(nn.Module):
    """Encode all channels separately"""

    def __init__(self, T, nchns, out_channels=5, **kwargs):
        super(CnnSeparateEncoder, self).__init__()

        nlayers = 4
        out_channels_list = [out_channels] * nlayers
        self.layers = ConvPoolCascade(1, T, out_channels_list)
        self.d_out = [nchns, self.layers[-1].d_out[1] * out_channels]

    def forward(self, x):
        # Reshape the input s.t. all channels are input separately
        N, C_in, l_in = list(x.shape)
        x = x.view(N*C_in, 1, l_in)
        # Perform convolution and pooling
        for layer in self.layers:
            x = layer(x)
        # Reshape output into original channels
        return x.view(N, C_in, -1)


class BlockCnn(nn.Module):
    """Use convolution blocks"""

    def __init__(self, T, nchns, out_channels=[5, 5, 5, 5], nlayers=[2, 2, 2, 2],
                 apply_batchnorm=True):
        super().__init__()

        block1 = ConvBlock([nchns, T], nlayers=nlayers[0],
                           out_channels=out_channels[0],
                           apply_batchnorm=apply_batchnorm)
        self.blocks = nn.ModuleList([block1])
        for ii in range(1, len(out_channels)):
            d_in = [out_channels[ii-1], self.blocks[ii-1].d_out[1]]
            self.blocks.append(ConvBlock(d_in, nlayers=nlayers[ii],
                                         out_channels=out_channels[ii],
                                         apply_batchnorm=apply_batchnorm))
        self.d_out = self.blocks[-1].d_out

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Identity(nn.Module):
    """Don't do anything"""

    def __init__(self, T, nchns):
        super(Identity, self).__init__()
        self.d_out = [nchns, T]

    def forward(self, x):
        return x
