import torch
from torch import nn


def conv3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """One dimensional length three kernel convolution"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1(in_channels, out_channels, stride=1):
    """One dimensional length 1 kernel convolution"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannels, channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        norm_layer = nn.BatchNorm1d
        self.conv1 = conv3(inchannels, channels, stride)
        self.bn1 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(channels, channels)
        self.bn2 = norm_layer(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResEncoder(nn.Module):

    def __init__(self, T, nchns, res_channels=20, res_layers=2):
        super(ResEncoder, self).__init__()
        # if norm_layer is None:
        norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        # Channel-wise embedding
        self.res_channels = res_channels
        self.conv1 = nn.Conv1d(1, res_channels, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.conv2 = nn.Conv1d(res_channels*nchns, res_channels, kernel_size=1, stride=1, padding=0,
                               bias=True, groups=res_channels)
        self.bn1 = norm_layer(res_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Set up residual blocks
        block = BasicBlock
        self.layers = nn.ModuleList()
        for _ in range(res_layers - 1):
            self.layers.append(self._make_layer(
                block, res_channels, res_channels))
        self.layers.append(self._make_layer(
            block, res_channels, res_channels, stride=2))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.d_out = [res_channels, 50]

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        zero_init_residual = True
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, channels, outchannels, blocks=1, stride=1,
                    dilate=False):
        return block(channels, outchannels)

    def _forward_impl(self, x):
        """Forward pass

        Args:
            x (tensor): Model input

        Returns:
            tensor: Model output
        """
        batch_size, nchns, L = x.shape
        x = self.conv1(x.view(batch_size*nchns, 1, L))
        x = x.view(batch_size, nchns, self.res_channels, -1).transpose(1, 2)
        x = x.reshape(batch_size, nchns * self.res_channels, -1)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
