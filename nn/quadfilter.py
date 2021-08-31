import torch
from torch import nn


class QuadFilter(nn.Module):
    def __init__(self, dilation):
        super(QuadFilter, self).__init__()
        self.dilation = dilation
        self.weight = nn.Parameter(torch.Tensor(self.dilation))
        self.weight.data.uniform_(-10, 10)

    def forward(self, x):
        B, C_in, L = x.shape
        h = x.repeat((1, self.dilation, 1))
        for ii in range(self.dilation):
            h[:, ii*C_in:(ii+1)*C_in, :] *= self.weight[ii]
            h[:, ii*C_in:(ii+1)*C_in, 1:] += x[:, :, :-1]
            h[:, ii*C_in:(ii+1)*C_in, :-1] += x[:, :, 1:]
            h = h / torch.sqrt(torch.square(self.weight[ii]) + 2)
        return h


class FilterNet(nn.Module):
    def __init__(self, d_in):
        super(FilterNet, self).__init__()
        self.filt1 = QuadFilter(2)
        self.filt2 = QuadFilter(2)
        self.filt3 = QuadFilter(2)
        self.filt4 = QuadFilter(2)
        self.linear1 = nn.Linear(d_in[0]*16, 20)
        self.linear2 = nn.Linear(20, 2)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        h = self.filt1(x)
        h = self.filt2(h)
        h = self.filt3(h)
        h = self.filt4(h)
        h = torch.mean(torch.square(h), dim=2)
        h = self.relu(self.linear1(h))
        h = self.linear2(h)
        return h
