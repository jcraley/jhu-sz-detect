import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceLoss(nn.Module):
    def __init__(self, A):
        """Define the transition loss"""
        super(SequenceLoss, self).__init__()
        self.A = A

    def forward(self, input, target):
        T, C = input.size()
        y_hat = F.softmax(input, dim=1)
        # print(y_hat)
        counts = torch.zeros((C, C))
        for ii in range(C):
            for jj in range(C):
                counts[ii, jj] = torch.sum(y_hat[:T-1, ii]
                                           * y_hat[1:T, jj]) / T
        return -torch.sum(counts * torch.log(self.A))
