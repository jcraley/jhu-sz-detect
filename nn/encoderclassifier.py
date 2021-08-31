import torch
from torch import nn


class EncoderClassifier(nn.Module):
    """
    A encoder and classifier stage
    """

    def __init__(self, encoder, classifier):
        super(EncoderClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        "Forward pass"
        h = self.encode(x)
        return self.classify(h)

    def encode(self, x):
        if x.dim() == 4:
            b, t, c, l = x.size()
            c_out, l_out = self.encoder.d_out
            return self.encoder(x.view(b*t, c, l)).view(b, t, c_out, l_out)
        else:
            return self.encoder(x)

    def classify(self, h):
        return self.classifier(h)

    def predict_proba(self, x):
        if x.dim() == 4:
            return self.softmax(self.forward(x).view(-1, 2))
        else:
            h = self.forward(x)
            return self.softmax(h)
