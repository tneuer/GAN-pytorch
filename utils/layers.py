import torch
from torch.nn import Module

class LayerPrintSize(Module):
    def __init__(self):
        super(LayerPrintSize, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class LayerReshape(Module):
    def __init__(self, shape):
        super(LayerReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        x = torch.reshape(input=x, shape=(-1, *self.shape))
        return x


class LayerDebug(Module):
    def __init__(self):
        super(LayerDebug, self).__init__()

    def forward(self, x):
        return x