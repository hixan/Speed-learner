# import torch
from typing import List, Any
from torch import nn


class Naive(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(Naive, self).__init__()
        channels = [in_channels, 10, 20, out_channels]
        layers = [
            'conv2d',
            'softmax',
            'conv2d',
            'softmax',
            'conv2d',
        ]
        self.layers: List[Any] = []
        for layer in layers:
            if layer == 'conv2d':
                self.layers.append(nn.Conv2d(channels.pop(0), channels[0],
                                             kernel_size=8))
            elif layer == 'softmax':
                self.layers.append(nn.Softmax())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
