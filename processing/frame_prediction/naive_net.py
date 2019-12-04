from typing import List
from torch import nn


class Naive(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(Naive, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=7, padding=3)
        self.activ1 = nn.Tanh()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
        self.activ2 = nn.Tanh()
        self.conv3 = nn.Conv2d(20, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = x / 255
        x = self.conv1(x)
        shape = x.shape[-2:]
        x = self.activ1(x)
        assert x.shape[-2:] == shape
        x = self.conv2(x)
        assert x.shape[-2:] == shape
        x = self.activ2(x)
        assert x.shape[-2:] == shape
        x = self.conv3(x)
        assert x.shape[-2:] == shape

        return x * 255
