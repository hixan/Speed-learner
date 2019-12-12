from typing import List
import torch
import re
from glob import glob
from pathlib import Path
from torch import nn


class Naive(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(Naive, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 5, kernel_size=15, padding=7)
        self.activ1 = nn.Tanh()
        self.conv2 = nn.Conv2d(5, 14, kernel_size=11, padding=5)
        self.activ2 = nn.Tanh()
        self.conv3 = nn.Conv2d(14, out_channels, kernel_size=3, padding=1)

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

    def load_latest(self, directory: Path):
        # load latest learned states
        x = glob(str(directory / 'naive_state_dict') + '*')
        latest = sorted(
            x,
            key=lambda x: int(re.search(r'naive_state_dict(\d+)', x).groups()[0])  # type: ignore
        )[-1]
        self.load_state_dict(torch.load(latest))
        self.eval()


