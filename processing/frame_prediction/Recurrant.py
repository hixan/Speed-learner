import torch
from typing import Union
import re
from glob import glob
from pathlib import Path
from torch import nn


class Conv2DRecurrant(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Union[int, None] = None,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True
    ):
        super(Conv2DRecurrant, self).__init__()

        if padding is None:
            padding = kernel_size // 2

        # in channels is multiplied by two because it takes in the selected
        # memory and the input
        # output is not modified

        self.mem_forget_gate = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

        self.mem_remember_gate = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

        self.mem_remember_values = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

        self.output_gate = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

        self.memory = None
        self.prev_output = None

    def forward(self, x):

        if self.memory is None:
            self.memory = torch.ones(x.shape, requires_grad=False).float()
        if self.prev_output is None:
            self.prev_output = torch.ones(x.shape, requires_grad=False).float()

        self.memory.detach_()
        self.prev_output.detach_()

        # concatenate input and memory
        inpt = torch.cat((self.prev_output, x), dim=1)

        forget_gate = torch.sigmoid(self.mem_forget_gate(inpt))
        mem_remember_gate = torch.sigmoid(self.mem_remember_gate(inpt))
        mem_remember_values = torch.tanh(self.mem_remember_values(inpt))
        output_gate = torch.sigmoid(self.output_gate(inpt))

        self.memory *= forget_gate
        self.memory += mem_remember_values * mem_remember_gate
        output = torch.tanh(self.memory) * output_gate
        self.prev_output = output
        return output

    def load_latest(self, directory: Path):
        # load latest learned states
        x = glob(str(directory / 'recurrant_state_dict') + '*')
        latest = sorted(
            x,
            key=lambda x: int(
                re.search(  # type: ignore
                    r'recurrant_state_dict(\d+)',
                    x
                ).groups()[0]  # type: ignore
            )
        )[-1]
        self.load_state_dict(torch.load(latest))
        self.eval()


class LSTMNet(nn.Module):

    def __init__(self):
        super(LSTMNet, self).__init__()

        self.conv1 = Conv2DRecurrant(1, 1)

    def forward(self, x):
        return self.conv1(x)
