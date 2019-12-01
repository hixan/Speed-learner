from .naive_net import Naive
import torch
from torch import nn, optim
import json
from pathlib import Path
from typing import Dict, Union, List
import cv2

DATADIR = Path('processed_data/frame_prediction')

def main():
    net = Naive(3, 1)

    with open(DATADIR / 'all_out.json') as f:
        data_meta = json.load(f)
    print(type(data_meta))


def read_observation(observation: Dict[str, Union[str, List[str]]]):
    datadir = DATADIR / 'frames'
    y = torch.tensor(cv2.imread(str(datadir / observation['y']), 0))
    x = torch.tensor(list(map(lambda x:cv2.imread(str(datadir / x), 0), observation['x'])))
    return y.reshape(1, *y.shape), x


