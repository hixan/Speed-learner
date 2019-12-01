from .naive_net import Naive
import numpy as np  # type: ignore
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torchvision.utils import make_grid  # type: ignore
import json
from pathlib import Path
from typing import Dict, Union, List, Callable, Sequence
import cv2  # type: ignore
from matplotlib import pyplot as plt

DATADIR = Path('processed_data/frame_prediction')


def main():
    # net = Naive(3, 1)

    data = DashcamPredictionDataset(
        DATADIR / 'all_out.json',
        DATADIR / 'frames',
        transform=None
    )
    DashcamPredictionDataset.show_observations(data[3, 5, 68, 657, 3857])


Observation = Dict[str, Union[str, List[str]]]


class DashcamPredictionDataset(Dataset):
    '''frames dataset generated from:
    processing.frame_prediction.generate_train.generate_training_data
    '''

    def __init__(
            self,
            metadata: Path,
            root_dir: Path,
            transform: Callable = None
    ):
        with open(metadata, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.meta['data'])

    def __getitem__(self, idx: Union[int, Sequence]):
        if isinstance(idx, int):
            idx = [idx]

        rv_meta: List[Observation] = [self.meta['data'][i] for i in idx]

        rv_data: Dict[str, torch.Tensor] = self._read_observations(rv_meta)

        if self.transform:
            return self.transform(rv_data)

        return rv_data

    def _read_observations(self, observations: Sequence[Observation]):

        y = torch.tensor([
            cv2.imread(
                str(self.root_dir / observation['y']),  # type: ignore
                0
            )
            for observation in observations
        ])
        x = torch.tensor([
            list(map(lambda x:cv2.imread(str(self.root_dir / x), 0),
                     observation['x']))
            for observation in observations
        ])
        return {
            'y': y.reshape(len(observations), 1, *y[0].shape),
            'x': x
        }

    @staticmethod
    def show_observations(observations, padding: int = 1):

        grids = []
        for x, y in zip(observations['x'], observations['y']):
            y = np.expand_dims(y, 1)
            x = np.expand_dims(x, 1)
            joined = np.append(x, y, axis=0)
            grids.append(np.swapaxes(
                make_grid(
                    list(map(torch.tensor, joined)),
                    padding=padding
                ), -2, -1
            ))

        grid = np.swapaxes(make_grid(grids), 0, -1)

        plt.axis('off')
        plt.imshow(grid.numpy())
        plt.show()
