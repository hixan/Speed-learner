from .naive_net import Naive
import random
import numpy as np  # type: ignore
import logging
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torchvision.utils import make_grid  # type: ignore
import json
from pathlib import Path
from typing import Dict, Union, List, Callable, Sequence
import cv2  # type: ignore


DATADIR = Path('processed_data/frame_prediction')

Observation = Dict[str, Union[str, List[str]]]


def step(x, y, net, optimizer, criterion, log_info=False,
         logger=logging.Logger('step'), write_examples=None, identifier=None,
         examples_file=None):

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    lossval = loss.item()

    if not (log_info or write_examples):
        return lossval

    o = outputs.detach().numpy()
    if log_info:
        logger.debug(f'loss: {lossval}.\nrange: {np.min(o)}, {np.max(o)}')
    if write_examples:
        if identifier is None:
            identifier = random.randint(0, 1000000000000)
        if examples_file is None:
            examples_file = Path('processed_data/frame_prediction/examples')
        grid = DashcamPredictionDataset.show_observations(
            observations={
                'x': x,
                'y': y - outputs
            },
            padding=0
        )
        fname = str(examples_file / f'{identifier}.bmp')
        cv2.imwrite(fname, grid)
        logger.info(f'wrote examples to {fname}.')

    return lossval


class DashcamPredictionDataset(Dataset):
    '''frames dataset generated from:
    processing.frame_prediction.generate_train.generate_training_data
    '''

    def __init__(
            self,
            metadata: Path,
            root_dir: Path,
            transform: Callable = None,
            logger: Union[logging.Logger, None] = None
    ):
        if logger is None:
            logger = logging.Logger('DashcamPredictionDataset logger')
        self.logger: logging.Logger = logger.getChild('DashcamPredictionDataset')
        with open(metadata, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.root_dir = root_dir
        self.logger.debug('dashcam prediction dataset instanciated succesfully.')

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

        try:
            y = torch.tensor([
                cv2.imread(
                    str(self.root_dir / observation['y']),  # type: ignore
                    0
                )
                for observation in observations
            ])
        except ValueError:
            self.logger.log(
                logging.ERROR,
                f'encountered an error with data: {[ob["y"] for ob in observations]}'
            )
            return {
                'x': torch.Tensor(),
                'y': torch.Tensor()
            }

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
            x = x.detach().numpy()
            y = y.detach().numpy()
            y = np.expand_dims(y, 1)
            x = np.expand_dims(x, 1)
            joined = np.append(x, y, axis=0)
            grids.append(np.swapaxes(
                make_grid(
                    list(map(torch.tensor, joined)),
                    padding=padding
                ), -2, -1
            ))

        grid = np.swapaxes(make_grid(grids), 0, -1).detach().numpy()
        #grid -= np.min(grid)
        #grid /= np.max(grid)
        #grid *= 255
        #grid = grid.astype(np.uint8)
        return grid
