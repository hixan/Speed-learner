from processing.frame_prediction.naive_net import Naive
import re
from processing.frame_prediction.generate_train import (
    generate_training_data, reduce_meta_files
)
from tools import init_logging  # type: ignore
from processing.frame_prediction.train_naive import (
    DashcamPredictionDataset, step
)
import os
import sys
from pyspark import SparkContext  # type: ignore
import torch
from torch import nn, optim
from functools import reduce
from itertools import count
import numpy as np  # type: ignore
from pathlib import Path
from glob import glob
from random import sample, shuffle
from typing import List
import logging
import json
import cv2  # type: ignore

logger = init_logging('process')

logger.info('\n\n\n\n\ndebugging logger is working')

DATADIR = Path('processed_data/frame_prediction/')


def reduce_metas():
    metas = []
    for f in glob(str(DATADIR / 'meta*')):
        with open(f, 'r') as file:
            metas.append(json.load(file))

    with open(DATADIR / 'all_out.json', 'w') as f:
        json.dump(reduce(reduce_meta_files, metas), f)
    return len(metas)


def train_naive(datalimit=None, logger=logger):

    for f in glob('processed_data/frame_prediction/examples/*'):
        os.remove(f)  # remove all old examples
    with open('losses.txt', 'r') as f:
        losses = json.load(f)
    logger = logger.getChild('train_naive')
    BATCH_SIZE = 35
    STATE_DICTS_DIR = Path('state_dicts/frame_prediction/')

    net = Naive(3, 1)

    # load latest learned states
    x = glob(str(STATE_DICTS_DIR / 'naive_state_dict') + '*')
    latest = sorted(
        x,
        key=lambda x: int(re.search(r'naive_state_dict(\d+)', x).groups()[0])
    )[-1]
    net.load_state_dict(torch.load(latest))
    net.eval()

    n_reduced = reduce_metas()
    logger.info(f'reduced {n_reduced} metadata files')

    dataset = DashcamPredictionDataset(
        DATADIR / 'all_out.json',
        DATADIR / 'frames',
        logger=logger
    )

    # loss criterion and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.03)

    # loop over epochs
    for epoch in count(len(losses)):  # loop over the dataset multiple times
        losses.append([])

        idxs = list(range(len(dataset)))
        shuffle(idxs)
        batches = np.array_split(idxs, len(dataset) // BATCH_SIZE)
        logger.info(f'initiating epoch {epoch} with {len(batches)} batches.')
        for i, indexes in enumerate(batches, 0):
            # get the inputs; data is a list of [inputs, labels]
            data = dataset[indexes]
            inputs, labels = data['x'].float(), data['y'].float()
            ident = (epoch * (10 ** int(np.log(len(batches))) + 1) + i)
            ident = f'{str(ident):0>10}'
            loss = step(inputs, labels, net, optimizer, criterion,
                        log_info=(i % 20) == 0,
                        logger=logger.getChild('step'),
                        write_examples=True,
                        identifier=ident)
            logger.debug(f'processed step {ident}. Loss was {loss}')
            losses[-1].append(loss)
            with open('losses.txt', 'w') as f:
                json.dump(losses, f)

        torch.save(
            net.state_dict(),
            f'{STATE_DICTS_DIR / "naive_state_dict"}{epoch}'
        )

    print('Finished Training')


def generate_trainset(logger=logger):
    log = logging.getLogger('py4j')  # logger.getChild('generate_trainset')

    # Script variables
    DATADIR = Path('data')
    FILE_COUNT = None  # number of files to sample. None selects all files
    OUTPUT_DIR = Path('./processed_data/frame_prediction/')

    sample_files: List[Path] = list(map(Path, glob(f'{DATADIR}/*/*.MP4')))
    if FILE_COUNT is not None:
        sample_files = sample(sample_files, FILE_COUNT)
    completed = set()

    def process_video(filepath, check_processed=True):

        log.error('test')

        rv = generate_training_data(
            filepath,
            output_dir=OUTPUT_DIR,
            average_sps=0.2,
            relative_chain=(0, -2, -4, -8),
            transform=lambda x: cv2.resize(
                cv2.cvtColor(x, cv2.COLOR_BGR2GRAY),
                (192, 108)
            ),
            timefmt='%y%m%d-%H%M%S%f',
            logger=log,
            check_processed=check_processed,
        )
        completed.add(filepath)
        log.error(f'count for this thread: {len(completed)}\n')
        return rv

    sc = SparkContext()
    video_files = sc.parallelize(sample_files)
    meta_files = video_files.map(process_video)
    with open(OUTPUT_DIR / f'all_out.json', 'w') as f:
        json.dump(meta_files.reduce(reduce_meta_files), f)


if __name__ == '__main__':

    function = 'train_naive'

    if 'generate' in sys.argv:
        generate_trainset()

    if 'train' in sys.argv:
        train_naive()
