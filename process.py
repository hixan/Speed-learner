from processing.frame_prediction.naive_net import Naive
from processing.frame_prediction.Recurrant import LSTMNet
from processing.frame_prediction.generate_train import (
    generate_training_data, reduce_meta_files
)
from processing.read_data import VideoReader
from processing.frame_prediction.extrapolate import main as extrapolate
from tools import init_logging  # type: ignore
from processing.frame_prediction.train_naive import (
    DashcamPredictionDataset, step
)
from matplotlib import pyplot as plt
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

logger = init_logging('process', debug=True)

logger.info('\n\n\n\n\nlogger is working')

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
    #with open('losses.txt', 'r') as f:
    #    losses = json.load(f)
    losses = []
    logger = logger.getChild('train_naive')
    BATCH_SIZE = 80
    STATE_DICTS_DIR = Path('state_dicts/frame_prediction/')

    net = Naive(3, 1)
    #net.load_latest(STATE_DICTS_DIR)

    n_reduced = reduce_metas()
    logger.info(f'reduced {n_reduced} metadata files')

    dataset = DashcamPredictionDataset(
        DATADIR / 'all_out.json',
        DATADIR / 'frames',
        logger=logger
    )

    # loss criterion and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=.00001, momentum=0.03)

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
                        write_examples=(i % 5) == 0,
                        identifier=ident)
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


def train_lstmnet(logger):

    net = LSTMNet()

    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=.00001, momentum=0.03)

    file = VideoReader(list(map(Path, glob(f'data/*/*.MP4')))[0])
    next(file)  # skip meta or first frame

    def frames():
        for frame, loc in file:
            im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            im_sized = (cv2.resize(im_gray, (1280//2, 720//2)) / 128) - 1
            im_tensor = (torch.tensor(im_sized)).view(
                1, 1, 720//2, 1280//2).float()
            yield im_tensor

    def show_image(*tensors, name=None):
        shows = [tensor.detach().numpy()[0, 0] for tensor in tensors]
        show = ((np.concatenate(shows, axis=0) + 1) * 128).astype('uint8')
        #plt.hist(show.flatten())
        #plt.yscale('log')
        #plt.show()
        logger.info(f'showing image {name} dims {show.shape}')
        cv2.imshow('frame', show)

    fs = frames()
    last = next(fs)
    show = True
    for _ in range(500):
        print(_)
        curr = next(fs)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        out = (net(last) + 1 * 128)
        loss = criterion(out, curr)
        loss.backward()
        optimizer.step()
        if show:
            show_image(last, out, curr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            show = False

        last = curr


if __name__ == '__main__':

    if 'generate-naive' in sys.argv:
        generate_trainset(logger=logger.getChild('generate-naive'))

    if 'train-naive' in sys.argv:
        train_naive(logger=logger.getChild('train-naive'))

    if 'extrapolate-naive' in sys.argv:
        extrapolate(logger=logger.getChild('extrapolate-naive'))

    if 'train-lstmnet' in sys.argv:
        train_lstmnet(logger=logger.getChild('train-lstmnet'))
