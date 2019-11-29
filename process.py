from processing.frame_prediction.generate_train import generate_training_data
from pyspark import SparkContext
import torch
from processing.read_data import VideoReader
from processing.frame_prediction.naive_net import Naive
from processing.frame_prediction.train_naive import main as train_main
import numpy as np
from pathlib import Path
from glob import glob
from random import sample
from tqdm import tqdm
from typing import List
import logging
import json
import cv2
import time

def init_logging():
    logging.INFOFRAME = 21
    logging.INFOFILE = 22
    logging.addLevelName(logging.INFOFRAME, 'INFOFRAME')
    logging.addLevelName(logging.INFOFILE, 'INFOFILE')

    logger = logging.getLogger('process')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('process.log')
    fh.setLevel(logging.DEBUG)
    fmtstr = '%(name)s:%(levelname)s: %(asctime)s \n%(message)s'
    fh.setFormatter(logging.Formatter(fmtstr))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmtstr.replace('\n', '')))
    ch.setLevel(logging.INFOFILE)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def test_naive():
    EXAMPLE = Path(
        './processed_data/frame_prediction/frames/170523-173539271876.jpg'
    )
    EXAMPLE2 = Path(
        './processed_data/frame_prediction/frames/170523-173539305301.jpg'
    )

    im = cv2.imread(str(EXAMPLE))[:,:,1]
    im2 = cv2.imread(str(EXAMPLE2))[:,:,1]

    im_pytorch = torch.Tensor((im, im2)).reshape(1, 2, *im.shape)
    logger.log(logging.INFO, f'image dimensions: {im.shape}')

    net = Naive(2, 1)
    logger.log(logging.INFO, f'net initialized: {net}')
    out = net(im_pytorch)[0][0].detach().numpy()
    logger.log(logging.INFO, f'output_range: {np.min(out)}, {np.max(out)}')
    out = out - np.min(out)
    logger.log(logging.INFO, f'output_range: {np.min(out)}, {np.max(out)}')
    out = out / np.max(out) * 255
    out = out.astype('uint8')
    logger.log(logging.INFO, f'output dimensions: {out.shape}')
    logger.log(logging.INFO, f'output_range: {np.min(out)}, {np.max(out)}')

    cv2.imshow('input', np.concatenate((im, im2), axis=0))
    cv2.waitKey(0)
    cv2.imshow('output', out)
    cv2.waitKey(0)


def generate_trainset():
    log = logger.getChild('generate_trainset')

    # Script variables
    DATADIR = Path('data')
    FILE_COUNT = None  # number of files to sample. None selects all files
    OUTPUT_DIR = Path('./processed_data/frame_prediction/')

    sample_files : List[Path] = list(map(Path, glob(f'{DATADIR}/*/*.MP4')))
    if FILE_COUNT is not None:
        sample_files = sample(sample_files, FILE_COUNT)

    def reduce_meta_files(l1, l2):
        args1 = l1['args']
        args2 = l2['args']
        fp = args1['filepath'] + '\t' + args2['filepath']
        del args1['filepath']
        del args2['filepath']

        assert args1 == args2
        args1['filepath'] = fp
        l1['data'] += l2['data']
        l1['args'] = args1
        return l1

    def process_video(filepath, check_processed=True):
        print(filepath)
        return generate_training_data(
            filepath,
            output_dir=OUTPUT_DIR,
            average_sps=0.2,
            relative_chain=(0,-2,-4,-8),
            rescale=.3,
            timefmt='%y%m%d-%H%M%S%f',
            logger=log,
            check_processed=check_processed,
        )

    sc = SparkContext()
    video_files = sc.parallelize(sample_files)
    meta_files = video_files.map(process_video)
    with open(OUTPUT_DIR / f'all_out.json', 'w') as f:
        json.dump(meta_files.reduce(reduce_meta_files), f)



logger = init_logging()

if __name__ == '__main__':

    function = 'generate_trainset'

    if function == 'train_naive':
        train_main()

    if function == 'test_naive':
        test_naive()

    if function == 'generate_trainset':
        generate_trainset()


