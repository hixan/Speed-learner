from glob import glob
import torch
from ..read_data import VideoReader
from .naive_net import Naive
import logging
from random import choice, seed
from pathlib import Path
import cv2  # type: ignore
import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore



def main(logger=logging.Logger('None')):
    vfile = Path(choice(glob('data/*/*.MP4')))
    assert vfile.exists()

    net = Naive(3, 1)
    net.load_latest(Path('state_dicts/frame_prediction/'))

    vid = VideoReader(vfile)
    next(vid)
    frames = []
    for frame, timestamp in vid:
        scale = cv2.resize(frame, (192, 108))
        gray = cv2.cvtColor(scale, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        if len(frames) > 30:
            break

    framecount = 80

    logger.info('generating gradient image')
    needed = np.array((-8, -4, -2)) + 8
    fs = np.expand_dims(np.array(frames)[needed], 0)
    input = torch.tensor(fs).float().requires_grad_(True)
    output = net(input)
    grad_direction = np.zeros(output.shape)
    grad_direction[0,0,80,100] = 1.
    gd = torch.tensor(grad_direction).float()
    output.backward(gd)
    grad = input.grad.detach().numpy()
    #out = output.detach().numpy()
    grad = net.conv3.weight.data.detach().numpy()
    logger.debug(grad.shape)
    grad = np.pad(grad, 1, 'constant')
    grad = np.concatenate(grad, 1)
    grad = np.concatenate(grad, 1)
    logger.debug(grad.shape)


    plt.imshow(grad)
    plt.show()



