import pandas as pd
import skimage.measure
import numpy as np
from pathlib import Path
from glob import glob
import random
from random import sample
from ..read_data import VideoReader
from pynmea2 import ParseError
from typing import List, Tuple
import json
import logging
from tqdm import tqdm
import cv2


# Script variables
DATADIR = Path('../../data')
FILE_COUNT = 1  # number of files to sample. None selects all files
random.seed(19)
plate_output = Path('plate_images/')

use_image = None  # 'test_image.jpg'  # Set to NONE if actually running script
gen_image = False


log = logging
log.basicConfig(filename="find_numberplates.log", level=0)
log.info('Runtime Begin')

sample_files : List[Path] = list(map(Path, glob(f'{DATADIR}/*/*.MP4')))
if FILE_COUNT is not None:
    sample_files = sample(sample_files, FILE_COUNT)

log.info('generated sample:')
for f in sample_files:
    log.info(f'SAMPLE: {f}')

def bounding_box(contour):
    t = np.max(contour[:,:,0])
    l = np.max(contour[:,:,1])
    b = np.min(contour[:,:,0])
    r = np.min(contour[:,:,1])
    return np.array([
        [[t,l]],
        [[t,r]],
        [[b,r]],
        [[b,l]],
    ])


def find_polygons(image, min_area: int = 50, max_area: int = 10000, sides: Tuple[int] = (4, 5, 6),
                  pooling_dims: Tuple[int, int] = (3,3), thresh_blocksize: int = 11,
                  thresh_C: int = 2, ):
    if thresh_blocksize % 2 != 1:
        raise ValueError(f'thresh_blocksize must be an odd number. Got {thresh_blocksize}')


    area_mult = pooling_dims[0] * pooling_dims[1]
    # Change min and max area to account for pooling
    #max_area /= area_mult
    #min_area /= area_mult
    im = image

    mask = im > np.array([127,127,127])
    im = mask.astype('uint8') * im

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.blur(im, (3,3))

    # for help with thresholding see
    # https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    #im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, thresh_blocksize, thresh_C)

    im = (skimage.measure.block_reduce(im, pooling_dims, np.sum) / area_mult).astype('uint8')


    #im = cv2.Canny(im, 50, 510)
    # for help with contour finding see
    # https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
    contours, h = cv2.findContours(im, 1, 5)

    quads = []
    for cont in contours:
        approx = cv2.approxPolyDP(cont, 0.1*cv2.arcLength(cont, True), True)
        # TODO remove approx or remove true from if
        if True or len(approx) in sides:
            bb = bounding_box(cont)
            area = cv2.contourArea(bb)
            if max_area >= area >= min_area:
                quads.append(bb)

    return quads, im



if gen_image:
    log.warning('Testing is set to true, wont actually calculate new images or plates')
    file = list(map(Path, ['Before Accident/FILE190419-181952.MP4']))  # TODO remove line
    log.info(f'USING FILE: {DATADIR / file}')
    video = VideoReader(DATADIR / file)
    log.info(next(video))
    for _ in tqdm(range(4000)):
        next(video)
    image, timestamp = next(video)
    cv2.imwrite(use_image, image)
elif use_image:
    frame = cv2.imread(use_image)
    quads, thresh  = find_polygons(frame,
                                    sides=(4,5,),
                                    min_area=50,
                                    max_area=1000,
                                    thresh_blocksize=5,
                                    thresh_C=8,
                                    pooling_dims=(5,5))
    #cont_drawn = cv2.drawContours(frame, quads, -1, (0, 255, 0), 3)
    #cv2.imshow('frame', cont_drawn)
    cont_thresh = cv2.drawContours(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), quads, -1, (0, 0, 255), 3)
    cv2.imshow('thresh', cont_thresh)
    cv2.waitKey(0)
else:
    plate_count = 0
    for sample in sample_files:
        vid = VideoReader(sample)
        timestamp = next(vid)
        for frame, timestamp in tqdm(vid):
            quads, thresh  = find_polygons(frame,
                                           sides=range(4,6),
                                           min_area=100,
                                           max_area=800,
                                           thresh_blocksize=51,
                                           thresh_C=5,
                                           pooling_dims=(2,2))
            #cont_drawn = cv2.drawContours(frame, quads, -1, (0, 255, 0), 3)
            #cv2.imshow('frame', cont_drawn)
            cont_thresh = cv2.drawContours(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), quads, -1, (0, 0, 255), 3)
            cont_img = cv2.drawContours(cv2.resize(frame, thresh.shape[::-1]), quads, -1, (0,0,255), 3)
            cv2.imshow('thresh', cont_thresh)
            if cv2.waitKey(3) & 0xFF == ord('q'):
                break
        vid.release()









