from ..read_data import VideoReader
from random import sample
import numpy as np
from functools import reduce
import cv2
from pathlib import Path
from typing import List
from tqdm import tqdm
from operator import or_

def generate_training_data(file: Path, output_dir: Path, sample_size: int, relative_chain, rescale: int, timefmt: str = '%y%m%d-%H%M%S%f'):
    vid = VideoReader(file)
    next(vid)
    framegroups = list(map(lambda x: list(relative_chain + x), sorted(sample(range(-min(relative_chain), len(vid)-max(relative_chain)), sample_size))))
    framegroups = [[1,2,3],[2,3,4]]
    print(framegroups)
    needed_frames = reduce(or_, map(set, framegroups), set())
    saved_frames = {}
    for i, (frame, timestamp) in enumerate(vid):
        if i in needed_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, tuple((np.array(frame.shape) * rescale).astype('uint8')[::-1]))
            print(f'added frame at {i}')
            saved_frames[i] = frame, timestamp
            needed_frames.remove(i)
            if len(needed_frames) == 0:
                break

    for framegroup in framegroups:
        frames = [saved_frames[i] for i in framegroup]
        groupname = frames[0][1].strftime(timefmt)
        cv2.imwrite(str(output_dir / 'y' / (groupname + '.jpg')), frames[0][0])
        for frame in frames[1:]:
            cv2.imwrite(str(output_dir / 'x' / (groupname + '_' + f'{frame[1].strftime(timefmt)}.jpg')), frame[0])



