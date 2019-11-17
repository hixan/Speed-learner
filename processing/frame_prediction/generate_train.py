from ..read_data import VideoReader
from random import sample
import numpy as np
from functools import reduce
import cv2
from pathlib import Path
from typing import List
from tqdm import tqdm
from operator import or_
from typing import Tuple, Union
import logging


def generate_training_data(file: Path, output_dir: Path, sample_size: int, relative_chain: Tuple[int, ...], rescale: int = .3, timefmt: str = '%y%m%d-%H%M%S%f', logger: Union[logging.Logger, None] = None):

    vid = VideoReader(file)
    next(vid)
    framegroups = list(map(lambda x: list(relative_chain + x), sorted(sample(range(-min(relative_chain), len(vid)-max(relative_chain)), sample_size))))
    needed_frames = reduce(or_, map(set, framegroups), set())
    saved_frames = {}

    for i, (frame, timestamp) in tqdm(enumerate(vid), total=max(needed_frames), desc='frames'):
        if i in needed_frames:

            frame_filename = output_dir / 'frames' / timestamp.strftime(timefmt + '.jpg')

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scaled_size = (np.array(frame.shape[::-1]) * rescale).astype('int64')
            frame = cv2.resize(frame, tuple(scaled_size))

            logger.log(logging.INFOFRAME, f'added frame {i} from {vid.video_path} as {frame_filename}')

            cv2.imwrite(str(frame_filename), frame)
            saved_frames[i] = frame_filename.name
            needed_frames.remove(i)
            if len(needed_frames) == 0:
                break
    logger.log(logging.INFOFILE, f'processed {len(saved_frames)} frames from {file}.')
    output_meta = [{'y': saved_frames[y], 'x': list(map(lambda b: saved_frames[b], x))} for y, *x in framegroups]
    logger.log(logging.INFOFILE, f'finished processing {file}.')
    return output_meta
