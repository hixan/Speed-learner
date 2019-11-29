from ..read_data import VideoReader
import json
from random import sample
import numpy as np
from functools import reduce
import cv2
from pathlib import Path
from typing import List
from operator import or_
from typing import Tuple, Union
import logging

def generate_training_data(
        vid: Union[Path, VideoReader],
        output_dir: Path,
        average_sps: float,  # average samples per second
        relative_chain: Tuple[int, ...],
        rescale: int = .3,
        timefmt: str = '%y%m%d-%H%M%S%f',
        logger: Union[logging.Logger, None] = None,
        check_processed: bool = True
):
    if logger is None:
        logger = logging.Logger()  # will log messages but wont output anywhere
    if type(vid) is not VideoReader:
        try:
            vid : VideoReader = VideoReader(vid)
        except OSError:
            raise
    args = {
        'filepath': str(vid.video_path),
        'output_dir': str(output_dir),
        'average_sps': average_sps,
        'relative_chain': list(relative_chain),
        'rescale': rescale,
        'timefmt': timefmt,
    }
    output_file = (
        output_dir /
        f'meta_{vid.starting_timestamp.strftime(timefmt)}.json'
    )
    if check_processed and output_file.exists():
        # dont re-evaluate frames if computation is the same. Just load return
        # value from previous run.
        with open(output_file, 'r') as f:
            meta_old = json.load(f)
            if meta_old['args'] == args:
                return meta_old
    next(vid)  # skip metadata
    sample_size = int(vid.frame_count / vid.fps * average_sps)
    framegroups = list(map(
        lambda x: list(np.array(relative_chain) + x),
        sorted(sample(range(
            -min(relative_chain),
            len(vid)-max(relative_chain)
        ), sample_size))
    ))
    needed_frames = reduce(or_, map(set, framegroups), set())
    saved_frames = {}

    for i, (frame, timestamp) in enumerate(vid):
        if i in needed_frames:
            frame_filename = output_dir / 'frames' / timestamp.strftime(
                f'{timefmt}.bmp'
            )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scaled_size = (np.array(frame.shape[::-1]) *
                        rescale).astype('uint8')
            frame = cv2.resize(frame, tuple(scaled_size))

            logger.log(
                logging.INFO,
                f'added frame {i} from {vid.video_path} as {frame_filename}'
            )

            cv2.imwrite(str(frame_filename), frame)
            saved_frames[i] = frame_filename.name
            needed_frames.remove(i)
            if len(needed_frames) == 0:
                break
    logger.log(logging.INFO,
            f'processed {len(saved_frames)} frames from {vid.video_path}.')
    output_meta = {
        'args': args,
        'data': [
            {'y': saved_frames[y], 'x': list(map(lambda b: saved_frames[b], x))}
            for y, *x in framegroups
        ]
    }
    logger.log(logging.INFO, f'finished processing {vid.video_path}.')

    with open(output_file, 'w') as f:
        print(json.dumps(output_meta), file=f)
    return output_meta
