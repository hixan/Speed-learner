from ..read_data import VideoReader
import json
from random import sample
import numpy as np  # type: ignore

from functools import reduce
import cv2  # type: ignore
from pathlib import Path
from operator import or_
from typing import Tuple, Union, Set, Dict, Callable
import logging


def reduce_meta_files(l1: Dict, l2: Dict):
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


def generate_training_data(
        vid: Union[Path, VideoReader],
        output_dir: Path,
        average_sps: float,  # average samples per second
        relative_chain: Tuple[int, ...],
        transform: Union[Callable, None] = None,
        timefmt: str = '%y%m%d-%H%M%S%f',
        logger: Union[logging.Logger, None] = None,
        check_processed: bool = True
):
    if logger is None:
        # will log messages but wont output anywhere
        logger = logging.Logger(f'{__name__}: generate_training_data')
    if isinstance(vid, Path):
        vid = VideoReader(vid)

    args = {
        'filepath': str(vid.video_path),
        'output_dir': str(output_dir),
        'average_sps': average_sps,
        'relative_chain': list(relative_chain),
        'transform': transform is not None,
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
    needed_frames: Set[int] = reduce(or_, map(set, framegroups), set())
    saved_frames: Dict[int, str] = {}

    for i, (frame, timestamp) in enumerate(vid):
        if i in needed_frames:
            frame_filename = output_dir / 'frames' / timestamp.strftime(
                f'{timefmt}.bmp'
            )

            if transform is not None:
                frame = transform(frame)

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
            {
                'y': saved_frames[y],
                'x': list(map(lambda b: saved_frames[b], x))
            } for y, *x in framegroups
        ]
    }
    logger.log(logging.INFO, f'finished processing {vid.video_path}.')

    with open(output_file, 'w') as f:
        print(json.dumps(output_meta), file=f)
    return output_meta
