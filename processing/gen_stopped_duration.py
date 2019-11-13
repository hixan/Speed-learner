import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from random import sample
from read_data import NmeaFile
from pynmea2 import ParseError
from typing import List
import json
import logging
from tqdm import tqdm


# Script variables
DATADIR = Path('../data')
FILE_COUNT = None  # number of files to sample. None selects all files

logger = logging.Logger('warnings')

sample_files : List[Path] = list(map(Path, glob(f'{DATADIR}/*/*.NMEA')))
if FILE_COUNT is not None:
    sample_files : List[Path] = sample(sample_files, FILE_COUNT)

stopped_durations = []

for f in tqdm(sample_files):
    try:
        nmea = NmeaFile.DataFrame(f)
    except ParseError:
        logger.warning(f'\rcould not parse file {f}')
        continue  # skip this file

    # should only come from one file.
    try:
        assert nmea['video_file'].describe()['unique'] == 1
        assert nmea['nmea_file'].describe()['unique'] == 1
    except KeyError:
        logger.warning(f'\rthere was a problem with file {f}')

    # detect stopped sections

    current_stopped = False
    threshold = 0

    for timestamp, speed in nmea[['video_timestamp', 'speed']].values:
        if not current_stopped:
            if speed <= threshold:
                start_stopped = timestamp
                current_stopped = True
        else:
            if speed > threshold:
                stopped_durations.append(
                    timestamp - start_stopped
                )
                current_stopped = False
    if current_stopped:  # add last loud section if necessary.
        stopped_durations.append(timestamp + 1 - start_stopped)

with open('../visualizations/stopped_durations.json', 'w') as f:
    print(json.dumps(stopped_durations), file=f)
print('done.')
