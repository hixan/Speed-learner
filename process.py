from processing.frame_prediction.generate_train import generate_training_data
import numpy as np
from pathlib import Path
from glob import glob
from random import sample
from tqdm import tqdm
from typing import List

# Script variables
DATADIR = Path('data')
FILE_COUNT = 2  # number of files to sample. None selects all files

sample_files : List[Path] = list(map(Path, glob(f'{DATADIR}/*/*.MP4')))
if FILE_COUNT is not None:
    sample_files = sample(sample_files, FILE_COUNT)

for f in tqdm(sample_files):
    print(f)
    generate_training_data(f, Path('./processed_data'), 20, np.array((0, -1, -2, -3)), .3)
