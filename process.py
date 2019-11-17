from processing.frame_prediction.generate_train import generate_training_data
import numpy as np
from pathlib import Path
from glob import glob
from random import sample
from tqdm import tqdm
from typing import List
import logging
import json

logging.INFOFRAME = 21
logging.INFOFILE = 22
logging.addLevelName(logging.INFOFRAME, 'INFOFRAME')
logging.addLevelName(logging.INFOFILE, 'INFOFILE')

logger = logging.getLogger('process')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('process.log', 'w')
fh.setLevel(logging.INFOFRAME)
fmtstr = '%(name)s:%(levelname)s: %(asctime)s \n%(message)s'
fh.setFormatter(logging.Formatter(fmtstr))
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(fmtstr.replace('\n', '')))
ch.setLevel(logging.INFOFILE)
logger.addHandler(fh)
logger.addHandler(ch)

if __name__ == '__main__':

    generate_trainset = True

    if generate_trainset:
        log = logger.getChild('generate_trainset')
        # Script variables
        DATADIR = Path('data')
        FILE_COUNT = 2  # number of files to sample. None selects all files
        OUTPUT_DIR = Path('./processed_data/frame_prediction/')

        sample_files : List[Path] = list(map(Path, glob(f'{DATADIR}/*/*.MP4')))
        if FILE_COUNT is not None:
            sample_files = sample(sample_files, FILE_COUNT)

        meta = []
        for f in tqdm(sample_files, desc='files'):
            meta.extend(generate_training_data(
                file=f,
                output_dir=OUTPUT_DIR,
                sample_size=5,
                relative_chain=np.array((0, -2, -4, -6)),
                rescale=.3,
                logger=log
            ))
        with open(OUTPUT_DIR / 'meta.json', 'w') as f:
            print(json.dumps(meta), file=f)
