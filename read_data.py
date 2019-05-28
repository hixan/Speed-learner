import cv2
import pynmea2
from collections import namedtuple
import pandas as pd
import datetime
import textwrap

# set up path variables
mp4s = []
with open('files.txt', 'r') as read:
    for rpath in read.readlines():
        path = rpath.strip()
        if path[-3:].upper() == 'MP4':
            mp4s.append(path)


GSENSORD = namedtuple('GSENSORD', [*'xyz'])
GPSDC = namedtuple('GPSDC', ())


def parse_nmea_line(line):  # handle sentances pynmea2 cant handle
    start, *rest = line.strip().split(',')
    if start == '$GSENSORD':  # g-sensor data
        return GSENSORD(*rest)
    elif start == '$GPSDC':  # gps disconnected as far as i can tell
        return GPSDC()
    return pynmea2.parse(line)


def read_file(video_path, sample_function=None, read_nmea=True,
              gps_assignment='floor'):
    '''video_path like /media/user/device/.../file_identifier.mp4
    sample_function returns true if frame number is to be included in sample'''

    print(f'reading {video_path}')
    *path, last = video_path.split('/')

    # get starting timestamp from filename (in this dataset, filenames are
    # formatted FILEyymmdd-hhmmss.extension
    starting_timestamp = datetime.datetime.strptime(last.strip(),
                                                    'FILE%y%m%d-%H%M%S.MP4')
    # handle nmea if necessary
    if read_nmea:
        nmea_path = video_path[:-4] + '.NMEA'  # todo use actual pathlib
        with open(nmea_path, 'r') as nmea_file:
            nmea_data = [parse_nmea_line(line) for line in nmea_file]
    else:
        nmea_data = None

    cap = cv2.VideoCapture(video_path)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if sample_function is None:
        def sample_function(n):
            return True
    frame_number = 0
    ret = cap.isOpened()  # only loop if capture object opened correctly
    while ret:  # keep reading until frame doesnt exist
        frame_number += 1

        ret, frame = cap.read() # step through to next frame

        # yield data if contained in sample
        if sample_function(frame_number):

            relative_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

            # calculate absolute timestamp
            timestamp = starting_timestamp + \
                datetime.timedelta(milliseconds=relative_timestamp)

            yield frame_number, frame, timestamp


for fn, fr, msec in read_file(mp4s[0], lambda x: x % 1 == 0):
    print(msec)
