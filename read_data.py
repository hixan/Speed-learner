import cv2
import pynmea2
from collections import namedtuple
import datetime

# set up path variables

GSENSORD = namedtuple('GSENSORD', [*'xyz'])
GPSDC = namedtuple('GPSDC', ())


# {{{
def parse_nmea_line(line):  # handle sentances pynmea2 cant handle
    start, *rest = line.strip().split(',')
    if start == '$GSENSORD':  # g-sensor data
        return GSENSORD(*rest)
    elif start == '$GPSDC':  # gps disconnected as far as i can tell
        return GPSDC()
    return pynmea2.parse(line)
# }}}


# {{{
def read_nmea_file(filename):
    with open(filename, 'r') as nmea_file:
        return [parse_nmea_line(line) for line in nmea_file]
# }}}


# {{{
def read_video_file(video_path):
    ''' generator for video frame data from a file
    :param video_path: string like /media/user/device/.../YYMMDD-hhmmss.mp4
        specifies video file location and filename.
    :return: frame, timestamp
    '''

    print(f'reading {video_path}')
    *path, last = video_path.split('/')

    # get starting timestamp from filename (in this dataset, filenames are
    # formatted FILEyymmdd-hhmmss.extension
    starting_timestamp = datetime.datetime.strptime(last.strip(),
                                                    'FILE%y%m%d-%H%M%S.MP4')
    cap = cv2.VideoCapture(video_path)

    ret = cap.isOpened()  # only loop if capture object opened correctly
    while ret:  # keep reading until frame doesnt exist

        # calculate absolute timestamp
        relative_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = starting_timestamp + \
            datetime.timedelta(milliseconds=relative_timestamp)


        ret, frame = cap.read()  # step through to next frame
        yield frame, timestamp
        # there is no way to skip frames, all must be returned. Sampling must
        # be done therefore outside this function with no cost to speed.
# }}}


if __name__ == '__main__':
    mp4s = []
    with open('files.txt', 'r') as read:
        for rpath in read.readlines():
            path = rpath.strip()
            if path[-3:].upper() == 'MP4':
                mp4s.append(path)
    for fr, msec in read_video_file(mp4s[0], lambda x: x % 1 == 0):
        print(msec, end='\r')
