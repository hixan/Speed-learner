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
class VideoReader:
    '''
    generator for video frame data from a file
    '''

    def __init__(self, video_path):
        '''
        :param video_path: string like /media/user/device/.../YYMMDD-hhmmss.mp4
            specifies video file location and filename.
        :return: frame, timestamp
        '''

        *path, last = video_path.split('/')
        self.cap = cv2.VideoCapture(video_path)
        self.starting_timestamp = datetime.datetime.strptime(
            last.strip(),
            'FILE%y%m%d-%H%M%S.MP4'
        )

        # gather metadata from file (not for use in this function)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.format = self.cap.get(cv2.CAP_PROP_FORMAT)

    def __iter__(self):
        # only start iterating if capture object is open
        ret = self.cap.isOpened()
        while ret:
            # calculate absolute timestamp
            relative_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = self.starting_timestamp + \
                datetime.timedelta(milliseconds=relative_timestamp)

            ret, frame = self.cap.read()  # step through to next frame
            yield frame, timestamp
            # there is no way to skip frames, all must be returned. Sampling
            # must be done therefore outside this function with no cost to
            # speed.
        self.cap.close()

# }}}


if __name__ == '__main__':
    mp4s = []
    with open('files.txt', 'r') as read:
        for rpath in read.readlines():
            path = rpath.strip()
            if path[-3:].upper() == 'MP4':
                mp4s.append(path)
    gen = VideoReader(mp4s[0])
    for fr, msec in gen:
        print(msec, end='\r')