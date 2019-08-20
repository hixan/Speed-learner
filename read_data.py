import cv2
import pynmea2
from collections import namedtuple
from datetime import datetime, timedelta
from transformations import settings
import pandas as pd
from pathlib import Path
from math import floor


GSENSORD = namedtuple('GSENSORD', [*'xyz'])
GPSDC = namedtuple('GPSDC', ())


def parse_nmea_line(line):  # handle sentances pynmea2 cant handle {{{
    start, *rest = line.strip().split(',')
    if start == '$GSENSORD':  # g-sensor data
        return GSENSORD(*rest)
    elif start == '$GPSDC':  # gps disconnected as far as i can tell
        return GPSDC()
    return pynmea2.parse(line)
# }}}


def read_nmea_file(filename):  # {{{
    with open(filename, 'r') as nmea_file:
        return tuple(parse_nmea_line(line) for line in nmea_file)


# }}}

class NmeaFile:  # {{{
    # reads nmea file and adjusts timestamps to correspond to video timestamps
    #
    # useful nmea sentances:
    # GSENSORD - gsensor data - 3 values correspond to x, y, z, TODO check
    # order

    # GPRMC - position velocity time. contains timestamp, latitude, longitude,
    #   speed in knots, track angle in degrees, date

    # GPGGA - fix data includes altitude.

    # not useful:
    # GPGSA - sattelite info
    # GPGSV - sattelite info

    @staticmethod
    def DataFrame(filepath: Path,  # {{{
                  timestampformat: str = 'FILE%y%m%d-%H%M%S.NMEA'):

        columns = ('time', 'latitude', 'longitude', 'sense_x', 'sense_y',
                   'sense_z', 'speed', 'direction', 'video_time',
                   'video_file')

        parsed = read_nmea_file(str(filepath))
        # throw away gpgsv data which happens to be at positions n, n+4, n+5,
        # n+6, n+7 and group other types (gsensor, rmc, gga)
        piter = iter(parsed)
        observations = []
        while True:
            try:
                n = next(piter)
                while type(n) is not GSENSORD:
                    n = next(piter)
                n2 = next(piter)
                if type(n2) is not pynmea2.types.talker.RMC:
                    n2 = parse_nmea_line('$GPRMC,,,,,,,,,,,,')  # no data
                observations.append((n, n2))
            except StopIteration:
                break

        starting_timestamp = None  # will store the first timestamp in series
        vals = {c: [] for c in columns}

        # add observations to the dataframe
        for gsense, rmc in observations:
            assert type(gsense) is GSENSORD and \
                type(rmc) is pynmea2.types.talker.RMC
            tstamp = datetime.combine(rmc.datestamp, rmc.timestamp)
            speed = rmc.spd_over_grnd * 1.852  # speed in km, originally knots
            direction = rmc.true_course  # direction in degrees from N
            if starting_timestamp is None:
                starting_timestamp = tstamp
            vtimestamp = tstamp - starting_timestamp
            mul_lat = -1 if rmc.lat_dir == 'S' else 1
            mul_lon = -1 if rmc.lon_dir == 'W' else 1
            lat = NmeaFile._ddm_to_dd(rmc.lat, mul_lat)
            lon = NmeaFile._ddm_to_dd(rmc.lon, mul_lon)

            # TODO - categorical video file?
            vals['time'].append(tstamp)
            vals['speed'].append(speed)
            vals['longitude'].append(lon)
            vals['latitude'].append(lat)
            vals['direction'].append(direction)
            vals['sense_x'].append(float(gsense.x))
            vals['sense_y'].append(float(gsense.y))
            vals['sense_z'].append(float(gsense.z))
            vals['video_time'].append(vtimestamp)
            vals['video_file'].append(filepath.stem + '.MP4')

        return pd.DataFrame(vals)  # }}}

    @staticmethod
    def _ddm_to_dd(ddm: str, mult: int = 1) -> float:
        deg = int(ddm[:ddm.index('.')-2])
        sec = float(ddm[ddm.index('.')-2:])
        return (deg + sec/60) * mult

    @staticmethod
    def _dd_to_dms(dd: float, choice: (str, str) = ('N', 'S')) -> \
            (int, int, int, str):

        if dd < 0:
            rs = choice[1]
            dd = -dd
        else:
            rs = choice[0]

        degs = floor(dd)
        m = (dd - degs) * 60
        mins = floor(m)
        secs = (m - mins) * 60
        return degs, mins, secs, rs
# }}}


class VideoReader:  # {{{
    '''
    generator for video frame data from a file
    '''

    def __init__(self, video_path, starting_timestamp=None):  # {{{
        '''
        :param video_path: string like /media/user/device/.../YYMMDD-hhmmss.mp4
            specifies video file location and filename.
        :return: frame, timestamp
        '''

        *path, last = video_path.split('/')
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if starting_timestamp is None:
            self.starting_timestamp = datetime.strptime(
                last.strip(),
                'FILE%y%m%d-%H%M%S.MP4'
            )
        else:
            self.starting_timestamp = starting_timestamp

        # gather metadata from file (not for use in this function)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.format = self.cap.get(cv2.CAP_PROP_FORMAT)
        self._ret_names = settings.ret_names
        # only start iterating if capture object is open
        self.ret = self.cap.isOpened()
        # }}}

    def __next__(self):  # {{{
        if self._ret_names:
            self._ret_names = False  # only return once
            return self.starting_timestamp.strftime('%y%m%d-%H%M%S%f'),

        # calculate absolute timestamp
        relative_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = self.starting_timestamp + \
            timedelta(milliseconds=relative_timestamp)

        self.ret, frame = self.cap.read()  # step through to next frame

        if not self.ret:  # stop iterator if no more frames
            self.cap.release()
            raise StopIteration(f'file {self.video_path} has no more frames.')

        return frame, timestamp
        # there is no way to skip frames, all must be returned. Sampling
        # can be done outside this function with no additional cost to
        # efficiency.
        # }}}

    def __iter__(self):  # {{{
        return self
    # }}}

    def release(self):  # {{{
        self.cap.release()
    # }}}

# }}}


if __name__ == '__main__':
    print(NmeaFile.DataFrame(Path('example_data/FILE180603-225817.NMEA')).describe())
