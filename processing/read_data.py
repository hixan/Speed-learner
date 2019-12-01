import cv2  # type: ignore
import pynmea2  # type: ignore
from datetime import datetime, timedelta
from .transformations import settings
import pandas as pd  # type: ignore
from pathlib import Path
from math import floor
from typing import NamedTuple, Dict, List, Tuple

GSENSORD = NamedTuple('GSENSORD', (('x', int), ('y', int), ('z', int)))
GPSDC = NamedTuple('GPSDC', ())


def parse_nmea_line(line):  # handle sentances pynmea2 cant handle
    start, *rest = line.strip().split(',')
    if start == '$GSENSORD':  # g-sensor data
        return GSENSORD(*rest)
    elif start == '$GPSDC':  # gps disconnected as far as i can tell
        return GPSDC()
    try:
        return pynmea2.parse(line)
    except pynmea2.ParseError as e:
        raise e


def read_nmea_file(filename):
    with open(filename, 'r') as nmea_file:
        return tuple(parse_nmea_line(line) for line in nmea_file)


class NmeaFile:
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
    def DataFrame(filepath: Path,
                  timestampformat: str = 'FILE%y%m%d-%H%M%S.NMEA'):
        '''read nmea file PATH and return a dataframe.
        This assumes the NMEA file is of the following format:
            GPGSV (useless)
            GSENSORD - usefull - gsensor data
            GPRMC - usefull - lon,lat, speed etc.
            other stuff

        Columns of the dataframe is as follows:
|column name    |type    |description                                         |
|---------------+--------+----------------------------------------------------|
|time           |DateTime|timestamp from RMC                                  |
|speed          |float   |speed in Km/h from RMC                              |
|longitude      |float   |longitude from RMC                                  |
|latitude       |float   |latitude from RMC                                   |
|direction      |float   |direction from RMC (in degrees clockwise from north)|
|sense_x        |float   |GSensor Data (from GSENSORD sentance)               |
|sense_y        |float   |GSensor Data (from GSENSORD sentance)               |
|sense_z        |float   |GSensor Data (from GSENSORD sentance)               |
|video_timestamp|float   |timestamp of reading wrt start of video             |
|video_file     |str     |filename of the corresponding video file            |
|nmea_file      |str     |filename of the corresponding nmea file             |
|directory      |str     |directory of video and nmea file                    |
|---------------+--------+----------------------------------------------------|
        '''

        columns = ('time', 'latitude', 'longitude', 'sense_x', 'sense_y',
                   'sense_z', 'speed', 'direction', 'video_timestamp',
                   'nmea_file', 'video_file', 'directory')

        parsed = read_nmea_file(str(filepath))

        piter = iter(parsed)
        observations = []
        while True:
            try:
                n = next(piter)
                while type(n) is not GSENSORD:
                    n = next(piter)  # skip to the next GSENSORD line
                n2 = next(piter)
                if type(n2) is not pynmea2.types.talker.RMC:
                    n2 = parse_nmea_line('$GPRMC,,,,,,,,,,,,')  # no data
                observations.append((n, n2))
            except StopIteration:
                break

        starting_timestamp = None  # will store the first timestamp in series
        vals: Dict[str, List] = {c: [] for c in columns}

        # add observations to the dataframe
        for gsense, rmc in observations:
            assert type(gsense) is GSENSORD and \
                type(rmc) is pynmea2.types.talker.RMC
            tstamp = datetime.combine(rmc.datestamp, rmc.timestamp)
            speed = rmc.spd_over_grnd * 1.852  # speed in km, originally knots
            direction = rmc.true_course  # direction in degrees from N
            if starting_timestamp is None:
                starting_timestamp = tstamp
            vtimestamp = (tstamp - starting_timestamp).total_seconds()
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
            vals['video_timestamp'].append(vtimestamp)
            vals['video_file'].append(
                filepath.with_suffix('.MP4').name)  # file name with mp4 suffix
            vals['nmea_file'].append(filepath.name)  # file name with extension
            vals['directory'].append(filepath.parent)

        return pd.DataFrame(vals)

    @staticmethod
    def _ddm_to_dd(ddm: str, mult: int = 1) -> float:
        '''converts degree decimal minutes to decimal degrees'''
        deg = int(ddm[:ddm.index('.')-2])
        sec = float(ddm[ddm.index('.')-2:])
        return (deg + sec/60) * mult

    @staticmethod
    def _dd_to_dms(dd: float, choice: Tuple[str, str] = ('N', 'S')) -> \
            Tuple[int, int, float, str]:
        '''converts decimal degrees to degree minutes seconds'''

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


class VideoReader:
    '''
    generator for video frame data from a file
    '''

    def __init__(self, video_path: Path, starting_timestamp=None):
        '''Wrapper for cv2.capture object, specific to the case of video where
        title has the timestamp of the beginning of the video encoded. (And
        has an NMEA file associated with the same name, but is not as
        important)
        :param video_path: string like /media/user/device/.../YYMMDD-hhmmss.mp4
            specifies video file location and filename.
        :return: frame, timestamp
        :raise OSError: The video filepath does not exist
        '''
        if not video_path.exists():
            raise OSError(f'file path {video_path} does note exist.')

        *path, last = video_path.parts
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path.absolute()))
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

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __next__(self):
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

        # there is no way to skip frames, all must be returned. Sampling
        # can be done outside this function with no additional cost to
        # efficiency.
        return frame, timestamp

    def __iter__(self):
        return self

    def release(self):
        self.cap.release()


if __name__ == '__main__':
    print(NmeaFile.DataFrame(Path('example_data/FILE180603-225817.NMEA'))
          .describe())
