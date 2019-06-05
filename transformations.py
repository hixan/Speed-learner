import cv2
import numpy as np


class settings:  # {{{
    ret_names = True
    verbose = False
# }}}


def transform_name(name, *args, **kwargs):  # {{{
    '''generate names with consistent format according to arguments
    :param name: name of transformation
    :param *args: unnamed arguments to be included in the name of the transform
    :param kwargs: named arguments include name and value in the transform
    '''
    rval = name
    if len(args) > 0:
        rval += '-' + '-'.join(args)
    elif len(kwargs) > 0:
        rval += '--'
    if len(kwargs) > 0:
        rval += '--'.join(f'{key}={value}' for key, value in kwargs.items())

    return rval
# }}}


def resize(stream, scale=.1, dims=None):  # {{{
    '''downsize an image stream
    :param stream: iterable of (frame, identifier) pairs
    :param scale: scaling factor - default .1 (ignored if dims is used)
    :param dims: (width, height) of wanted size
    '''
    if settings.ret_names:
        yield (*next(stream), transform_name('downsize', scale=str(scale)))

    if dims is None:  # calculate new_dims from old dimensions
        # get old dimensions from stream
        frame, timestamp = next(stream)

        # numpy indexing is reversed
        new_dims = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
        if settings.verbose:
            print('downsize:', frame.shape, timestamp)
        yield cv2.resize(frame, new_dims), timestamp
    else:
        new_dims = dims
    for frame, timestamp in stream:
        if settings.verbose:
            print('downsize:', frame.shape, timestamp)
        yield cv2.resize(frame, new_dims), timestamp
# }}}


def running_average(stream, window_size=10, skip_incomplete=False):  # {{{
    '''creates an average over a window length $window_size.
    :param stream: iterable of (frame, identifier) pairs
    :param window_size: number of frames to be included in running average
    :param skip_incomplete: if true treat video with padding to enable output
        to be the same length as the input.
    '''
    if settings.ret_names:
        yield (*next(stream),
               transform_name('running_average', window_size=str(window_size)))

    frame, timestamp = next(stream)

    # initialize running conditions
    window_objs = [frame / window_size]
    running_average = np.copy(window_objs[0])  # == sum(window_objs) always

    for frame, timestamp in stream:
        # this frame has a weight of 1/window_size
        this_frame = frame / window_size
        running_average += this_frame
        window_objs.append(this_frame)
        # only contribute average values that have appropriate weight -
        # (window_size * 1/window_size)
        if len(window_objs) == window_size:
            if settings.verbose:
                print(f'running_average({window_size}):',
                      running_average.shape, timestamp)
            yield running_average.astype(np.uint8), timestamp
            running_average -= window_objs[0]
            del window_objs[0]
        elif not skip_incomplete:
            yield running_average, timestamp
# }}}


def edge_detect(stream, threshold1, threshold2):  # {{{
    ''' runs a Canny edge detection algorithm over each frame. The smaller of
    threshold1 and threshold2 is used for edge linking. The largest is used to
    find initial segments of strong edges. See
    <http://en.wikipedia.org/wiki/Canny_edge_detector>
    :param stream: iterable of (frame, identifier) pairs
    :param threshold1: first threshold value
    :param threshold2: second threshold value
    '''
    if settings.ret_names:
        yield (*next(stream),
               transform_name('edge_detect', t1=threshold1, t2=threshold2))
    for frame, timestamp in stream:
        if settings.verbose:
            print('edge_detect:', frame.shape, timestamp)
        yield cv2.Canny(frame, threshold1, threshold2), timestamp
# }}}


def limit(stream, framecount):  # {{{
    '''limit the number of frames of stream to framecount. If stream contains
    less then $framecount pairs this has no effect.
    :param stream: iterable of (frame, identifier) pairs
    :param framecount: maximum number of frames to output.
    '''
    if settings.ret_names:
        yield (*next(stream), transform_name('limit', n=framecount))

    for i, (frame, timestamp) in enumerate(stream):
        if i == framecount:
            if settings.verbose:
                print('Limit:', framecount)
            break
        yield frame, timestamp
# }}}


def write_video(stream, fps, actions=None, dirs=('results',)):  # {{{
    if settings.ret_names:
        actions = next(stream)
    frame, timestamp = next(stream)
    if len(frame.shape) == 3:
        is_colour = True
    elif len(frame.shape) == 2:
        is_colour = False
    else:
        assert False

    if settings.verbose:
        print('write_video:', frame.shape, timestamp)
    name = '/'.join(dirs) + '/' + '_'.join(actions) + '.avi'

    # initialize video writing
    out = cv2.VideoWriter(
        name,
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        frame.shape[:2][::-1],  # numpy uses matrix indexing so must be reverse
        is_colour
    )

    # write all frames to file
    out.write(frame)
    for frame, timestamp in stream:
        if settings.verbose:
            print('write_video:', frame.shape, timestamp)
        out.write(frame)
    out.release()
    return name
# }}}
