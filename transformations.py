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
        frame, identifier = next(stream)

        # numpy indexing is reversed
        new_dims = int(frame.shape[1] * scale), int(frame.shape[0] * scale)
        if settings.verbose:
            print('downsize:', frame.shape, identifier)
        yield cv2.resize(frame, new_dims), identifier
    else:
        new_dims = dims
    for frame, identifier in stream:
        if settings.verbose:
            print('downsize:', frame.shape, identifier)
        yield cv2.resize(frame, new_dims), identifier
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

    frame, identifier = next(stream)

    # initialize running conditions
    window_objs = [frame / window_size]
    running_average = np.copy(window_objs[0])  # == sum(window_objs) always

    for frame, identifier in stream:
        # this frame has a weight of 1/window_size
        this_frame = frame / window_size
        running_average += this_frame
        window_objs.append(this_frame)
        # only contribute average values that have appropriate weight -
        # (window_size * 1/window_size)
        if len(window_objs) == window_size:
            if settings.verbose:
                print(f'running_average({window_size}):',
                      running_average.shape, identifier)
            yield running_average.astype(np.uint8), identifier
            running_average -= window_objs[0]
            del window_objs[0]
        elif not skip_incomplete:
            yield running_average, identifier
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
    for frame, identifier in stream:
        if settings.verbose:
            print('edge_detect:', frame.shape, identifier)
        yield cv2.Canny(frame, threshold1, threshold2), identifier
# }}}


def limit(stream, framecount):  # {{{
    '''limit the number of frames of stream to framecount. If stream contains
    less then $framecount pairs this has no effect.
    :param stream: iterable of (frame, identifier) pairs
    :param framecount: maximum number of frames to output.
    '''
    if settings.ret_names:
        yield (*next(stream), transform_name('limit', n=framecount))

    for i, (frame, identifier) in enumerate(stream):
        if i == framecount:
            if settings.verbose:
                print('Limit:', framecount)
            break
        yield frame, identifier
# }}}


def mask(stream1, stream2, inverse=False):  # {{{
    '''performs a per element multiplication between stream 1 and stream 2.
    returns identifier from the stream with most channels. If both have the
    same default to stream1 identifiers'''
    gen = zip(stream1, stream2)
    if settings.ret_names:
        names1, names2 = next(gen)
        yield ('(', *names1, transform_name('weight_mask'), *names2, ')')

    for (f1, id1), (f2, id2) in gen:

        if len(f1.shape) == len(f2.shape) and f1.shape != f2.shape:
            raise ValueError(f'stream frames must have the same dimensions,' +
                             f' recieved {f1.shape} and {f2.shape}')

        if len(f1.shape) > len(f2.shape):
            f1, f2 = f2, f1  # channelled image always
            id1, id2 = id2, id1
        if len(f1.shape) < len(f2.shape):
            f1 = np.concatenate([f1[..., np.newaxis]]*f2.shape[2], axis=2)

        if inverse:
            yield np.multiply(f2, (256 - f1) / 256).astype(np.uint8), id1
        yield np.multiply(f2, f1 / 256).astype(np.uint8), id1
# }}}


def write_video(stream, fps, actions=None, dirs=('results',)):  # {{{
    if settings.ret_names:
        actions = next(stream)
    frame, identifier = next(stream)
    if len(frame.shape) == 3:
        is_colour = True
    elif len(frame.shape) == 2:
        is_colour = False
    else:
        assert False

    if settings.verbose:
        print('write_video:', frame.shape, identifier)
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
    for frame, identifier in stream:
        if settings.verbose:
            print('write_video:', frame.shape, identifier)
        out.write(frame)
    out.release()
    return name
# }}}
