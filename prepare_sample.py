from read_data import VideoReader, read_nmea_file
import cv2
import numpy as np
import threading
import concurrent.futures
import random
from time import sleep

# variables about sample size
file_count = 4
frame_count = 100
thread_count = 1

dataset = []
dataset_lock = threading.Lock()

thread_number = 0
thread_number_lock = threading.Lock()


# therad worker function
def generate_data(videofile):
    frame_generator = VideoReader(videofile)
    frame_sample = sorted(random.sample(range(1, frame_generator.frame_count),
                                        frame_count))
    previous_frame = None
    for i, (frame, timestamp) in enumerate(frame_generator):
        if len(frame_sample) == 0:
            break
        if frame_sample[0] != i:
            previous_frame = frame
            continue  # skip this frame
        del frame_sample[0]  # this frame index is in the past now
        difference = np.absolute(previous_frame - frame)
        to_append = previous_frame, frame, difference, timestamp
        with dataset_lock:
            dataset.append(to_append)
        print(i)


if __name__ == '__main__':
    mp4s = []
    with open('files.txt', 'r') as read:
        for rpath in read.readlines():
            path = rpath.strip()
            if path[-3:].upper() == 'MP4':
                mp4s.append(path)

    file_samples = random.sample(mp4s, file_count)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=thread_count
    ) as executor:
        executor.map(generate_data, file_samples)
    for prev, curr, diff, timestamp in dataset:
        cv2.imwrite(f'results/{timestamp}-diff.png', diff)
        cv2.imwrite(f'results/{timestamp}-frames.png',
                    np.concatenate(prev, curr, axis=1))
    print('done.')
