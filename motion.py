import cv2
import numpy as np
from video_helpers import tile
from functools import reduce

#history 30 threshold 3 misses all events
#history 30 threshold 1 misses all events
#history 60 threshold 1 misses 2

from video_helpers import Viewer


def difference_mask(v, threshold=15, kernel_size=3):

    open_kernel = np.ones(kernel_size * kernel_size).reshape((kernel_size, kernel_size))
    v = v.astype('float32')
    mask = np.zeros((v.shape), dtype='float32')

    # Compute difference
    mask[1:] = v[:-1] - v[1:]
    mask[0] = mask[1]

    # Make positive
    mask = mask**2

    # Create mask from threshold
    mask[mask >= threshold] = 255
    mask[mask < threshold] = 0

    # Remove noise
    mask[:] = [cv2.morphologyEx(frame,
                                cv2.MORPH_OPEN,
                                open_kernel) for frame in mask]

    return mask.astype('uint8')


def make_mask(vid):
    vid = vid.astype('uint8')

    mask = difference_mask(vid)
    # tiled = tile([[vid, mask]], start_at=0)
    # v = Viewer('p', (0,0))
    # v.play(tiled, True, 30, loop=True)

    return mask

def find_stillness_events(mask, file, rank, min_event_len=1, post_event_len=2, threshold=0.1):

    n = mask.shape[0]
    event_window = []
    event_list = []
    num_frames_post_event = 0
    stillness_event_start = 0
    motion_event_start = 0
    in_motion_event = False

    # kernel_size = 1
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Motion event scanning/detection loop.
    i = 0
    motion_score_max = 0
    motion_score_min = 100000000
    motion_event_start = None
    motion_event_end = None

    frame_width = 4
    scores = mask.sum((1, 2))

    # frame_viewer = Viewer('frame', (0,0))
    for i in range(n):
        score = scores[i]
        # if file == 2 and rank == 1 and i > 0:
        #     frame_viewer.update_frame(mask[i], 60)
        #     print(i)

        event_window.append(score)
        event_window = event_window[-min_event_len:]

        if in_motion_event:
            # in event or post event, write all queued frames to file,
            # and write current frame to file.
            # if the current frame doesn't meet the threshold, increment
            # the current scene's post-event counter.
            if score >= threshold:
                num_frames_post_event = 0
                motion_score_min = min(motion_score_min, score)
                motion_score_max = max(motion_score_max, score)
            else:
                num_frames_post_event += 1
                if num_frames_post_event >= post_event_len:
                    in_motion_event = False
                    stillness_event_start = i

        else:
            if len(event_window) >= min_event_len and all(
                    score >= threshold for score in event_window):
                stillness_event_end = i - min_event_len
                if stillness_event_end != stillness_event_start:
                    event_list.append((stillness_event_start, stillness_event_end))
                in_motion_event = True
                event_window = []
                num_frames_post_event = 0

    # If we're still in a motion event, we still need to compute the duration
    # and ending timecode and add it to the event list.
    if not in_motion_event:
        stillness_event_end = i
        event_list.append((stillness_event_start, stillness_event_end))

    return event_list


def print_motion_event(motion_event_start, motion_event_end,
                       motion_score_max, motion_score_min):

    fm_str = '''
Motion Event

Start: frame {}, second {}
End:   frame {}, second {}

Max Score: {}
Min Score: {}
'''
    print(fm_str.format(
        motion_event_start,
        motion_event_start / 30,
        motion_event_end,
        motion_event_end / 30,
        motion_score_max,
        motion_score_min))


def background_subtraction_mask(v):
    open_kernel = np.ones((3, 3), np.uint8)
    dilate_kernel = np.ones((5, 5), np.uint8)

    subtractor = cv2.createBackgroundSubtractorMOG2(10, detectShadows=False)
    return [
        cv2.dilate(
            cv2.morphologyEx(
                subtractor.apply(frame),
                cv2.MORPH_OPEN,
                open_kernel),
            dilate_kernel)
        for frame in v]
