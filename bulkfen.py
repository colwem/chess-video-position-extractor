#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# TensorFlow Chessbot
# This contains ChessboardPredictor, the class responsible for loading and
# running a trained CNN on chessboard screenshots. Used by chessbot.py.
# A CLI interface is provided as well.
#
#   $ ./tensorflow_chessbot.py -h
#   usage: tensorflow_chessbot.py [-h] [--url URL] [--filepath FILEPATH]
#
#    Predict a chessboard FEN from supplied local image link or URL
#
#    optional arguments:
#      -h, --help           show this help message and exit
#      --url URL            URL of image (ex. http://imgur.com/u4zF5Hj.png)
#     --filepath FILEPATH  filepath to image (ex. u4zF5Hj.png)
#
# This file is used by chessbot.py, a Reddit bot that listens on /r/chess for
# posts with an image in it (perhaps checking also for a statement
# "white/black to play" and an image link)
#
# It then takes the image, uses some CV to find a chessboard on it, splits it up
# into a set of images of squares. These are the inputs to the tensorflow CNN
# which will return probability of which piece is on it (or empty)
#
# Dataset will include chessboard squares from chess.com, lichess
# Different styles of each, all the pieces
#
# Generate synthetic data via added noise:
#  * change in coloration
#  * highlighting
#  * occlusion from lines etc.
#
# Take most probable set from TF response, use that to generate a FEN of the
# board, and bot comments on thread with FEN and link to lichess analysis.
#
# A lot of tensorflow code here is heavily adopted from the
# [tensorflow tutorials](https://www.tensorflow.org/versions/0.6.0/tutorials/pdes/index.html)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Ignore Tensorflow INFO debug messages
import tensorflow as tf
import numpy as np
import cv2
import csv

from helper_functions import flatten, split_by_fun, streamify
from chessboard_finder import findChessboardCorners as get_corners
from video_helpers import VideoContainer, Viewer, apply_number,\
    show_together, board_arrays_to_mp4,\
    print_stream, overlay_video, render_svgs
from motion import find_stillness_events, make_mask
from chess_helpers import board_arrays_to_svgs, board_arrays_to_fens
from itertools import groupby, count
from functools import partial


def load_graph(frozen_graph_filepath):
    # Load and parse the protobuf file to retrieve the unserialized graph_def.
    with tf.gfile.GFile(frozen_graph_filepath, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import graph def and return.
    with tf.Graph().as_default() as graph:
        # Prefix every op/nodes in the graph.
        tf.import_graph_def(graph_def, name="tcb")
    return graph


class ChessboardPredictor(object):
    """ChessboardPredictor using saved model"""

    def __init__(self, frozen_graph_path='saved_models/frozen_graph.pb'):
        # Restore model using a frozen graph.
        print("\t Loading model '%s'" % frozen_graph_path)
        graph = load_graph(frozen_graph_path)
        self.sess = tf.Session(graph=graph)

        # Connect input/output pipes to model.
        self.x = graph.get_tensor_by_name('tcb/Input:0')
        self.keep_prob = graph.get_tensor_by_name('tcb/KeepProb:0')
        self.prediction = graph.get_tensor_by_name('tcb/prediction:0')
        self.probabilities = graph.get_tensor_by_name('tcb/probabilities:0')
        print("\t Model restored.")

    def get_predictions(self, tiles):
        """Run trained neural network on tiles generated from image"""
        if tiles is None or len(tiles) == 0:
            print("Couldn't parse chessboard")
            return None, 0.0

        # Reshape into Nx1024 rows of input data, format used by neural network
        N = tiles.shape[2]
        validation_set = np.swapaxes(np.reshape(tiles, [32 * 32, N]), 0, 1)

        # Run neural network on data
        guess_prob, guessed = self.sess.run(
            [self.probabilities, self.prediction],
            feed_dict={self.x: validation_set, self.keep_prob: 1.0})

        # Prediction bounds
        certainty = [x[0][x[1]] for x in zip(guess_prob, guessed)]
        return zip(guessed, certainty)


    def close(self):
        print("Closing session.")
        self.sess.close()


# Piece Formating
# Utility functions
def chunk(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def start_time(str):
    import time
    time = time.time()
    print('{} timer started'.format(str))
    return time


def end_time(str, start_time):
    import time
    now = time.time()
    duration = now - start_time
    print('{} completed in {:.2f}s'.format(str, duration))


### Heavy logic


def video_to_board(vid, corner_group):
    # img is a grayscale image
    # corners = (x0, y0, x1, y1) for top-left corner to bot-right corner of board
    start, end = corner_group[0], corner_group[1]
    length = vid.shape[0]

    corners = corner_group[2]

    height, width = vid.shape[1:3]

    # corners could be outside image bounds, pad image as needed
    padl_x = max(0, -corners[0])
    padl_y = max(0, -corners[1])
    padr_x = max(0, corners[2] - width)
    padr_y = max(0, corners[3] - height)

    # vid_padded = np.pad(vid, ((0, 0), (padl_y, padr_y), (padl_x, padr_x)), mode='edge')

    chessboard_vid = vid[:,
    (padl_y + corners[1]):(padl_y + corners[3]),
    (padl_x + corners[0]):(padl_x + corners[2])]

    # 256x256 px image, 32x32px individual tiles
    # Normalized

    resized = np.empty((length, 256, 256))
    for i in range(start, end + 1):
        resized[i] = cv2.resize(chessboard_vid[i], (256, 256))

    return resized


# shape (N, height, width) -> (N, file, rank, height, width)
# shape (N,    256,   256) -> (N,    8,    8,     32,    32)
def board_to_squares(board):
    N_frames = board.shape[0]
    squares = np.empty((N_frames, 8, 8, 32, 32))
    for frame in range(N_frames):
        for file in range(8):
            for rank in range(8):
                h_start = (32 * file)
                h_end = (32 * (file + 1))
                w_start = (32 * rank)
                w_end = (32 * (rank + 1))
                squares[frame, file, rank] = board[frame, h_start:h_end, w_start:w_end]
    return squares

def get_events_from_corner_group(corner_group, vid):
    board_vid = video_to_board(vid, corner_group)
    mask = make_mask(board_vid)
    print('mask made')
    # squares_vid = board_to_squares(board_vid)

    def get_square(vid, file, rank):
        start_x, end_x = 32 * file, 32 * (file + 1)
        start_y, end_y = 32 * rank, 32 * (rank + 1)
        r = vid[:, start_y:end_y, start_x:end_x]
        return r

    def prep_event(start, end, file, rank):
        mid = (start + end) // 2
        sqr = get_square(board_vid, file, rank)[mid].copy()
        return start, end, file, rank, sqr

    events = [prep_event(start, end, file, rank)
              for rank in range(8)
              for file in range(8)
              for start, end in find_stillness_events(get_square(mask, file, rank), file, rank)]

    # for start, end, file, rank, sqr in events:
    #     display(sqr, name="{},{} {} - {}".format(file, rank, start, end))

    return events


def update_frame_numbers(events, initial_frame_number):
    return [(start + initial_frame_number, end + initial_frame_number, file, rank, img)
            for start, end, file, rank, img in events]


def eqlf(a, b):
    if a is None or b is None:
        if a is None and b is None:
            return True
        return False
    return (a == b).all()


def rec(vid, start_corners, end_corners, start, end, depth, min_depth=0):
    if min_depth > depth or not eqlf(start_corners, end_corners):
        mid = (start + end) // 2

        mid_corners = get_corners(vid[mid])
        r_start = rec(vid, start_corners, mid_corners, start, mid, depth+1)

        mid += 1
        mid_corners = get_corners(vid[mid])
        r_end = rec(vid, mid_corners, end_corners, mid, end, depth+1)

        if eqlf(r_start[-1][2], r_end[0][2]):
            r_start[-1] = (r_start[-1][0], r_end[0][1], r_start[-1][2])
            return r_start + r_end[1:]
        else:
            return r_start + r_end
    else:
        return [(start, end, start_corners)]


def get_corner_groups(vid):
    # find chessboard corners
    corner_groups = rec(vid, get_corners(vid[0]), get_corners(vid[-1]), 0, len(vid) - 1, 0)

    corner_groups = [c for c in corner_groups if c[2] is not None]
    return corner_groups


def get_events_from_vid(vid, initial_frame_number):
    events = [get_events_from_corner_group(corner_group, vid)
              for corner_group in get_corner_groups(vid)
              if corner_group[2] is not None]
    events = flatten(events)
    events = update_frame_numbers(events, initial_frame_number)
    return events


def apply_piece_predictions_to_events(events, predictor, batch_size=1000):
    batches = chunk(events, batch_size)

    def process(batch):
        length = len(batch)
        x = np.empty((32, 32, length))
        for i in range(length):
            x[:, :, i] = batch[i][4]
        x = x / 255
        return [(start, end, file, rank, piece, certainty)
                for ((start, end, file, rank, _), (piece, certainty))
                in zip(batch, predictor.get_predictions(x))]
    return flatten([process(batch) for batch in batches])


def concatenate_identical_predictions(events):

    def compress(e1, e2):
        return (e1[0],
                e2[1],
                *e1[2:])

    def is_duplicate(e1, e2):
        start, end, file, rank, piece_code, certainty = zip(e1, e2)

        both_equal = lambda p: p[0] == p[1]
        return both_equal(file) \
               and both_equal(rank) \
               and both_equal(piece_code) \
               and end[0] <= start[1]

    def dedup(events):
        deduped = []
        i = 0
        events = list(events)
        length = len(events)
        while i < length:
            j = 1
            while i + j < length and is_duplicate(events[i], events[i+j]):
                j += 1
            if j > 1:
                deduped.append(compress(events[i], events[i+j-1]))
            else:
                deduped.append(events[i])
            i += j

        return deduped

    sorted_events = sorted(events, key=lambda e: (e[2], e[3], e[0], e[1]))
    groups = groupby(sorted_events, key=lambda e: (e[2], e[3]))

    return flatten([dedup(group) for key, group in groups])



###########################################################
# MAIN CLI


def insert_break(val):
    return val


def main(args):
    # Load image from filepath or URL
    # Initialize predictor, takes a while, but only needed once
    predictor = ChessboardPredictor()
    # Load image from file
    # open dir, loop through files
    processing_time = start_time("Processing")
    time = start_time("Event creation")
    events = []
    video_container = VideoContainer(args.filepath)
    processed = 0
    start_at = 0
    end_at = video_container.length
    for vid, initial_frame in video_container.get_video_array(start_at=start_at,
                                                              end_at=end_at,
                                                              chunk_size=1000,
                                                              overlap=1):
        events.append(get_events_from_vid(vid, initial_frame))
        processed += len(vid)
        print("batch processed")

    events = flatten(events)
    print("number of events: ", len(events))
    end_time('Event Creation', time)

    time = start_time("Predictions")
    events = apply_piece_predictions_to_events(events, predictor)
    end_time("Predictions", time)

    events = concatenate_identical_predictions(events)
    print("number of events after dedup: ", len(events))
    print("processed: ", processed)
    board_arrays = np.zeros((processed, 8, 8), dtype='uint8') + 20
    for start, end, file, rank, piece_code, certainty in events:
        board_arrays[start - start_at:end - start_at + 1, file, rank] = piece_code

    deduped = [x[0] for x in groupby(board_arrays_to_fens(board_arrays))]

    print(deduped)
    exit()
    time = start_time("Write Video")
    # board_arrays_to_mp4(board_arrays)
    # show_together(args.filepath, board_arrays, 550)

    svgs = board_arrays_to_svgs(board_arrays)
    rendered_svgs = render_svgs(svgs)

    video_stream = VideoContainer(args.filepath).stream(start_at)

    out_stream = overlay_video(video_stream, rendered_svgs, (0, 0))
    out_stream = streamify(partial(apply_number, loc=(20,300)), out_stream, count())

    print_stream('file.avi', out_stream, view_while_writing=True)

    end_time("Write Video", time)

    np.set_printoptions(edgeitems=4)

    end_time('Processing', processing_time)

    predictor.close()
    exit()

    # from itertools import groupby
    # print('l tab bf ', len(table))
    # table = [list(g)[0] for k, g in groupby(table, lambda x: x[0])]
    # print('l tab af ', len(table))

    with open('fens.csv', 'w') as fenCsvFile:
        writer = csv.writer(fenCsvFile, dialect='excel')
        writer.writerows(table)



if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    import argparse

    parser = argparse.ArgumentParser(description='Predict a chessboard FENs from supplied directory of local images')
    parser.add_argument('-f', '--filepath', required=True, help='path to video to process')
    args = parser.parse_args()
    main(args)
