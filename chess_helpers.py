import chess
import chess.svg
import numpy as np
from functools import reduce

def board_arrays_to_svgs(board_arrays):
    return [board_array_to_svg(board) for board in board_arrays]


def board_arrays_to_fens(board_arrays):
    return [board_array_to_fen(board_array) for board_array in board_arrays]


def fens_to_board_arrays(fens):
    return [fen_to_board_array(fen) for fen in fens]


def board_array_to_board(board_array, swap_axes=True):
    return fen_to_board(board_array_to_fen(board_array, swap_axes))


def board_to_board_array(board):
    def get_piece_char(piece):
        if piece is None:
            return ' '
        return piece.symbol()

    board_array = [[piece_char_to_piece_code(get_piece_char(board.piece_at(chess.square(j, i))))
        for j in range(8)] for i in range(7, -1, -1)]
    board_array = np.array(board_array)
    return board_array


def board_array_to_fen(board_array, swap_axes=True, key=' KQRBNPkqrbnp'):
    board_array = board_array.copy()
    board_array[board_array > 12] = 0
    if swap_axes:
        board_array = board_array.swapaxes(0,1)
    fen = '/'.join([''.join([key[piece] for piece in row]) for row in board_array])
    return shorten_fen(fen)


def fen_to_board_array(fen):
    return board_to_board_array(fen_to_board(fen))


def fen_to_board(fen):
    board = chess.Board()
    board.set_board_fen(fen)
    return board


def shorten_fen(fen):
    replace_char = ' '
    return reduce(lambda s, x: s.replace(*x), [(replace_char * i, str(i)) for i in reversed(range(1,9))], fen)

def piece_list_to_fen(lst):
    if len(lst) != 64:
        raise Exception("length of piece list is not 64")
    return '/'.join([''.join(lst[i * 8:(i + 1) * 8]) for i in reversed(range(8))])

def piece_char_to_piece_code(piece_char, key=' KQRBNPkqrbnp'):
    return key.index(piece_char)


def board_array_to_svg(board_array):
    board = board_array_to_board(board_array)
    return chess.svg.board(board)
