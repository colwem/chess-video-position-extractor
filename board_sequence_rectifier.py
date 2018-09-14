import numpy as np
import treelib
from chess_helpers import board_array_to_board, board_to_board_array


def narrow_search(ds):
    eql = ds == ds[0, :, :]
    eql = eql.all(0)
    rows = eql.all(1)
    columns = eql.all(0)

    i = 0
    while i < len(rows) and rows[i]:
        i += 1
    i += -1

    j = 0
    while j < len(columns) and columns[j]:
        j += 1
    j += -1

    return i, j


def contract_d(ds, i, j):
    d = ds[0, i:, j:]
    d[1:, 1:] = 0
    return d


def extend_d(t):
    d = np.zeros((t.shape[0], t.shape[1] + 1))
    d[:, :-1] = t
    d[0, -1] = d[0, -2] + 1
    return d


def rectify_board_sequence(board_arrays):

    def eqlf(s, t):
        return (s == t).all()

    def subsf(s, t):
        return diff_table(s, t) / 10

    first_node = create_node(board_array_to_board(board_arrays[0], swap_axes=False))
    d = initialize_d(len(board_arrays) + 1, 2)
    first_node.data['d'],  first_node.data['distance'] = \
        edit_distance(board_arrays,
                      first_node.data['sequence'],
                      d=d.copy(),
                      eqlf=eqlf,
                      subsf=subsf)
    min_node = first_node

    tree = treelib.Tree()
    tree.add_node(first_node)
    i = 0
    j = 0
    while True:
        tree = extend_tree(tree)
        d = extend_d(d)
        print('depth ', tree.depth())
        print('Num leaves ', len(tree.leaves()))
        for leaf in tree.leaves():
            leaf.data['d'], leaf.data['distance'] = \
                edit_distance(board_arrays[i:],
                              leaf.data['sequence'][j:],
                              d=d.copy(),
                              eqlf=eqlf,
                              subsf=subsf)
            min_node = min(min_node, leaf, key=lambda x: x.data['distance'])

        print("min_node")
        print("min_node distance: ", min_node.data["distance"])
        print("min_node depth: ", tree.depth(min_node.identifier))
        print("min_node fens:")
        print("\n".join(
            [ancestor.data["board"].fen().split()[0]
             for ancestor in [min_node] + list(ancestry(tree, min_node))]))
        print(min_node.data['board'])
        print('')
        if tree.depth() - tree.depth(min_node) > 3:
            print("breaking")
            break;

        tree = prune_tree(tree, 8)
        ts = [leaf.data['d'] for leaf in tree.leaves()]
        ds = np.array(ts)
        ii, jj = narrow_search(ds)
        d = contract_d(ds, ii, jj)
        i, j = i + ii, j + jj

    return min_node.data['sequence']


def initialize_d(m, n):
    d = np.zeros(m * n).reshape((m, n))
    for i in range(m):
        d[i, 0] = i

    for j in range(n):
        d[0, j] = j

    return d


def edit_distance(s, t, d=[], subsf=None, eqlf=None):
    if not subsf:
        subsf = lambda x, y: 1

    if not eqlf:
        eqlf = lambda x, y: x == y

    m, n = len(s) + 1, len(t) + 1
    if not len(d):
        d = initialize_d(m, n)

    for j in range(1, n):
        for i in range(1, m):
            if eqlf(s[i-1], t[j-1]):
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min((
                    d[i-1, j] + 1,
                    d[i, j-1] + 1,
                    d[i-1, j-1] + subsf(s[i-1], t[j-1])
                ))

    return d, d[-1, -1]


def backtrace(d):
    x, y = d.shape - (1,1)
    options = ((x-1), #substitution
               (x - 1, y), #insertion
               (1, y - 1)) #deletion
    options = [option for option in options if option[0] >= 0 and option[1] >= 0]
    previous = min(options, key=lambda x: d[x[0], x[1]])

    return backtrace(d[:previous[0], :previous[1]]) + [previous]


def extend_tree(tree):
    for leaf in tree.leaves():
        nodes = generate_legal_move_nodes(leaf)
        nodes += generate_historical_nodes(tree, leaf)
        for node in nodes:
            tree.add_node(node, parent=leaf)

    return tree


def generate_legal_move_nodes(parent):
    board = parent.data['board']
    boards = [new_board_from_move(board, move)
              for move in board.legal_moves]

    return [create_node(board, parent) for board in boards]


def ancestry(tree, node):
    ancestor = tree.parent(node.identifier)
    while ancestor:
        yield ancestor
        ancestor = tree.parent(ancestor.identifier)


def generate_historical_nodes(tree, node):
    ancestors = list((create_node(ancestor.data['board'], parent=node)
                      for ancestor in ancestry(tree, node)
                      if ancestor.data['board'] != node.data['board']))

    return ancestors

def prune_tree(tree, num_leaves=2):
    depth = tree.depth()

    # keep only the lowest scoring N leaves
    for leaf in sorted(tree.leaves(), key=lambda x: x.data['distance'])[num_leaves:]:
        tree.remove_node(leaf.identifier)

    # prune leafless branches
    tree = prune_short_branches(tree)

    return tree


def prune_short_branches(tree):
    depth = tree.depth()
    fixed = True
    for leaf in tree.leaves():
        if tree.depth(leaf.identifier) != depth:
            fixed = False
            tree.remove_node(leaf.identifier)
    if not fixed:
        return prune_short_branches(tree)
    return tree


def diff_table(a, b):
    diff = a - b

    return np.count_nonzero(diff)


def new_board_from_move(board, move):
    b = board.copy()
    b.push(move)

    return b


def create_node(board, parent=None):
    sequence = [board_to_board_array(board)]
    if parent:
        sequence = parent.data['sequence'] + sequence
    # tag = moves_to_string(sequence_to_moves(board))
    # tag = '\n'.join([fen.split()[0]
    #                for fen in [board_array_to_board(board_array, swap_axes=False).fen()
    #                            for board_array in sequence]])
    node = treelib.Node(data={'board': board,
                                       'sequence': sequence})

    return node


def sequence_to_moves(sequence):
    boards = [board_array_to_board(board_array, swap_axes=False)
              for board_array in sequence]
    moves = []
    for board1, board2 in zip(boards[:-1], boards[1:]):
        for move in board1.legal_moves:
            b = board1.copy()
            b.push(move)
            if b == board2:
                moves.append(move)
                break
        for i, board3 in enumerate(boards[1:-1:-1]):
            if board2 == board3:
                moves.append(-i)
                break

    return moves


def moves_to_string(moves):

    return ' '.join([str(move) for move in moves])







