#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import os

from ..config import ROWS, COLS, SGF_DIR
from ..gobang import Action, Piece, GoBang
from . import sgf

RE_WHITE_PREFIX = "W白"
RE_BLACK_PREFIX = "B黑"

LOC = "abcdefghijklmnopqrs-"


def validate_action(sgf_action):
    return len(sgf_action) == 2 and sgf_action[0] in LOC and sgf_action[1] in LOC


def sgf_string_to_gobang(sgf_string):
    collection = sgf.parse(sgf_string)
    assert len(collection) > 0

    main_tree = collection[0]
    assert len(main_tree.nodes) > 0

    summary = main_tree.nodes[0].properties
    assert summary['SZ'][0] == str(min(ROWS, COLS))

    winner = Piece.NONE
    if 'RE' in summary:
        if summary['RE'][0][0] in RE_WHITE_PREFIX:
            winner = Piece.WHITE
        elif summary['RE'][0][0] in RE_BLACK_PREFIX:
            winner = Piece.BLACK

    history = []
    last_piece = Piece.WHITE
    last_x = -1
    last_y = -1
    if main_tree.rest is not None:
        for node in main_tree.rest:
            if 'W' in node.properties:
                if not validate_action(node.properties['W'][0]):
                    return None
                piece = Piece.WHITE
                x = LOC.index(node.properties['W'][0][0])
                y = LOC.index(node.properties['W'][0][1])
            elif 'B' in node.properties:
                if not validate_action(node.properties['B'][0]):
                    return None
                piece = Piece.BLACK
                x = LOC.index(node.properties['B'][0][0])
                y = LOC.index(node.properties['B'][0][1])
            else:
                continue
            history.append((x, y, piece))
            last_piece = piece
            last_x = x
            last_y = y

    game = GoBang()
    game.winner = winner
    game.last_piece = last_piece
    game.last_action = Action.get(last_x, last_y)
    game.history = history
    for x, y, p in history:
        game.board[y][x] = p

    return game


def sgf_file_to_gobang(sgf_path):
    with open(sgf_path, 'r', encoding='utf-8') as f:
        sgf_string = f.read()

    return sgf_string_to_gobang(sgf_string)


def action_to_sgf_string(piece, action):
    if piece == Piece.BLACK:
        p_str = 'B'
    else:
        p_str = 'W'

    x_str = LOC[action.x]
    y_str = LOC[action.y]

    return f'{p_str}[{x_str}{y_str}]'


def sgf_string_to_action(sgf_string):
    if len(sgf_string) != 5:
        return None

    piece = Piece.NONE
    if 'B' == sgf_string[0]:
        piece = Piece.BLACK
    elif 'W' == sgf_string[0]:
        piece = Piece.WHITE

    x = LOC.index(sgf_string[2]) if sgf_string[2] != '-' else -1
    y = LOC.index(sgf_string[3]) if sgf_string[3] != '-' else -1

    return piece, Action.get(x, y)


def main():
    sgf_files = glob.glob(SGF_DIR + '*.sgf')
    bad_files = []
    for path in sgf_files:
        game = sgf_file_to_gobang(path)
        if game is None:
            bad_files.append(os.path.basename(path))
            continue
        # game.print_history()

    print('BAD FILES: ' + str(bad_files))


if __name__ == '__main__':
    main()
