#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from enum import IntEnum

import numpy as np

from .config import COLS, ROWS, STATE_DEPTH, WIN_NUMBER, HISTORY_DIR


class Piece(IntEnum):
    NONE = -1
    WHITE = 0
    BLACK = 1

    @staticmethod
    def next(piece):
        return Piece.WHITE if piece == Piece.BLACK else Piece.BLACK


# Action 不能使用 namedtuple 定义，因为 namedtuple 有维度，会影响 shape 推断
class Action:
    # 游戏过程中，Action会被频繁构造。缓存所有可能的Action以提升性能。
    ACTIONS = None
    ALL = []

    @staticmethod
    def init():
        if Action.ACTIONS is not None:
            return

        Action.ACTIONS = [[INVALID_ACTION for _ in range(COLS)] for _ in range(ROWS)]
        for i in range(COLS):
            for j in range(ROWS):
                Action.ACTIONS[j][i] = Action(i, j)
                Action.ALL.append(Action.ACTIONS[j][i])

    @staticmethod
    def get(x, y):
        if x < 0 or y < 0:
            return INVALID_ACTION
        else:
            return Action.ACTIONS[y][x]

    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __eq__(self, other):
        return self._x == other.x and self._y == other.y

    def __hash__(self):
        return hash((self._x, self._y))


INVALID_ACTION = Action(-1, -1)
Action.init()


class Player:
    def __init__(self, name):
        self._name = name
        self._piece = Piece.NONE
        self._action = INVALID_ACTION
        pass

    @property
    def nickname(self):
        return self._name

    @nickname.setter
    def nickname(self, name):
        self._name = name

    @property
    def piece(self):
        return self._piece

    @piece.setter
    def piece(self, piece):
        self._piece = piece

    def play(self, env):
        return self._action

    def act(self, action):
        self._action = action


class Direction(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    LEFT_SLASH = 2
    RIGHT_SLASH = 3


class GoBang:
    def __init__(self, record=False):
        self._board = np.ones((ROWS, COLS), dtype=int) * Piece.NONE
        self._winner = Piece.NONE
        self._last_piece = Piece.WHITE
        self._last_action = INVALID_ACTION
        if record:
            self._history = []
        else:
            self._history = None

    def reset(self):
        self._board[:] = Piece.NONE
        self._winner = Piece.NONE
        self._last_piece = Piece.WHITE
        self._last_action = INVALID_ACTION
        if self._history is not None:
            self._history.clear()

    def copy_to(self, other):
        other.board[:] = self._board
        other.winner = self._winner
        other.last_piece = self._last_piece
        other.last_action = self.last_action
        if self._history is not None and other.history is not None:
            other.history = self._history[:]

    def is_valid(self, action, allow_isolation=True):
        if action == INVALID_ACTION or self._board[action.y][action.x] != Piece.NONE:
            return False

        if not allow_isolation and self.is_isolated(action):
            return False

        piece = Piece.next(self.last_piece)
        if piece == Piece.BLACK:
            # https://baike.baidu.com/item/%E7%A6%81%E6%89%8B/214940?fr=aladdin
            # 三三禁手
            line_3_3 = 0
            # 四四禁手
            line_4_4 = 0
            # 长连禁手
            line_6 = 0
            for direction in Direction:
                line_count = self.line_count(action, piece, direction, recursive=True)
                if line_count[0] == WIN_NUMBER:
                    return True
                elif line_count[0] == WIN_NUMBER - 2 and line_count[1] and line_count[2]:
                    line_3_3 += 1
                elif line_count[0] == WIN_NUMBER - 1 and (line_count[1] or line_count[2]):
                    line_4_4 += 1
                elif line_count[0] > WIN_NUMBER:
                    line_6 += 1

            return line_3_3 < 2 and line_4_4 < 2 and line_6 == 0

        return True

    def available_actions(self):
        return [action for row in Action.ACTIONS for action in row if self.is_valid(action, False)]

    def further_count(self, piece, nearby_pos_x, nearby_pos_y, direction):
        possible_action = Action.get(nearby_pos_x, nearby_pos_y)
        possible_count, possible_valid_pos_1, possible_valid_pos_2, _ = \
            self.line_count(possible_action, piece, direction, recursive=False)
        if possible_count >= WIN_NUMBER:
            # 填一个空即连n，相当于冲4
            return WIN_NUMBER - 1
        elif possible_count == WIN_NUMBER - 1 and possible_valid_pos_1 and possible_valid_pos_2:
            # 填一个空即活4，相当于活3
            return WIN_NUMBER - 2
        else:
            # 其他情况，保持不变
            return 0

    def line_count(self, action, piece, direction, recursive=True):
        x = action.x
        y = action.y
        nearby_pos_1_x, nearby_pos_1_y, nearby_pos_2_x, nearby_pos_2_y = -1, -1, -1, -1
        count = 1
        be_recursive = False

        if direction == Direction.HORIZONTAL:
            for i in range(x - 1, -1, -1):
                if self._board[y][i] != piece:
                    nearby_pos_1_x, nearby_pos_1_y = i, y
                    break
                count += 1
                nearby_pos_1_x, nearby_pos_1_y = i - 1, y

            for i in range(x + 1, COLS):
                if self._board[y][i] != piece:
                    nearby_pos_2_x, nearby_pos_2_y = i, y
                    break
                count += 1
                nearby_pos_2_x, nearby_pos_2_y = i + 1, y
        elif direction == Direction.VERTICAL:
            for j in range(y - 1, -1, -1):
                if self._board[j][x] != piece:
                    nearby_pos_1_x, nearby_pos_1_y = x, j
                    break
                count += 1
                nearby_pos_1_x, nearby_pos_1_y = x, j - 1

            for j in range(y + 1, ROWS):
                if self._board[j][x] != piece:
                    nearby_pos_2_x, nearby_pos_2_y = x, j
                    break
                count += 1
                nearby_pos_2_x, nearby_pos_2_y = x, j + 1
        elif direction == Direction.LEFT_SLASH:
            j = y - 1
            for i in range(x + 1, COLS):
                if j < 0 or self._board[j][i] != piece:
                    nearby_pos_1_x, nearby_pos_1_y = i, j
                    break
                count += 1
                j -= 1
                nearby_pos_1_x, nearby_pos_1_y = i + 1, j

            j = y + 1
            for i in range(x - 1, -1, -1):
                if j >= ROWS or self._board[j][i] != piece:
                    nearby_pos_2_x, nearby_pos_2_y = i, j
                    break
                count += 1
                j += 1
                nearby_pos_2_x, nearby_pos_2_y = i - 1, j
        elif direction == Direction.RIGHT_SLASH:
            j = y + 1
            for i in range(x + 1, COLS):
                if j >= ROWS or self._board[j][i] != piece:
                    nearby_pos_1_x, nearby_pos_1_y = i, j
                    break
                count += 1
                j += 1
                nearby_pos_1_x, nearby_pos_1_y = i + 1, j

            j = y - 1
            for i in range(x - 1, -1, -1):
                if j < 0 or self._board[j][i] != piece:
                    nearby_pos_2_x, nearby_pos_2_y = i, j
                    break
                count += 1
                j -= 1
                nearby_pos_2_x, nearby_pos_2_y = i - 1, j

        valid_pos_1 = self._board[nearby_pos_1_y][nearby_pos_1_x] == Piece.NONE \
            if nearby_pos_1_x in range(COLS) and nearby_pos_1_y in range(ROWS) \
            else False
        valid_pos_2 = self._board[nearby_pos_2_y][nearby_pos_2_x] == Piece.NONE \
            if nearby_pos_2_x in range(COLS) and nearby_pos_2_y in range(ROWS) \
            else False

        if not recursive or count >= WIN_NUMBER - 1:
            return count, valid_pos_1, valid_pos_2, be_recursive

        # 小于连4的情况下，仍然有可能是冲4或者活3
        # 冲4：--o*-oo--
        # 活3：--o-*o---
        # 冲4条件：中间空格，连上即成连n（n>=5）
        # 活3条件：中间空格，连上即成活4
        # 注：只有可行位置的下一位置是同色棋子，才值得探索
        if direction == Direction.HORIZONTAL:
            further_pos_1_x, further_pos_1_y, further_pos_2_x, further_pos_2_y = \
                nearby_pos_1_x - 1, nearby_pos_1_y, nearby_pos_2_x + 1, nearby_pos_2_y
        elif direction == Direction.VERTICAL:
            further_pos_1_x, further_pos_1_y, further_pos_2_x, further_pos_2_y = \
                nearby_pos_1_x, nearby_pos_1_y - 1, nearby_pos_2_x, nearby_pos_2_y + 1
        elif direction == Direction.LEFT_SLASH:
            further_pos_1_x, further_pos_1_y, further_pos_2_x, further_pos_2_y = \
                nearby_pos_1_x + 1, nearby_pos_1_y - 1, nearby_pos_2_x - 1, nearby_pos_2_y + 1
        elif direction == Direction.RIGHT_SLASH:
            further_pos_1_x, further_pos_1_y, further_pos_2_x, further_pos_2_y = \
                nearby_pos_1_x + 1, nearby_pos_1_y + 1, nearby_pos_2_x - 1, nearby_pos_2_y - 1
        else:
            further_pos_1_x, further_pos_1_y, further_pos_2_x, further_pos_2_y = \
                -1, -1, -1, -1

        further_pos_1 = self._board[further_pos_1_y][further_pos_1_x] == piece \
            if further_pos_1_x in range(COLS) and further_pos_1_y in range(ROWS) \
            else False
        further_pos_2 = self._board[further_pos_2_y][further_pos_2_x] == piece \
            if further_pos_2_x in range(COLS) and further_pos_2_y in range(ROWS) \
            else False

        origin_piece = self._board[y][x]
        self._board[y][x] = piece
        max_count = count
        if valid_pos_1 and further_pos_1:
            max_count = max(max_count, self.further_count(piece, nearby_pos_1_x, nearby_pos_1_y, direction))

        if valid_pos_2 and further_pos_2:
            max_count = max(max_count, self.further_count(piece, nearby_pos_2_x, nearby_pos_2_y, direction))

        if max_count > count:
            count = max_count
            be_recursive = True
            if max_count == WIN_NUMBER - 1:
                # 冲4
                valid_pos_1, valid_pos_2 = (False, True)
            elif max_count == WIN_NUMBER - 2:
                # 活3
                valid_pos_1, valid_pos_2 = (True, True)

        self._board[y][x] = origin_piece
        return count, valid_pos_1, valid_pos_2, be_recursive

    def is_successful(self, action, piece):
        for direction in Direction:
            count = self.line_count(action, piece, direction, recursive=False)[0]
            # 只有白方允许大于5的长连
            if count == WIN_NUMBER or (piece == Piece.WHITE and count > WIN_NUMBER):
                return True

        return False

    def likely_to_be_successful(self, action, piece):
        for direction in Direction:
            count, valid_pos_1, valid_pos_2, _ = self.line_count(action, piece, direction, recursive=False)
            if count == WIN_NUMBER - 1 and valid_pos_1 and valid_pos_2:
                return True

        return False

    def turn(self, player, print_board=False):
        piece = player.piece
        if self._last_piece == piece or self.game_over():
            return INVALID_ACTION

        action = player.play(self)
        if action == INVALID_ACTION:
            self._winner = self._last_piece
            return INVALID_ACTION

        self._last_piece = piece
        self._board[action.y][action.x] = piece
        self._last_action = action
        if self._history is not None:
            self._history.append((action.x, action.y, piece))
        success = self.is_successful(action, piece)

        if success:
            self._winner = piece

        if print_board:
            self.print()

        return action

    def game_over(self):
        if self._winner != Piece.NONE:
            return True

        return not self._board.__contains__(Piece.NONE)

    def is_isolated(self, action):
        for j in range(action.y - 2, action.y + 3):
            if -1 < j < ROWS:
                for i in range(action.x - 2, action.x + 3):
                    if j == action.y and i == action.x:
                        continue
                    if -1 < i < COLS and self._board[j][i] != Piece.NONE:
                        return False
        return True

    @staticmethod
    def board_to_state(board, piece):
        # layers[0]: 当前行棋棋子占据的位置
        # layers[1]: 对手行棋棋子占据的位置
        # layers[2]: 当前行棋是黑子，全为1；当前行棋是白子，全为0
        layers = np.zeros((STATE_DEPTH, ROWS, COLS), dtype=float)
        layers[0][board == piece] = 1.0
        layers[1][board == Piece.next(piece)] = 1.0
        layers[2] = piece * 1.0

        # 返回NCHW格式
        return layers.transpose((1, 2, 0))

    @staticmethod
    def state_shape():
        return ROWS, COLS, STATE_DEPTH

    def to_state(self, piece):
        return GoBang.board_to_state(self._board, piece)

    @property
    def winner(self):
        return self._winner

    @winner.setter
    def winner(self, winner):
        self._winner = winner

    @property
    def board(self):
        return self._board

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, history):
        self._history = history

    @property
    def last_piece(self):
        return self._last_piece

    @last_piece.setter
    def last_piece(self, piece):
        self._last_piece = piece

    @property
    def last_action(self):
        return self._last_action

    @last_action.setter
    def last_action(self, last_action):
        self._last_action = last_action

    @staticmethod
    def print_board(board, last_action=None):
        print('=' * (COLS * 2 + 1))
        for j in range(ROWS):
            for i in range(COLS):
                piece = board[j][i]
                if piece == Piece.BLACK:
                    flag = 'x' if last_action is None or i != last_action.x or j != last_action.y else 'X'
                elif piece == Piece.WHITE:
                    flag = 'o' if last_action is None or i != last_action.x or j != last_action.y else 'O'
                else:
                    flag = '-'
                print(' ' + flag, end='')
            print('')
        print('')

    def print(self):
        self.print_board(self._board, self._last_action)

    def print_history(self):
        if self._history is None:
            return

        board = [[Piece.NONE] * COLS for _ in range(ROWS)]
        for x, y, p in self._history:
            board[y][x] = p
            self.print_board(board, Action.get(x, y))
        if self._winner == Piece.NONE:
            print('TIE!')
        else:
            print('WINNER is {} !'.format('black' if self._winner == Piece.BLACK else 'white'))

    def save_history(self, name):
        if self._history is None:
            return

        if not os.path.exists(HISTORY_DIR):
            os.makedirs(HISTORY_DIR)

        his = HISTORY_DIR + name + '.his'
        if os.path.exists(his):
            os.remove(his)
        with open(his, 'wb') as npy:
            np.save(npy, np.array(self._history, np.int32))

    def load_history(self, name):
        his = HISTORY_DIR + name + '.his'
        if os.path.exists(his):
            self._history = np.load(his).tolist()


def test_line():
    game = GoBang()
    piece = Piece.WHITE

    # ---*-o---（活1）
    game.board[0][5] = Piece.WHITE
    action = Action.get(3, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 1 and valid_pos_1 and valid_pos_2

    game.board[:] = Piece.NONE
    game.board[5][0] = Piece.WHITE
    action = Action.get(0, 3)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.VERTICAL)
    assert count == 1 and valid_pos_1 and valid_pos_2

    game.board[:] = Piece.NONE
    game.board[5][5] = Piece.WHITE
    action = Action.get(2, 8)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.LEFT_SLASH)
    assert count == 1 and valid_pos_1 and valid_pos_2

    game.board[:] = Piece.NONE
    game.board[5][5] = Piece.WHITE
    action = Action.get(8, 8)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.RIGHT_SLASH)
    assert count == 1 and valid_pos_1 and valid_pos_2

    # ---o*-ox-（活2）
    game.board[:] = Piece.NONE
    game.board[0][3] = Piece.WHITE
    game.board[0][6] = Piece.WHITE
    game.board[0][7] = Piece.BLACK
    action = Action.get(4, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 2 and valid_pos_1 and valid_pos_2

    game.board[:] = Piece.NONE
    game.board[3][0] = Piece.WHITE
    game.board[6][0] = Piece.WHITE
    game.board[7][0] = Piece.BLACK
    action = Action.get(0, 4)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.VERTICAL)
    assert count == 2 and valid_pos_1 and valid_pos_2

    game.board[:] = Piece.NONE
    game.board[7][7] = Piece.WHITE
    game.board[4][10] = Piece.WHITE
    game.board[3][11] = Piece.BLACK
    action = Action.get(8, 6)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.LEFT_SLASH)
    assert count == 2 and valid_pos_1 and valid_pos_2

    game.board[:] = Piece.NONE
    game.board[7][7] = Piece.WHITE
    game.board[10][10] = Piece.WHITE
    game.board[11][11] = Piece.BLACK
    action = Action.get(8, 8)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.RIGHT_SLASH)
    assert count == 2 and valid_pos_1 and valid_pos_2

    # ---o*o---（活3）
    game.board[:] = Piece.NONE
    game.board[0][3] = Piece.WHITE
    game.board[0][5] = Piece.WHITE
    action = Action.get(4, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 3 and valid_pos_1 and valid_pos_2

    game.board[:] = Piece.NONE
    game.board[3][0] = Piece.WHITE
    game.board[5][0] = Piece.WHITE
    action = Action.get(0, 4)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.VERTICAL)
    assert count == 3 and valid_pos_1 and valid_pos_2

    game.board[:] = Piece.NONE
    game.board[7][7] = Piece.WHITE
    game.board[5][9] = Piece.WHITE
    action = Action.get(8, 6)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.LEFT_SLASH)
    assert count == 3 and valid_pos_1 and valid_pos_2

    game.board[:] = Piece.NONE
    game.board[7][7] = Piece.WHITE
    game.board[9][9] = Piece.WHITE
    action = Action.get(8, 8)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.RIGHT_SLASH)
    assert count == 3 and valid_pos_1 and valid_pos_2

    # --xo*o---（单3）
    game.board[:] = Piece.NONE
    game.board[0][2] = Piece.BLACK
    game.board[0][3] = Piece.WHITE
    game.board[0][5] = Piece.WHITE
    action = Action.get(4, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 3 and not valid_pos_1 and valid_pos_2

    # ---o*o-x-（活3）
    game.board[:] = Piece.NONE
    game.board[0][3] = Piece.WHITE
    game.board[0][5] = Piece.WHITE
    game.board[0][7] = Piece.BLACK
    action = Action.get(4, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 3 and valid_pos_1 and valid_pos_2

    # ---o*-o--（活3）
    game.board[:] = Piece.NONE
    game.board[0][3] = Piece.WHITE
    game.board[0][6] = Piece.WHITE
    action = Action.get(4, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 3 and valid_pos_1 and valid_pos_2

    # ---o*-ox-（活2）
    game.board[:] = Piece.NONE
    game.board[0][3] = Piece.WHITE
    game.board[0][6] = Piece.WHITE
    game.board[0][7] = Piece.BLACK
    action = Action.get(4, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 2 and valid_pos_1 and valid_pos_2

    # o*-o-----（单2）
    game.board[:] = Piece.NONE
    game.board[0][0] = Piece.WHITE
    game.board[0][3] = Piece.WHITE
    action = Action.get(1, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 2 and not valid_pos_1 and valid_pos_2

    # o*o------（单3）
    game.board[:] = Piece.NONE
    game.board[0][0] = Piece.WHITE
    game.board[0][2] = Piece.WHITE
    action = Action.get(1, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 3 and not valid_pos_1 and valid_pos_2

    # o*o-o----（冲4）
    game.board[:] = Piece.NONE
    game.board[0][0] = Piece.WHITE
    game.board[0][2] = Piece.WHITE
    game.board[0][4] = Piece.WHITE
    action = Action.get(1, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 4 and not valid_pos_1 and valid_pos_2

    # --o*o-o--（冲4）
    game.board[:] = Piece.NONE
    game.board[0][2] = Piece.WHITE
    game.board[0][4] = Piece.WHITE
    game.board[0][6] = Piece.WHITE
    action = Action.get(3, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 4 and not valid_pos_1 and valid_pos_2

    # -xo*o-o--（冲4）
    game.board[:] = Piece.NONE
    game.board[0][1] = Piece.BLACK
    game.board[0][2] = Piece.WHITE
    game.board[0][4] = Piece.WHITE
    game.board[0][6] = Piece.WHITE
    action = Action.get(3, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 4 and not valid_pos_1 and valid_pos_2

    # --o*o-ox-（冲4）
    game.board[:] = Piece.NONE
    game.board[0][2] = Piece.WHITE
    game.board[0][4] = Piece.WHITE
    game.board[0][6] = Piece.WHITE
    game.board[0][7] = Piece.BLACK
    action = Action.get(3, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 4 and not valid_pos_1 and valid_pos_2

    # -xo*o-ox-（冲4）
    game.board[:] = Piece.NONE
    game.board[0][1] = Piece.BLACK
    game.board[0][2] = Piece.WHITE
    game.board[0][4] = Piece.WHITE
    game.board[0][6] = Piece.WHITE
    game.board[0][7] = Piece.BLACK
    action = Action.get(3, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 4 and not valid_pos_1 and valid_pos_2

    # -o-*o-ooo（冲4）
    game.board[:] = Piece.NONE
    game.board[0][1] = Piece.WHITE
    game.board[0][4] = Piece.WHITE
    game.board[0][6] = Piece.WHITE
    game.board[0][7] = Piece.WHITE
    game.board[0][8] = Piece.WHITE
    action = Action.get(3, 0)
    count, valid_pos_1, valid_pos_2, _ = game.line_count(action, piece, Direction.HORIZONTAL)
    assert count == 4 and not valid_pos_1 and valid_pos_2


def test_is_valid():
    game = GoBang()
    game.last_piece = Piece.WHITE

    # 三三禁手
    game.board[1][1] = Piece.BLACK
    game.board[1][3] = Piece.BLACK
    game.board[2][2] = Piece.BLACK
    game.board[3][2] = Piece.BLACK
    action = Action.get(2, 2)
    assert not game.is_valid(action)

    # 四四禁手
    game.board[1][1] = Piece.BLACK
    game.board[1][3] = Piece.BLACK
    game.board[1][4] = Piece.BLACK
    game.board[1][5] = Piece.WHITE
    game.board[2][2] = Piece.BLACK
    game.board[3][2] = Piece.BLACK
    game.board[5][2] = Piece.BLACK
    action = Action.get(2, 2)
    assert not game.is_valid(action)

    # 长连禁手
    game.board[1][1] = Piece.BLACK
    game.board[1][3] = Piece.BLACK
    game.board[1][4] = Piece.BLACK
    game.board[1][5] = Piece.BLACK
    game.board[1][6] = Piece.BLACK
    action = Action.get(2, 2)
    assert not game.is_valid(action)


def test_is_successful():
    game = GoBang()
    game.board[1][2] = Piece.WHITE
    game.board[2][2] = Piece.WHITE
    game.board[3][2] = Piece.WHITE
    game.board[4][2] = Piece.WHITE
    action = Action.get(2, 5)
    assert game.is_successful(action, Piece.WHITE)
