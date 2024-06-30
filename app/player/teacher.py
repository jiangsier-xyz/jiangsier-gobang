#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict

from ..config import COLS, ROWS, WIN_NUMBER
from ..gobang import Piece, Direction
from ..math_util import prob
from .robot import RobotPlayer, RobotBrain

line_weight = defaultdict(int)
for i in range(WIN_NUMBER):
    line_weight[i] = 10 ** i


class TeacherBrain(RobotBrain):
    def __init__(self, load_model=False):
        super(TeacherBrain, self).__init__(nickname='Teacher', load_model=False)
        pass

    def _build_net(self, model_name='Robot-Model'):
        pass

    def _build_shared_net(self, inputs):
        pass

    def policy_value(self, states):
        pass

    def train(self, state_batch, action_priors, winner_batch, learning_rate=None):
        pass

    def save(self, model_file=None):
        pass

    def load(self, model_file=None):
        pass


class Teacher(RobotPlayer):
    def __init__(self, name='Teacher', c_puct=15, n_playout=600, is_self_play=False, tau=1):
        super(Teacher, self).__init__(name, brain=RobotBrain.get(TeacherBrain), c_puct=c_puct, n_playout=n_playout,
                                      is_self_play=is_self_play, tau=tau)

    @staticmethod
    def _compute_weight(count, valid_pos_1, valid_pos_2):
        if not (valid_pos_1 or valid_pos_2):
            return 0

        factor = count if (valid_pos_1 and valid_pos_2) else (count - 1)
        factor = min(WIN_NUMBER, factor)

        # 调用频繁，查表比计算指数效率更高
        # return 10 ** factor
        return line_weight[factor]

    def policy_value(self, env, piece):
        center_x = COLS // 2
        center_y = ROWS // 2

        actions = env.available_actions()
        weights = np.zeros_like(env.board, dtype=float)

        action_priors = []
        forced_choice = None

        for action in actions:
            weight = 0. + center_x - abs(action.x - center_x) + center_y - abs(action.y - center_y)

            for p in (Piece.BLACK, Piece.WHITE):
                self_play = p == piece
                for direction in Direction.__members__.values():
                    count, valid_pos_1, valid_pos_2, _ = env.line_count(action, p, direction, recursive=False)
                    # 只有白方允许大于5的长连
                    if count == WIN_NUMBER or (p == Piece.WHITE and count > WIN_NUMBER):
                        if self_play:
                            return [(action, 1.0)], 1.0
                        else:
                            forced_choice = [(action, 1.0)], 0.
                    weight += (self._compute_weight(count, valid_pos_1, valid_pos_2) * (1.0 if self_play else 0.75))

            weights[action.y][action.x] = weight

        if forced_choice is not None:
            return forced_choice

        weights = prob(weights)

        for action in actions:
            if weights[action.y][action.x] > 1e-10:
                action_priors.append((action, weights[action.y][action.x]))

        return action_priors, 0.
