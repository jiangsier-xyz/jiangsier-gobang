#!/usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow.keras import *

from ..config import STATE_DEPTH, ROWS, COLS
from .robot import RobotBrain, RobotPlayer
from ..residual_block import *


class AceBrain(RobotBrain):
    def __init__(self, load_model=True):
        super(AceBrain, self).__init__(nickname='Ace', load_model=load_model)

    def _build_shared_net(self, inputs):
        # 共享网络
        return inputs

    @staticmethod
    def _build_policy_net(x):
        # 策略网络
        p = layers.Dense(STATE_DEPTH)(x)
        p = layers.BatchNormalization()(p)
        p = layers.ReLU()(p)
        p = layers.Flatten()(p)
        p = layers.Dense(ROWS * COLS, activation=activations.relu, kernel_regularizer=regularizers.l2())(p)
        p = layers.Softmax()(p)
        return layers.Reshape((ROWS, COLS), name='policy_output')(p)

    @staticmethod
    def _build_value_net(x):
        # 价值网络
        v = layers.Dense(STATE_DEPTH)(x)
        v = layers.BatchNormalization()(v)
        v = layers.ReLU()(v)
        v = layers.Dense(STATE_DEPTH, activation=activations.relu, kernel_regularizer=regularizers.l2())(v)
        v = layers.Flatten()(v)
        return layers.Dense(1, activation=activations.tanh, name='value_output')(v)


class Ace(RobotPlayer):
    def __init__(self, name='Ace', c_puct=15, n_playout=400, is_self_play=False, tau=1):
        super(Ace, self).__init__(name, brain=RobotBrain.get(AceBrain), c_puct=c_puct, n_playout=n_playout,
                                  is_self_play=is_self_play, tau=tau)
