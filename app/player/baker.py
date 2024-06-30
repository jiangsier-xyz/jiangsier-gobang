#!/usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow.keras import *

from .robot import RobotBrain, RobotPlayer
from ..residual_block import *


class BakerBrain(RobotBrain):
    def __init__(self, load_model=True):
        super(BakerBrain, self).__init__(nickname='Baker', load_model=load_model)

    def _build_shared_net(self, inputs):
        # 共享网络
        x = make_conv2d_layer(filters=32, kernel_size=(3, 3), activation=activations.relu)(inputs)
        x = make_conv2d_layer(filters=64, kernel_size=(3, 3), activation=activations.relu)(x)
        x = make_conv2d_layer(filters=128, kernel_size=(3, 3), activation=activations.relu)(x)
        return x


class Baker(RobotPlayer):
    def __init__(self, name='Baker', c_puct=15, n_playout=400, is_self_play=False, tau=1):
        super(Baker, self).__init__(name, brain=RobotBrain.get(BakerBrain), c_puct=c_puct, n_playout=n_playout,
                                    is_self_play=is_self_play, tau=tau)
