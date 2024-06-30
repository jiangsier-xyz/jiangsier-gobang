#!/usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow.keras import *

from .robot import RobotBrain, RobotPlayer
from ..residual_block import *


class DarlingBrain(RobotBrain):
    def __init__(self, load_model=True):
        super(DarlingBrain, self).__init__(nickname='Darling', load_model=load_model)

    def _build_shared_net(self, inputs):
        # 共享网络
        x = make_conv2d_layer(filters=64, kernel_size=(3, 3))(inputs)
        x = layers.BatchNormalization()(x)
        x = make_bottle_neck_layer(filter_num=64, blocks=3)(x)
        x = make_bottle_neck_layer(filter_num=128, blocks=4, stride=2)(x)
        x = make_bottle_neck_layer(filter_num=256, blocks=6, stride=2)(x)
        x = make_bottle_neck_layer(filter_num=512, blocks=3, stride=2)(x)
        return x


class Darling(RobotPlayer):
    def __init__(self, name='Darling', c_puct=10, n_playout=300, is_self_play=False, tau=1):
        super(Darling, self).__init__(name, brain=RobotBrain.get(DarlingBrain), c_puct=c_puct, n_playout=n_playout,
                                      is_self_play=is_self_play, tau=tau)
