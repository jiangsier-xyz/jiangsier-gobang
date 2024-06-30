#!/usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow.keras import *

from .robot import RobotBrain, RobotPlayer
from ..residual_block import *


class CaseyBrain(RobotBrain):
    def __init__(self, load_model=True):
        super(CaseyBrain, self).__init__(nickname='Casey', load_model=load_model)

    def _build_shared_net(self, inputs):
        # 共享网络
        x = make_conv2d_layer(filters=64, kernel_size=(3, 3))(inputs)
        x = layers.BatchNormalization()(x)
        x = make_basic_block_layer(filter_num=64, blocks=2)(x)
        x = make_basic_block_layer(filter_num=128, blocks=2, stride=2)(x)
        x = make_basic_block_layer(filter_num=256, blocks=2, stride=2)(x)
        x = make_basic_block_layer(filter_num=512, blocks=2, stride=2)(x)
        return x


class Casey(RobotPlayer):
    def __init__(self, name='Casey', c_puct=15, n_playout=400, is_self_play=False, tau=1):
        super(Casey, self).__init__(name, brain=RobotBrain.get(CaseyBrain), c_puct=c_puct, n_playout=n_playout,
                                    is_self_play=is_self_play, tau=tau)
