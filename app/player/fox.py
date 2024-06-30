#!/usr/bin/python
# -*- coding: utf-8 -*-

from .robot import RobotBrain, RobotPlayer
from ..residual_block import make_fc_block_layer


class FoxBrain(RobotBrain):
    def __init__(self, load_model=True):
        super(FoxBrain, self).__init__(nickname='Fox', load_model=load_model)

    def _build_shared_net(self, inputs):
        # 共享网络
        return make_fc_block_layer(blocks=8)(inputs)


class Fox(RobotPlayer):
    def __init__(self, name='Fox', c_puct=15, n_playout=400, is_self_play=False, tau=1):
        super(Fox, self).__init__(name, brain=RobotBrain.get(FoxBrain), c_puct=c_puct, n_playout=n_playout,
                                  is_self_play=is_self_play, tau=tau)
