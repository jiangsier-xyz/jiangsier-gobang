#!/usr/bin/python
# -*- coding: utf-8 -*-

from operator import itemgetter

from ..gobang import Player, INVALID_ACTION
from ..mcts import MCTS, default_policy_fn
from ..math_util import max_sample


class Monkey(Player):
    def __init__(self, name='Monkey', c_puct=15, n_playout=600, tau=1):
        super(Monkey, self).__init__(name)
        self._mcts = MCTS(policy_value_fn=None, c_puct=c_puct, n_playout=n_playout, name='mcts-' + name)
        self._tau = tau

    def play(self, env):
        if self._mcts.n_playout == 0:
            action_priors = default_policy_fn(env, self.piece)
        else:
            actions, priors, _ = self._mcts.get_action_priors(env, self._tau)
            action_priors = zip(actions, priors)
        action = max_sample(action_priors, key=itemgetter(1))[0] if len(action_priors) > 0 else INVALID_ACTION
        self._mcts.update(INVALID_ACTION)
        return action

    def reset(self, c_puct=15, n_playout=320):
        self._mcts.c_puct = c_puct
        self._mcts.n_playout = n_playout
        self._mcts.update(INVALID_ACTION)

    @property
    def mcts(self):
        return self._mcts
