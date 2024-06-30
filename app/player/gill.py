#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
from operator import itemgetter
from ..gobang import Player
from .ace import Ace
from .baker import Baker
from .casey import Casey
from .darling import Darling
# from .ellis import Ellis
from .fox import Fox


class Gill(Player):
    def __init__(self, name='Gill', c_puct=15, n_playout=400):
        super(Gill, self).__init__(name)
        child_playout = n_playout // 6
        players = dict()
        players['Ace'] = (Ace(c_puct=c_puct, n_playout=child_playout), 0.107)
        players['Baker'] = (Baker(c_puct=c_puct, n_playout=child_playout), 0.330)
        players['Casey'] = (Casey(c_puct=c_puct, n_playout=child_playout), 0.271)
        players['Darling'] = (Darling(c_puct=c_puct, n_playout=child_playout), 0.197)
        # players['Ellis'] = (Ellis(c_puct=c_puct, n_playout=child_playout), 0.142)
        players['Fox'] = (Fox(c_puct=c_puct, n_playout=child_playout), 0.103)
        self._players = players

    def play(self, env):
        actions = defaultdict(int)
        for player, weight in self._players.values():
            player.piece = self.piece
            action = player.play(env)
            actions[action] += weight

        return max(actions.items(), key=itemgetter(1))[0]

    def update(self, action):
        for player, _ in self._players.values():
            player.mcts.update(action)

    @property
    def mcts(self):
        return self
