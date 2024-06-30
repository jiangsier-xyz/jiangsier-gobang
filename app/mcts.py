#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy
import datetime
import os
import random
import sys
from collections import defaultdict
from operator import itemgetter

import numpy as np

from .config import COLS, ROWS, DEBUG, LOG_DIR
from .gobang import INVALID_ACTION, Action, Piece, Player, GoBang
from .math_util import max_sample, prob


class TreeNode:
    """
    蒙特卡洛树的树结点的类,
    每个结点用自己的Q值，
    先验概率P，
    访问计数值(visit-count-adjusted prior score):U
    U值的计算公式：U(s,a)=c_{puct}P(s,a)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}
    详情可以参考Alphazero的论文，或者：
    https://www.cnblogs.com/pinard/p/10609228.html
    """

    def __init__(self, parent, prior_p, name):
        self._parent = parent  # 父结点
        self._children = {}  # 子结点，代表动作，即落子动作
        self._n = 0   # 结点访问次数
        self._q = 0.  # Q值
        self._u = 0.  # U值
        self._p = prior_p  # 先验概率
        hash_str = f'{hash(self):08x}'[-8:]
        self._name = f'{name.replace("-", "_")}_{hash_str}'

    def expand(self, action_priors):
        """
        通过生成新的子结点来扩展树
        :param action_priors: 通过policy func获得的tuple（actions,prior_p）
        :return:
        """
        for action, prior in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prior, f'a_{action.x}_{action.y}')  # 父结点是self，先验概率prior

    def select(self, c_puct):
        """
        选择子结点，能够得到最大的 Q+u(P)
        :param c_puct:
        :return: (action,next_node)
        """
        items = [(act_node, act_node[1].get_value(c_puct)) for act_node in self._children.items()]
        return max_sample(items, key=itemgetter(1))[0]

    def update(self, leaf_value):
        """
        更新当前叶子结点的值
        :param leaf_value:
        :return:
        """
        self._n += 1
        self._q = self._q + 1.0 * (leaf_value - self._q) / self._n

    def update_recursive(self, leaf_value: float, level=0):
        """
        更新叶子结点后，我们还需要递归更新当前叶结点的所有祖先
        :param leaf_value:
        :param level:
        :return:
        """
        if self._parent:
            max_level = self._parent.update_recursive(-leaf_value, level + 1)
        else:
            max_level = level + 1

        self.update(leaf_value)
        return max_level

    def get_value(self, c_puct):
        """
        计算当前结点的value
        :param
            c_puct: a number in (0, inf) controlling the relative impact of
                value Q, and prior probability P, on this node's score.
        :return:
        """
        # https://www.cnblogs.com/pinard/p/10609228.html
        N = self._n if self._parent is None else self._parent.n
        self._u = (c_puct * self._p * np.sqrt(N) / (1 + self._n))
        # https://www.cnblogs.com/pinard/p/10470571.html
        # self._u = c_puct * self._p * np.sqrt(np.log(1 + self._parent.n) / (1 + self._n))  # n可能为0
        return self._u + self._q

    def is_leaf(self):
        """
        判断当前结点是否是叶子结点
        :return:
        """
        return self._children == {}

    def is_root(self):
        """
        判断当前结点是否是根结点
        :return:
        """
        return self._parent is None

    def to_dot(self, prefix='', color='white', label=None):
        if label is None:
            label = self._name
        dot = f'{prefix}{self._name} [style="filled" fillcolor="{color}" label="{label}"]\n'
        if self._parent is not None:
            label = f'n={self._n}\\np={self._p:.3f}\\nq={self._q:.3f}\\nu={self._u:.3f}'
            dot += f'{prefix}{self._parent.name} -> {self._name} [label="{label}"]\n'
        return dot

    @property
    def children(self):
        return self._children

    @property
    def name(self):
        return self._name

    @property
    def n(self):
        return self._n

    @property
    def p(self):
        return self._p

    @property
    def u(self):
        return self._u

    @property
    def q(self):
        return self._q

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent


class MCTS:
    def __init__(self, policy_value_fn, c_puct=15, n_playout=800, name=None):
        """
        :param policy_value_fn:
            当前采用的策略函数
            输入当前棋盘的状态，
            输出(action,prob)元组，和score[-1,1]
        :param c_puct:
            控制MCTS exploration and exploitation 的关系，值越大表示越依赖之前的先验概率
        :param n_playout:
            MCTS算法的执行次数，次数越大效果越好，但是耗费的时间也会越多.
        """
        self._root = TreeNode(None, 1.0, 'root')
        self._action = INVALID_ACTION
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._p1 = Player('mcts-black')
        self._p2 = Player('mcts-white')
        self._p1.piece = Piece.BLACK
        self._p2.piece = Piece.WHITE
        if name is not None:
            self._name = name
        else:
            hash_str = f'{hash(self):08x}'[-8:]
            self._name = 'mcts-' + hash_str

    def _alternate(self, last_piece):
        return self._p1 if self._p1.piece == Piece.next(last_piece) else self._p2

    def _default_value_fn(self, env, piece, action_priors):
        env_copy = GoBang()
        env.copy_to(env_copy)
        first_action = max_sample(action_priors, key=itemgetter(1))[0]
        player = self._alternate(env_copy.last_piece)
        player.act(first_action)
        env_copy.turn(player)

        actions = [action for row in Action.ACTIONS for action in row if env_copy.board[action.y][action.x] == Piece.NONE]
        while not env_copy.game_over():
            action = random.choice(actions)
            player = self._alternate(env_copy.last_piece)
            player.act(action)
            env_copy.turn(player)
            actions.remove(action)

        if env_copy.winner == Piece.NONE:
            return 0.0
        else:
            return 1.0 if env_copy.winner == piece else -1.0

    def _playout(self, env):
        """
        根据当前的状态进行play
        :param env:
        :return:
        """
        node = self._root
        while True:
            player = self._alternate(env.last_piece)
            if node.is_leaf():
                break
            # 贪心算法选择下一个move
            action, node = node.select(self._c_puct)
            player.act(action)
            env.turn(player)

        if env.game_over():
            if env.winner == Piece.NONE:
                leaf_value = 0.0
            else:
                # player代表下一个（当前）玩家，而不是node对应的玩家
                leaf_value = -1.0 if env.winner == player.piece else 1.0
        else:
            # 重新评估叶子结点
            if self._policy is None:
                # 策略无法通过先验获得，采用平均策略
                action_priors = default_policy_fn(env, player.piece)
                leaf_value = self._default_value_fn(env, player.piece, action_priors)
            else:
                action_priors, leaf_value = self._policy(env, player.piece)

            # 无子可落时，认负
            if len(action_priors) == 0:
                env.winner = env.last_piece
                # node对应的玩家是获胜的
                leaf_value = 1.0
            else:
                node.expand(action_priors)

        # 递归更新叶子结点的值和所有的祖先
        node.update_recursive(leaf_value)
        return node

    def get_action_priors(self, env, tau=1):
        """
            从当前状态开始获得所有可行动作及它们的概率，为了保证数据不出错，必须要采用深拷贝
        :param env:
            当前的游戏动作
        :param tau:
            温度系数，类似epsilon值
        :return:
        """
        root_children_count = 0
        env_copy = GoBang()
        for n in range(self._n_playout):
            env.copy_to(env_copy)
            if DEBUG:
                env_copy.history = []
            path_node = self._playout(env_copy)
            if DEBUG and env_copy.game_over():
                path_root = path_node.parent
                steps = 1
                while path_root != self._root:
                    path_node = path_root
                    path_root = path_node.parent
                    steps += 1
                if steps > 8:
                    hash_info = f'{hash(self):08x}'[-8:]
                    details_dir = os.path.join(LOG_DIR, f'{self._name}-playout-{hash_info}')
                    if not os.path.exists(details_dir):
                        os.makedirs(details_dir)

                    piece = Piece.next(env.last_piece)
                    piece_info = 'white' if piece == Piece.WHITE else 'black'
                    win_info = 'tie'
                    if env_copy.winner == piece:
                        win_info = 'win'
                    elif env_copy.winner == Piece.next(piece):
                        win_info = 'loss'
                    path_act = list(path_root.children.keys())[list(path_root.children.values()).index(path_node)]
                    hash_info = f'{hash(env_copy):08x}'[-8:]
                    time_info = datetime.datetime.now().strftime('%y%m%d-%H-%M-%S')
                    details_file = os.path.join(
                        details_dir,
                        f'{time_info}-{steps}-{piece_info}-{win_info}-{path_act.x}-{path_act.y}-{hash_info}.txt')
                    with open(details_file, 'w') as f:
                        out = sys.stdout
                        sys.stdout = f
                        print(f'playout {n + 1}, try {steps} steps, first [{path_act.x}, {path_act.y}]):')
                        env.print()
                        board = copy.deepcopy(env.board)
                        for x, y, p in env_copy.history:
                            board[y][x] = p
                            GoBang.print_board(board, Action.get(x, y))
                        print('======== MCTS tree information ========')
                        print(self.to_dot(focus=path_node))
                        sys.stdout = out
                    print(f'MCTS playout: {n + 1}, steps: {steps}, details: {details_file}')

            # 为提升性能，如果根节点只有一个选择，没必要再探索
            if root_children_count == 0:
                root_children_count = len(self._root.children.keys())
                if root_children_count == 0:
                    break
            if root_children_count == 1:
                break

        # 通过蒙特卡洛树计算所有动作的概率。探索次数相同的情况下，尊重先验策略。（先验概率小于1，所以是探索次数优先）
        action_visits = [(act, node.n + node.p) for act, node in self._root.children.items()]
        if len(action_visits) == 0:
            # 无子可走，认负
            return [INVALID_ACTION], np.array([1.0]), -1
        elif len(action_visits) == 1:
            # 只有一种可行走法
            return [action_visits[0][0]], np.array([1.0]), self._root.q
        actions, visits = zip(*action_visits)
        # 先归一化，避免指数计算溢出
        priors = prob(np.array(visits))
        if abs(1 - tau) > 1e-10:
            priors = np.power(priors, 1 / tau)
            priors = prob(priors)

        return actions, priors, self._root.q

    def update(self, action):
        """
        执行一个动作move后，更新蒙特卡洛树的子树
        :return:
        """
        self._action = action
        if action in self._root.children:
            self._root = self._root.children[action]
            self._root.parent = None
        else:
            self._root = TreeNode(None, 1.0, f'a_{action.x}_{action.y}')

    def _get_values(self, action, node, depth=1, level=0):
        v_list = defaultdict(float)
        v_list[(action, node)] = node.get_value(self._c_puct)
        if level >= depth:
            return v_list
        for act, child in node.children.items():
            v_list = {**v_list, **self._get_values(act, child, depth, level + 1)}
        return v_list

    def to_dot(self, focus=None):
        v_list = self._get_values(self._action, self._root, 1)
        choice = list(self._root.children.keys())[np.argmax([node.n for node in self._root.children.values()])]
        cc = np.array(list(v_list.values()))
        cc = cc - min(cc)
        cc = 255 - 155 * cc / max(cc)
        graph = f'digraph {self._root.name} {{\n'
        for ((action, node), v), c in zip(v_list.items(), cc):
            if self._root == node:
                color = 'lightpink'
                label = f'[{action.x}, {action.y}]\\nc={len(node.children)}\\nn={node.n}'
            else:
                if action == choice:
                    color = 'lightcoral'
                elif node == focus:
                    color = 'darkseagreen'
                else:
                    c_str = f'{int(round(c)):02x}'
                    color = f'#{c_str}{c_str}{c_str}'
                label = f'[{action.x}, {action.y}]\\nc={len(node.children)}\\nv={v:.3f}'
            sub_graph = node.to_dot(prefix='\t', color=color, label=label)
            if '' == sub_graph:
                continue
            graph += sub_graph + '\n'
        graph += '}\n'

        return graph

    @property
    def c_puct(self):
        return self._c_puct

    @property
    def n_playout(self):
        return self._n_playout

    @c_puct.setter
    def c_puct(self, c_puct):
        self._c_puct = c_puct

    @n_playout.setter
    def n_playout(self, n_playout):
        self._n_playout = n_playout

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def __str__(self):
        return self._name


def default_policy_fn(env, piece):
    if env.last_action == INVALID_ACTION:
        x = COLS // 2
        y = ROWS // 2
        return [(Action.get(x, y), 1.0)]

    actions = env.available_actions()
    priors = np.ones(len(actions), dtype=float) / len(actions)
    action_priors = []
    forced_choice = None

    for action, prior in zip(actions, priors):
        # 加入一些先验策略
        if env.is_successful(action, piece):
            return [(action, 1.0)]
        elif env.is_successful(action, Piece.next(piece)):
            forced_choice = [(action, 1.0)]
        elif env.likely_to_be_successful(action, piece):
            prior = 1.0
        elif env.likely_to_be_successful(action, Piece.next(piece)):
            prior = 0.9
        action_priors.append((action, prior))

    if forced_choice is not None:
        return forced_choice
    return action_priors
