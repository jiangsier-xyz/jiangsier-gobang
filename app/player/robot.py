#!/usr/bin/python
# -*- coding: utf-8 -*-

import abc
import glob
import json
import os
import random
import threading
from collections import OrderedDict
from datetime import *

import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from keras import *

from ..config import STATE_DEPTH, COLS, ROWS, MODEL_FORMAT, MODEL_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR
from ..gobang import INVALID_ACTION, Action, Piece, Player, GoBang
from ..math_util import prob
from ..mcts import MCTS
from ..record import sgf_util
from ..residual_block import *


class RobotBrain(metaclass=abc.ABCMeta):
    _brains = dict()
    _global_lock = threading.Lock()

    @staticmethod
    def get(brain_class, load_model=True):
        if not RobotBrain._brains.__contains__(brain_class.__name__):
            RobotBrain._global_lock.acquire()
            if not RobotBrain._brains.__contains__(brain_class.__name__):
                RobotBrain._brains[brain_class.__name__] = brain_class(load_model=load_model)
            RobotBrain._global_lock.release()
        return RobotBrain._brains[brain_class.__name__]

    def __init__(self, nickname, load_model=True, model_file=None, checkpoint=False, tensor_board=False):
        self._model_file = self._default_model_file(nickname) if model_file is None else model_file
        self._model = None
        if load_model and os.path.exists(self._model_file):
            self.load()
        else:
            self._build_net(nickname + '-Model')

        # See: https://stackoverflow.com/questions/40850089/is-keras-thread-safe
        # noinspection PyUnresolvedReferences
        if self._model is not None:
            self._model.make_predict_function()

        dt = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._callbacks = []
        if checkpoint:
            self._callbacks.append(
                callbacks.ModelCheckpoint(
                    CHECKPOINT_DIR + nickname + '/' + dt + '-{epoch:02d}-{loss:.3f}.' + MODEL_FORMAT)
            )
        if tensor_board:
            self._callbacks.append(
                callbacks.TensorBoard(
                    TENSORBOARD_DIR + nickname + '/' + dt + '.log', histogram_freq=1)
            )

    @staticmethod
    def _default_model_file(nickname):
        return MODEL_DIR + '{0}/{1}/{0}-{2}-{3}.{4}'.format(nickname, MODEL_FORMAT, str(ROWS), str(COLS), MODEL_FORMAT)

    def save(self, model_file=None):
        if model_file is None:
            model_file = self._model_file
        models.save_model(self._model, filepath=model_file)

    def load(self, model_file=None):
        if model_file is None:
            model_file = self._model_file
        self._model = models.load_model(model_file,
                                        custom_objects={'PolicyLoss': PolicyLoss,
                                                        'BasicBlock': BasicBlock,
                                                        'BottleNeck': BottleNeck,
                                                        'FcBlock': FcBlock})
        assert self._model is not None
        print("Model {} is successfully loaded.".format(model_file))
        self._model.summary()

    def policy_value(self, states):
        return self._model.predict(states, batch_size=1)

    def train(self, state_batch, action_priors, winner_batch, learning_rate=None):
        if learning_rate is not None:
            self._model.optimizer.learning_rate = learning_rate

        winner_batch = np.reshape(winner_batch, (-1, 1))  # 转化成列向量
        return self._model.fit(x=state_batch,
                               y={'policy_output': action_priors, 'value_output': winner_batch},
                               callbacks=self._callbacks)

    def _build_net(self, model_name='Robot-Model'):
        backend.clear_session()
        inputs = layers.Input(shape=GoBang.state_shape())

        x = self._build_shared_net(inputs)
        p = self._build_policy_net(x)
        v = self._build_value_net(x)
        self._compile_net(inputs, p, v, model_name)

    @abc.abstractmethod
    def _build_shared_net(self, inputs):
        # 共享网络
        NotImplementedError('Must be implemented in subclasses.')

    @staticmethod
    def _build_policy_net(x):
        # 策略网络
        p = make_conv2d_layer(filters=64, kernel_size=(3, 3))(x)
        p = layers.BatchNormalization()(p)
        p = layers.ReLU()(p)
        p = make_conv2d_layer(filters=STATE_DEPTH, kernel_size=(1, 1), activation=activations.relu)(p)
        p = layers.Flatten()(p)
        p = layers.Dense(ROWS * COLS, activation=activations.relu, kernel_regularizer=regularizers.l2())(p)
        p = layers.Softmax()(p)
        return layers.Reshape((ROWS, COLS), name='policy_output')(p)

    @staticmethod
    def _build_value_net(x):
        # 价值网络
        v = make_conv2d_layer(filters=64, kernel_size=(3, 3))(x)
        v = layers.BatchNormalization()(v)
        v = layers.ReLU()(v)
        v = make_conv2d_layer(filters=STATE_DEPTH, kernel_size=(1, 1), activation=activations.relu)(v)
        v = layers.Flatten()(v)
        return layers.Dense(1, activation=activations.tanh, name='value_output')(v)

    def _compile_net(self, inputs, p_output, v_output, model_name='Robot-Model'):
        self._model = models.Model(name=model_name, inputs=inputs, outputs=[p_output, v_output])
        self._model.summary()
        self._model.compile(optimizer=optimizers.Adam(),
                            loss={'policy_output': PolicyLoss(),
                                  'value_output': losses.MeanSquaredError()})


class RobotPlayer(Player):
    _records = OrderedDict()

    def __init__(self, name, brain, c_puct=15, n_playout=300, is_self_play=False, tau=1):
        super(RobotPlayer, self).__init__(name)
        self._brain = brain
        self._is_self_play = is_self_play
        self._tau = tau
        if n_playout > 0:
            self._mcts = MCTS(self.policy_value_fn, c_puct=c_puct, n_playout=n_playout, name='mcts-' + name)
        else:
            self._mcts = None
        self._cache_size = 300000
        self._policy_value_cache = OrderedDict()
        self._history = None
        if is_self_play:
            self.piece = Piece.BLACK
            self._history = []
        self._mcts_history = None
        self._random_factor = -1.0

    def reset(self, c_puct=15, n_playout=800, is_self_play=False, tau=1):
        self._is_self_play = is_self_play
        self._tau = tau
        if n_playout <= 0:
            self._mcts = None
        elif self._mcts is None:
            self._mcts = MCTS(self.policy_value_fn, c_puct=c_puct, n_playout=n_playout, name='mcts-' + self.nickname)
        else:
            self._mcts.c_puct = c_puct
            self._mcts.n_playout = n_playout
            self._mcts.update(INVALID_ACTION)
        # self._policy_value_cache.clear()
        if is_self_play:
            self.piece = Piece.BLACK
            if self._history is None:
                self._history = []
        if self._history is not None:
            self._history.clear()
        if self._mcts_history is not None:
            self._mcts_history.clear()
        self._random_factor = -1.0

    def _mcts_update(self, action):
        if self._mcts is not None:
            self._mcts.update(action)

    def play(self, env):
        if not self._is_self_play:
            self._mcts_update(env.last_action)

        piece = self.piece
        if self._mcts is not None:
            actions, priors, v = self._mcts.get_action_priors(env, self._tau)
        else:
            action_priors, v = self.policy_value_fn(env, piece)
            actions, priors = zip(*action_priors) if len(action_priors) > 0 else ([], [])

        # 无子可落时，认负
        if len(actions) == 0 or (len(actions) == 1 and actions[0] == INVALID_ACTION):
            return INVALID_ACTION

        if 1e-3 < self._random_factor < 1.0:
            action = np.random.choice(
                actions,
                p=(1 - self._random_factor) * np.asarray(priors) +
                    self._random_factor * np.random.dirichlet(np.ones(len(priors), dtype=float))
            )
        elif self._random_factor < -1e-3:
            action = actions[np.argmax(priors)]
        else:
            action = np.random.choice(actions, p=priors)

        if self._is_self_play:
            self.piece = Piece.next(piece)

        if self._history is not None and action != INVALID_ACTION:
            action_priors = np.zeros_like(env.board, dtype=float)
            for a, p in zip(actions, priors):
                action_priors[a.y][a.x] = p
            self._history.append((env.board.copy(), env.last_action, piece, action_priors, v))

        if self._mcts_history is not None and self._mcts is not None:
            self._mcts_history.append(self._mcts.to_dot())

        self._mcts_update(action)
        return action

    @staticmethod
    def _regularize_record(original_record):
        board, last_action, piece, action_priors, v = original_record
        return board.tolist(), sgf_util.action_to_sgf_string(piece, last_action), action_priors.tolist(), v

    @staticmethod
    def _revert_record(regularized_record):
        board_list, sgf_string, action_priors_list, v = regularized_record
        board = np.asarray(board_list)
        piece, last_action = sgf_util.sgf_string_to_action(sgf_string)
        action_priors = np.asarray(action_priors_list)
        return board, last_action, piece, action_priors, v

    @staticmethod
    def _regularize_history(original_history):
        return list(map(RobotPlayer._regularize_record, original_history))

    @staticmethod
    def _revert_history(regularized_history):
        return list(map(RobotPlayer._revert_record, regularized_history))

    @staticmethod
    def save_history_data(history, path):
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'w') as f:
            json.dump(RobotPlayer._regularize_history(history), f)

    @staticmethod
    def load_history_data(path):
        history_data = None
        if os.path.exists(path):
            with open(path, 'r') as f:
                history_data = RobotPlayer._revert_history(json.load(f))
        return history_data

    def save_history(self, path):
        if self._history is None:
            return
        RobotPlayer.save_history_data(self._history, path)

    def load_history(self, path):
        self._history = RobotPlayer.load_history_data(path)

    @staticmethod
    def _board_to_str(board, piece):
        board_hash = 'b' if piece == Piece.BLACK else 'w'
        for j in range(ROWS):
            for i in range(COLS):
                piece = board[j][i]
                if piece == Piece.BLACK:
                    board_hash += 'x'
                elif piece == Piece.WHITE:
                    board_hash += 'o'
                else:
                    board_hash += '-'
        return str(hex(hash(board_hash)))

    @staticmethod
    def _update_records(key, value):
        RobotPlayer._records[key] = value

    def _update_cache(self, key, value):
        if self._policy_value_cache == RobotPlayer._records:
            return

        if len(self._policy_value_cache) >= self._cache_size:
            self._policy_value_cache.popitem(last=False)
        self._policy_value_cache[key] = value

    @staticmethod
    def extend_data(play_data):
        extend_data = []
        for board, last_action, piece, action_priors, value in play_data:
            if last_action == INVALID_ACTION:  # 开局空棋盘，没必要做旋转、镜像扩展。
                extend_data.append((board, piece, action_priors, value))
                continue

            step = 1 if COLS == ROWS else 2
            for i in range(0, 4, step):
                # rotate counterclockwise
                extend_board = np.rot90(board, i)
                extend_action_priors = np.rot90(action_priors, i)
                extend_data.append((extend_board, piece, extend_action_priors, value))

                # flip horizontally
                extend_board = np.fliplr(extend_board)
                extend_action_priors = np.fliplr(extend_action_priors)
                extend_data.append((extend_board, piece, extend_action_priors, value))
        return extend_data

    @staticmethod
    def warmup(path):
        print(f'RobotPlayer warm-up from {path}.')
        record_count = len(RobotPlayer._records)
        record_files = glob.glob(os.path.join(path, '*.sgf'))
        random.shuffle(record_files)
        for f in record_files:
            env = sgf_util.sgf_file_to_gobang(f)
            winner_piece = env.winner
            board = np.ones_like(env.board) * Piece.NONE
            step_sum = len(env.history)
            step_count = 0
            for x, y, p in env.history:
                if winner_piece == Piece.NONE:
                    value = 0.0
                else:
                    value = 1.0 if winner_piece == p else -1.0
                value = value * step_count / step_sum
                step_count += 1
                board[y][x] = p
                action = Action.get(x, y)
                action_priors = [(action, 1.0)]
                p_v_key = RobotPlayer._board_to_str(board, p)
                RobotPlayer._update_records(p_v_key, (action_priors, value))
                board[y][x] = p

        record_files = glob.glob(os.path.join(path, '*.his'))
        random.shuffle(record_files)
        for f in record_files:
            play_data = RobotPlayer.load_history_data(f)
            if play_data is None:
                continue
            play_data = RobotPlayer.extend_data(play_data)
            for board, p, weights, value in play_data:
                action_priors = []
                for action in Action.ALL:
                    if weights[action.y][action.x] > 1e-10:
                        action_priors.append((action, weights[action.y][action.x]))
                p_v_key = RobotPlayer._board_to_str(board, p)
                RobotPlayer._update_records(p_v_key, (action_priors, value))

        added_records = len(RobotPlayer._records) - record_count
        print(f'RobotPlayer warmed-up by {len(RobotPlayer._records)}(+{added_records}) records.')

    def with_records(self):
        self._policy_value_cache = RobotPlayer._records

    def without_records(self):
        self._policy_value_cache = OrderedDict()

    def train(self, state_batch, action_priors, winner_batch, learning_rate=None):
        if self._policy_value_cache != RobotPlayer._records:
            self._policy_value_cache.clear()
        return self._brain.train(state_batch, action_priors, winner_batch, learning_rate)

    def policy_value(self, env, piece):
        p_v_key = self._board_to_str(env.board, piece)
        cached_policy_value = self._policy_value_cache.get(p_v_key)
        if cached_policy_value is not None:
            if self._policy_value_cache == RobotPlayer._records:
                print('Found policy_value from records!')
            return cached_policy_value

        state = env.to_state(piece)
        priors, value = self._brain.policy_value(state[np.newaxis])
        priors, value = np.squeeze(priors, axis=0), float(value)
        weights = np.zeros_like(env.board, dtype=float)
        action_priors = []
        forced_choice = None

        actions = env.available_actions()
        for action in actions:
            prior = priors[action.y][action.x]
            if env.is_successful(action, piece):
                choice = [(action, 1.0)], 1.0
                self._update_cache(p_v_key, choice)
                return choice
            elif env.is_successful(action, Piece.next(piece)):
                forced_choice = [(action, 1.0)], value
            elif env.likely_to_be_successful(action, piece):
                prior = 1.0
            elif env.likely_to_be_successful(action, Piece.next(piece)):
                prior = 0.9
            weights[action.y][action.x] = prior + 1e-10

        weights = prob(weights)

        for action in actions:
            if weights[action.y][action.x] > 1e-10:
                action_priors.append((action, weights[action.y][action.x]))

        if forced_choice is not None:
            choice = forced_choice
        else:
            choice = action_priors, value

        self._update_cache(p_v_key, choice)
        return choice

    def policy_value_fn(self, env, piece):
        if env.last_action == INVALID_ACTION:
            x = COLS // 2
            y = ROWS // 2
            return [(Action.get(x, y), 1.0)], 0.

        return self.policy_value(env, piece)

    @property
    def brain(self):
        return self._brain

    @property
    def mcts(self):
        return self._mcts

    @mcts.setter
    def mcts(self, mcts):
        self._mcts = mcts

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, history):
        self._history = history

    @property
    def mcts_history(self):
        return self._mcts_history

    @mcts_history.setter
    def mcts_history(self, mcts_history):
        self._mcts_history = mcts_history

    @property
    def tau(self):
        return self._tau

    @property
    def random_factor(self):
        return self._random_factor

    @random_factor.setter
    def random_factor(self, factor):
        self._random_factor = factor

    @Player.nickname.setter
    def nickname(self, name):
        self._name = name
        if self._mcts is not None:
            self._mcts.name = 'mcts-' + name


class PolicyLoss(Loss):
    def __init__(self, reduction="sum_over_batch_size", name='policy_loss'):
        super(PolicyLoss, self).__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-10), axis=(1, 2))
