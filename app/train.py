#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import math
import random
import sys
import time
from collections import defaultdict, deque, namedtuple
from datetime import datetime
from threading import Thread

import numpy as np
import tensorflow as tf

from .config import *
from .gobang import GoBang, Piece, Action, INVALID_ACTION
from .log_util import init_logger
from .player.ace import Ace
from .player.baker import Baker
from .player.casey import Casey
from .player.darling import Darling
# from .player.ellis import Ellis
from .player.fox import Fox
from .player.gill import Gill
from .player.monkey import Monkey
from .player.robot import RobotPlayer
from .player.teacher import Teacher
from .record import sgf_util


class TrainingThread(Thread):
    def __init__(self, pipeline):
        self._pipeline = pipeline
        thread_name = pipeline.player.nickname + '-training-thread-' + str(id(self))
        super(TrainingThread, self).__init__(name=thread_name)

    def run(self):
        self._pipeline.train()

    @property
    def pipeline(self):
        return self._pipeline


class CollectingThread(Thread):
    def __init__(self, player, n_games):
        thread_id = str(id(self))
        thread_name = player.nickname + '-collecting-thread-' + thread_id
        self._player = player.__class__(name=player.nickname + '-' + thread_id,
                                        c_puct=player.mcts.c_puct,
                                        n_playout=player.mcts.n_playout,
                                        is_self_play=True,
                                        tau=player.tau)
        self._n_games = n_games
        self._play_data = []
        super(CollectingThread, self).__init__(name=thread_name)

    def run(self):
        env = GoBang()
        for i in range(self._n_games):
            env.reset()
            play_data = TrainPipeline.self_play(env, self._player, False)

            # 拓展数据集
            self._play_data.extend(TrainPipeline.extend_data(play_data))

    @property
    def play_data(self):
        return self._play_data


def _c_puct(playout):
    if playout <= 0:
        return 0

    # search_space = COLS * ROWS
    search_space = 25  # 平均每步的搜索空间，经验值
    factor = (search_space + 10 * playout) / (10 * playout)

    if playout > search_space:
        return 2 * math.sqrt(playout) * factor
    else:
        return 2 * search_space * factor / math.sqrt(playout)


class TrainFactory:
    TRAIN_PLAYOUT = 1600
    ONLINE_PLAYOUT = 0

    players = dict()
    players['Ace'] = (Ace, _c_puct(ONLINE_PLAYOUT), ONLINE_PLAYOUT, True)
    players['Baker'] = (Baker, _c_puct(ONLINE_PLAYOUT), ONLINE_PLAYOUT, True)
    players['Casey'] = (Casey, _c_puct(ONLINE_PLAYOUT), ONLINE_PLAYOUT, True)
    players['Darling'] = (Darling, _c_puct(ONLINE_PLAYOUT), ONLINE_PLAYOUT, True)
    # players['Ellis'] = (Ellis, _c_puct(ONLINE_PLAYOUT), ONLINE_PLAYOUT, True)
    players['Fox'] = (Fox, _c_puct(ONLINE_PLAYOUT), ONLINE_PLAYOUT, True)
    players['Gill'] = (Gill, _c_puct(ONLINE_PLAYOUT), ONLINE_PLAYOUT * 6, False)
    players['Monkey'] = (Monkey, _c_puct(ONLINE_PLAYOUT), ONLINE_PLAYOUT * 100, False)
    players['Teacher'] = (Teacher, _c_puct(ONLINE_PLAYOUT), ONLINE_PLAYOUT, False)
    # players['TeacherMcts'] = (Teacher, _c_puct(TRAIN_PLAYOUT), TRAIN_PLAYOUT, False)

    @staticmethod
    def init():
        TrainFactory.init_paths()
        TrainFactory.init_gpus()

    @staticmethod
    def make_dirs_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def init_paths():
        for player_name in TrainFactory.players.keys():
            TrainFactory.make_dirs_if_not_exists(MODEL_DIR + player_name + '/' + MODEL_FORMAT + '/best')
            TrainFactory.make_dirs_if_not_exists(CHECKPOINT_DIR + player_name)
            TrainFactory.make_dirs_if_not_exists(TENSORBOARD_DIR + player_name)
            TrainFactory.make_dirs_if_not_exists(LOG_DIR)
            TrainFactory.make_dirs_if_not_exists(SGF_DIR)
            TrainFactory.make_dirs_if_not_exists(HISTORY_DIR)

    @staticmethod
    def init_gpus():
        # See: https://www.tensorflow.org/guide/gpu
        # Now we try to config tf to run multiple processes on 1 GPU.
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    @staticmethod
    def init_log(player_name):
        cur_time = time.strftime('%m%d-%H:%M:%S', time.localtime(time.time()))
        logger_name = '{}-{}-{}-{}'.format(player_name, str(ROWS), str(COLS), str(cur_time))
        logger = init_logger(name=logger_name, path=LOG_DIR)
        return logger

    @staticmethod
    def alternate(player1, player2, env):
        return player1 if player1.piece == Piece.next(env.last_piece) else player2

    @staticmethod
    def vs(player1, player2, rand_pieces=True, print_board=False, print_prefix='', record_board=False):
        win_info = defaultdict(int)
        env = GoBang()
        if record_board:
            env.history = []
        if rand_pieces:
            player1.piece = random.choice((Piece.BLACK, Piece.WHITE))
            player2.piece = Piece.next(player1.piece)

        if print_board:
            black = player1 if player1.piece == Piece.BLACK else player2
            white = player1 if player1.piece == Piece.WHITE else player2
            print(f'{print_prefix}{black.nickname}(x) vs {white.nickname}(o) :')

        while not env.game_over():
            player = TrainFactory.alternate(player1, player2, env)
            env.turn(player, print_board)

        win_info[env.winner] += 1
        return win_info, env

    @staticmethod
    def arena(game_batch_num=1000):
        Hero = namedtuple('Hero', ['player', 'profile'])
        heroes = []
        for player_name, (cls, c_puct, n_playout, _) in TrainFactory.players.items():
            player = cls(c_puct=c_puct, n_playout=n_playout)
            player.nickname = player_name
            if isinstance(player, RobotPlayer):
                player.random_factor = 0.0
            hero = Hero(player=player, profile=defaultdict(int))
            heroes.append(hero)

        for i in range(game_batch_num):
            hero1, hero2 = random.sample(heroes, 2)
            win_info, env = TrainFactory.vs(hero1.player, hero2.player,
                                            print_board=True,
                                            print_prefix=f'Game {i + 1} --')
            TrainFactory.statistics(hero1.player, hero1.profile, win_info)
            TrainFactory.statistics(hero2.player, hero2.profile, win_info)

        print('')
        print('Completed {} games! Result:'.format(str(game_batch_num)))
        heroes.sort(key=lambda h: h.profile['ratio'], reverse=True)
        for hero in heroes:
            print('player:{}, total:{}, wins:{}, losses:{}, ties:{}, ratio:{:.3f}'.format(
                hero.player.nickname, hero.profile['total'], hero.profile['wins'],
                hero.profile['losses'], hero.profile['ties'], hero.profile['ratio']))

    @staticmethod
    def statistics(player, profile, current_info, logger=None):
        tie = current_info[Piece.NONE]
        win = current_info[player.piece]
        lose = current_info[Piece.next(player.piece)]
        n = tie + win + lose

        profile['total'] += n
        profile['wins'] += win
        profile['losses'] += lose
        profile['ties'] += tie
        profile['ratio'] = (1.0 * profile['wins'] + 0.5 * profile['ties']) / (1.0 * profile['total']) \
            if profile['total'] > 0 else 0

        print_fn = print if logger is None else logger.info
        print_fn('player:{}, total:{}(+{}), wins:{}(+{}), losses:{}(+{}), ties:{}(+{}), ratio:{:.3f}'.format(
                player.nickname,
                profile['total'], n,
                profile['wins'], win,
                profile['losses'], lose,
                profile['ties'], tie,
                profile['ratio']))

    @staticmethod
    def train_all(game_batch_num=1000):
        if ASYNC_TRAINING:
            threads = []
            for player_name, (cls, c_puct, n_playout, trainable) in TrainFactory.players.items():
                if not trainable:
                    continue
                player = cls(c_puct=c_puct, n_playout=n_playout)
                player.nickname = player_name
                pipeline = TrainPipeline(player, game_batch_num=game_batch_num)
                thread = TrainingThread(pipeline)
                threads.append(thread)
                pipeline.logger.info('Start training ' + player.nickname)
                thread.start()
            for thread in threads:
                thread.join()
                thread.pipeline.logger.info('Finish training ' + thread.pipeline.player.nickname)
        else:
            for player_name, (cls, c_puct, n_playout, trainable) in TrainFactory.players.items():
                if not trainable:
                    continue
                player = cls(c_puct=c_puct, n_playout=n_playout)
                player.nickname = player_name
                pipeline = TrainPipeline(player, game_batch_num=game_batch_num)
                pipeline.logger.info('Start training ' + player.nickname)
                pipeline.train()
                pipeline.logger.info('Finish training ' + player.nickname)

    @staticmethod
    def train(player_name, game_batch_num=1000):
        cls, c_puct, n_playout, _ = TrainFactory.players[player_name]
        player = cls(c_puct=c_puct, n_playout=n_playout)
        player.nickname = player_name
        pipeline = TrainPipeline(player, game_batch_num=game_batch_num)
        pipeline.logger.info('Start training ' + player.nickname)
        pipeline.train()
        pipeline.logger.info('Finish training ' + player.nickname)


class TrainPipeline:
    def __init__(self, player, game_batch_num=1000):
        self._player = player
        self._buffer_size = 10000
        self._batch_size = 1024  # 每次取batch_size进行梯度下降
        self._data_buffer = deque(maxlen=self._buffer_size)
        self._data_batch = deque(maxlen=self._batch_size)

        self._game_batch_num = game_batch_num  # 最大收集/训练轮数
        self._epochs = 10  # 每批数据的训练次数
        self._kl_target = 0.02  # 目标KL散度，如果KL散度开始发散，提前终止训练
        self._learning_rate = 2e-3
        self._lr_multiplier = 1.0  # 根据KL散度自动调整学习速率
        self._check_freq = 100  # 每训练一段时间做一次评估（对战Monkey）
        self._evaluation_info = defaultdict(int)

        self._logger = TrainFactory.init_log(player.nickname)

        if USE_TRAINING_SET:
            self._training_set = self.load_training_set()

    @staticmethod
    def extend_data(play_data):
        extend_data = []
        for board, last_action, piece, action_priors, value in play_data:
            if last_action == INVALID_ACTION:  # 开局空棋盘，没必要做旋转、镜像扩展。
                extend_data.append((GoBang.board_to_state(board, piece), action_priors, value))
                continue

            step = 1 if COLS == ROWS else 2
            # 通过旋转、镜像增加数据集
            for i in range(0, 4, step):
                # rotate counterclockwise
                extend_board = np.rot90(board, i)
                extend_action_priors = np.rot90(action_priors, i)
                extend_data.append((GoBang.board_to_state(extend_board, piece),
                                    extend_action_priors, value))

                # flip horizontally
                extend_board = np.fliplr(extend_board)
                extend_action_priors = np.fliplr(extend_action_priors)
                extend_data.append((GoBang.board_to_state(extend_board, piece),
                                    extend_action_priors, value))
        return extend_data

    @property
    def player(self):
        return self._player

    @property
    def logger(self):
        return self._logger

    @staticmethod
    def self_play(env, player, print_board=False):
        # 真正的自我对弈，需要足够的蒙特卡洛搜索次数，以获得较好的局面，否则水平无法得到提高。但蒙特卡洛搜索太慢了，训练效率极低。
        # 在水平还较低的情况下，如果有明显较强的对手，可以通过学习对手来快速提高水平。直到瓶颈期再考虑完全自我对弈。
        if PLAY_WITH_TEACHER:
            if SAVE_TRAINING_SET:
                teacher_playout = TrainFactory.TRAIN_PLAYOUT
            else:
                teacher_playout = TrainFactory.ONLINE_PLAYOUT
            teacher = Teacher(c_puct=_c_puct(teacher_playout), n_playout=teacher_playout)
            player.reset(c_puct=_c_puct(TrainFactory.ONLINE_PLAYOUT), n_playout=TrainFactory.ONLINE_PLAYOUT)
            player.random_factor = 0.5
            player.piece = random.choice((Piece.BLACK, Piece.WHITE))
            teacher.piece = Piece.next(player.piece)
            player.history = None
            teacher.history = []

            while not env.game_over():
                current_player = TrainFactory.alternate(player, teacher, env)
                env.turn(current_player, print_board)

            player.history = teacher.history
            if SAVE_TRAINING_SET:
                TrainPipeline.save_training_set(teacher, player, env)
            teacher.history = None
        else:
            player.random_factor = 0.3
            while not env.game_over():
                env.turn(player, print_board)

        play_data = []
        for board, last_action, piece, action_priors, value in player.history:
            play_data.append((board, last_action, piece, action_priors, value))
        return play_data

    def collect_self_play_data(self, n_games=1):
        if COLLECT_THREADS <= 1:
            env = GoBang()
            for i in range(n_games):
                env.reset()
                play_data = self.self_play(env, self.player, False)

                # 拓展数据集
                play_data = self.extend_data(play_data)
                self._data_buffer.extend(play_data)
        else:
            task_n_games = defaultdict(int)
            task_idx = 0
            for i in range(n_games):
                task_n_games[task_idx] += 1
                task_idx = (task_idx + 1) % COLLECT_THREADS

            threads = []
            for i in range(min(n_games, COLLECT_THREADS)):
                thread = CollectingThread(self._player, task_n_games[i])
                threads.append(thread)
                self._logger.info('Start collecting ' + thread.name)
                thread.start()
            for thread in threads:
                thread.join()
                self._logger.info('Finish collecting ' + thread.name)
                self._data_buffer.extend(thread.play_data)

    def policy_update(self):
        random.shuffle(self._data_buffer)
        for i in range(self._batch_size):
            self._data_batch.append(self._data_buffer.popleft())
        state_batch, action_priors, winner_batch = zip(*self._data_batch)
        state_batch = np.stack(state_batch)
        action_priors = np.stack(action_priors)
        winner_batch = np.stack(winner_batch)
        history = None

        if STABILIZE_KL_DIVERGENCE:
            # 考虑KL散度变化，动态调整学习率
            kl = self._kl_target
            old_priors, old_v = self.player.brain.policy_value(state_batch)
            for i in range(self._epochs):
                history = self.player.train(state_batch, action_priors, winner_batch,
                                            learning_rate=self._learning_rate * self._lr_multiplier)
                new_priors, new_v = self.player.brain.policy_value(state_batch)
                kl = np.mean(np.sum(old_priors * (np.log(old_priors + 1e-10) - np.log(new_priors + 1e-10)), axis=1))
                # 防止很少出现的异常局面（可能是非常差的或者非常好的）对已训练结果影响太大，增加训练过程鲁棒性
                if kl > self._kl_target * 4:  # early stopping if D_KL diverges badly
                    self._logger.warning('Oops, KL divergence is too high: ' + str(kl))
                    break

            if kl > self._kl_target * 2 and self._lr_multiplier > 0.1:
                self._lr_multiplier /= 1.5
            elif kl < self._kl_target / 2 and self._lr_multiplier < 10:
                self._lr_multiplier *= 1.5
        else:
            # 不考虑KL散度变化，直接学习
            for i in range(self._epochs):
                history = self.player.train(state_batch, action_priors, winner_batch,
                                            learning_rate=self._learning_rate * self._lr_multiplier)

        return history

    @staticmethod
    def save_training_set(teacher, player, env):
        win_info = 'tie'
        if env.winner == teacher.piece:
            win_info = 'win'
        elif env.winner == player.piece:
            win_info = 'loss'
        player1 = teacher.nickname if teacher.piece == Piece.BLACK else player.nickname
        player2 = player.nickname if teacher.piece == Piece.BLACK else teacher.nickname
        hash_info = f'{hash(env):08x}'[-8:]
        time_info = datetime.now().strftime('%y%m%d-%H-%M-%S')
        record_path = os.path.join(HISTORY_DIR,
                                   f'{time_info}-{player1}-{player2}-{win_info}-{len(teacher.history)}-{hash_info}.his')
        teacher.save_history(record_path)

    def load_training_set(self):
        if PLAY_WITH_TEACHER:
            record_dir = HISTORY_DIR
            record_files = glob.glob(record_dir + '*.his')
        else:
            record_dir = SGF_DIR
            record_files = glob.glob(record_dir + '*.sgf')
        self._logger.info(f'Loading {len(record_files)} training samples from {record_dir}')
        if len(record_files) < self._game_batch_num:
            cycles = self._game_batch_num // len(record_files)
            remainder = self._game_batch_num % len(record_files)
            training_set = record_files * cycles
            training_set.extend(random.sample(record_files, remainder))
        elif len(record_files) > self._game_batch_num:
            training_set = random.sample(record_files, self._game_batch_num)
        else:
            training_set = record_files

        random.shuffle(training_set)
        assert len(training_set) == self._game_batch_num
        return training_set

    def collect_training_set_data(self, path):
        if PLAY_WITH_TEACHER:
            play_data = RobotPlayer.load_history_data(path)
            assert play_data is not None
        else:
            env = sgf_util.sgf_file_to_gobang(path)
            winner_piece = env.winner
            play_data = []
            board = np.ones_like(env.board) * Piece.NONE
            last_action = INVALID_ACTION
            step_sum = len(env.history)
            step_count = 0
            for x, y, p in env.history:
                action_priors = np.zeros_like(env.board, dtype=float)
                action_priors[y][x] = 1.0
                if winner_piece == Piece.NONE:
                    value = 0.0
                else:
                    value = 1.0 if winner_piece == p else -1.0
                # 前期策略对最终胜负的影响不具决定性，此处减少价值摇摆，避免价值网络不收敛。
                # （早期同样的局面，比如第1步居中，可能因为最终胜负的不确定性，value反复出现1和-1）
                value = value * step_count / step_sum
                step_count += 1

                play_data.append((board.copy(), last_action, p, action_priors, value))
                board[y][x] = p
                last_action = Action.get(x, y)

        # 拓展数据集
        play_data = self.extend_data(play_data)
        self._data_buffer.extend(play_data)

    def train(self, evaluation=True):
        teacher = Teacher(c_puct=_c_puct(TrainFactory.ONLINE_PLAYOUT), n_playout=TrainFactory.ONLINE_PLAYOUT)\
            if evaluation else None
        best_score = 0.0
        best_model_path = MODEL_DIR + '{0}/{1}/best/{0}-{2}-{3}-'.format(
            self._player.nickname, MODEL_FORMAT, str(ROWS), str(COLS))
        try:
            for i in range(self._game_batch_num):
                if len(self._data_buffer) < self._batch_size:
                    if USE_TRAINING_SET:
                        self.collect_training_set_data(self._training_set[i])
                    else:
                        self._player.reset(c_puct=_c_puct(TrainFactory.TRAIN_PLAYOUT),
                                           n_playout=TrainFactory.TRAIN_PLAYOUT,
                                           is_self_play=True)
                        self.collect_self_play_data(COLLECT_THREADS)

                if len(self._data_buffer) >= self._batch_size:
                    history = self.policy_update()
                    if history is not None:
                        self._logger.info('batch i:{},\thistory:{}'.format(i + 1, history.history))

                if (i + 1) % self._check_freq == 0:
                    self._player.brain.save()
                    if not evaluation:
                        continue

                    self._logger.info('current self-play batch: {}'.format(i + 1))
                    self._player.reset(c_puct=_c_puct(TrainFactory.ONLINE_PLAYOUT),
                                       n_playout=TrainFactory.ONLINE_PLAYOUT,
                                       is_self_play=False)
                    tie = self._evaluation_info['ties']
                    win = self._evaluation_info['wins']
                    n = self._evaluation_info['total']
                    for n_game in range(100):
                        win_info, _ = TrainFactory.vs(self._player, teacher,
                                                      print_board=True,
                                                      print_prefix=f'Batch {i + 1} Game {n_game + 1} --')
                        TrainFactory.statistics(self._player, self._evaluation_info, win_info, self._logger)
                    tie = self._evaluation_info['ties'] - tie
                    win = self._evaluation_info['wins'] - win
                    n = self._evaluation_info['total'] - n
                    score = 1.0 * (win + 0.5 * tie) / n
                    if score >= best_score:
                        self._logger.info(f'update new best policy, score: {score:.3f}')
                        best_score = score
                        model_file = best_model_path + datetime.now().strftime('%Y%m%d-%H%M%S') + '.' + MODEL_FORMAT
                        self._player.brain.save(model_file=model_file)
        except KeyboardInterrupt:
            snapshot_model_path = MODEL_DIR + '{0}/{1}/{0}-{2}-{3}-snapshot.{4}'.format(
                self._player.nickname, MODEL_FORMAT, str(ROWS), str(COLS), MODEL_FORMAT)
            self._player.brain.save(snapshot_model_path)
            self._logger.info('quit')
        finally:
            self._player.brain.save()


def main():
    player_name = None
    game_batch_num = 500

    if len(sys.argv) > 1:
        player_name = sys.argv[1]
    if len(sys.argv) > 2:
        game_batch_num = int(sys.argv[2])

    TrainFactory.init()
    if player_name == 'arena' or player_name == 'Arena':
        TrainFactory.arena(game_batch_num)
        return

    if player_name is None:
        TrainFactory.train_all(game_batch_num)
    else:
        TrainFactory.train(player_name, game_batch_num)


if __name__ == '__main__':
    main()
