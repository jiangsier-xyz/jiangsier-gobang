#!/usr/bin/python
# -*- coding: utf-8 -*-

import threading
import time
import random

import argparse

import flask
from flask import *
from flask_socketio import SocketIO, emit

import uuid

from .config import *
from .gobang import Piece, INVALID_ACTION, GoBang, Player, Action
from .log_util import init_logger
from .player.robot import RobotPlayer
from .record import sgf_util
from .train import TrainFactory


def init_log(player_name):
    cur_time = time.strftime('%m%d-%H:%M:%S', time.localtime(time.time()))
    logger_name = '{}-{}-{}-{}'.format(player_name, str(ROWS), str(COLS), str(cur_time))
    logger = init_logger(name=logger_name, path=LOG_DIR)
    return logger


app = Flask(__name__)
app.config['SECRET_KEY'] = 'GoBang!'
socketio = SocketIO(app)
_logger = init_log('RobotService')


class WarmupThread(threading.Thread):
    def __init__(self):
        thread_name = 'warmup-thread-' + str(id(self))
        super(WarmupThread, self).__init__(name=thread_name)

    def run(self):
        a = time.perf_counter()
        _logger.info('Robots warmup start.')
        RobotPlayer.warmup(SGF_DIR)
        RobotPlayer.warmup(HISTORY_DIR)
        perf = time.perf_counter() - a
        _logger.info(f'Robots warmup end. Elapsed time: {perf:.3f} s')


class RobotFactory:
    robots = TrainFactory.players
    _warehouse = dict()
    _global_lock = threading.Lock()

    @staticmethod
    def robot(robot_name, game, gid=None):
        RobotFactory._global_lock.acquire()
        if gid is not None and gid in RobotFactory._warehouse:
            robot = RobotFactory._warehouse.get(gid)
            if robot.piece != Piece.next(game.last_piece):
                robot.piece = Piece.next(game.last_piece)
                robot.mcts.update(INVALID_ACTION)
        else:
            cls, c_puct, n_playout, _ = RobotFactory.robots[robot_name]
            robot = cls(c_puct=c_puct, n_playout=n_playout)
            robot.nickname = robot_name
            robot.piece = Piece.next(game.last_piece)
            if gid is not None:
                RobotFactory._warehouse[gid] = robot
        RobotFactory._global_lock.release()
        return robot

    @staticmethod
    def ask(robot_name, game, gid=None):
        robot = RobotFactory.robot(robot_name, game, gid)
        return robot.piece, robot.play(game)

    @staticmethod
    def end(gid):
        RobotFactory._global_lock.acquire()
        if gid is not None and gid in RobotFactory._warehouse:
            robot = RobotFactory._warehouse.pop(gid)
            _logger.info(f'Game {gid} ends with robot {robot.nickname}-{id(robot)}')
        RobotFactory._global_lock.release()

    @staticmethod
    def contains(robot_name):
        return robot_name in RobotFactory.robots


# WarmupThread().start()


################
# RESTful API
@app.route('/game/robot/play', methods=['GET', 'POST'])
def request_action():
    game_id = request.form.get('gid')
    robot_name = request.form['robot'].capitalize()
    sgf = request.form['sgf']
    if sgf == '' or not RobotFactory.contains(robot_name):
        return ''

    game = sgf_util.sgf_string_to_gobang(sgf)
    if game is None:
        return ''

    piece, action = RobotFactory.ask(robot_name, game, game_id)
    return sgf_util.action_to_sgf_string(piece, action)


@app.route('/game/robot/end', methods=['GET', 'POST'])
def game_over():
    RobotFactory.end(request.form['gid'])
    return 'OK'


@app.route('/favicon.ico')
def get_fav():
    return current_app.send_static_file('favicon.ico')


@app.route('/')
def home():
    return flask.render_template('go-bang.html')


@app.route('/hello')
def hello_world():
    return 'Hello! I am robot.'


################
# WebSocket API
def _uuid():
    return str(uuid.uuid4()).replace('-', '')


@socketio.on('connect', namespace='/game')
def connect():
    name = request.args['name']
    piece = request.args['piece']
    robot_name = request.args['robot']
    gid = f'{name}({piece})-vs-{robot_name}-{_uuid()}'
    session['gid'] = gid
    session['robot_name'] = robot_name.capitalize()
    session['name'] = name
    session['piece'] = piece
    _logger.info(f'{gid} connected!')


@socketio.on('disconnect')
def disconnect():
    gid = session['gid']
    _logger.info(f'{gid} disconnected!')
    RobotFactory.end(gid)
    session.clear()


@socketio.on('start', namespace='/game')
def start():
    gid = session['gid']
    robot_name = session['robot_name']
    name = session['name']
    player = Player(name)
    s_piece = session['piece'].upper()
    if s_piece == 'BLACK':
        piece = Piece.BLACK
    elif s_piece == 'WHITE':
        piece = Piece.WHITE
    else:
        piece = random.choice((Piece.BLACK, Piece.WHITE))

    player.piece = piece
    game = GoBang()

    robot = RobotFactory.robot(robot_name, game, None)
    robot.piece = Piece.next(piece)

    session['game'] = game
    session['player'] = player
    session['robot'] = robot

    _logger.info(f'{gid} game started!')

    if piece == Piece.WHITE:
        action = game.turn(robot)
        emit('act', {'action': {'x': action.x, 'y': action.y}, 'piece': robot.piece})
        _logger.info(f'{gid}: {robot_name} play [{action.x}, {action.y}]')
    emit('message', {'text': f'{name}，轮到你啦！'})


@socketio.on('play', namespace='/game')
def play(msg):
    if 'game' not in session:
        return

    gid = session['gid']
    robot = session['robot']
    game = session['game']
    player = session['player']
    action_info = msg['action']
    action = Action.get(action_info['x'], action_info['y'])

    if not game.is_valid(action):
        emit('message', {'text': f'({action.x}, {action.y})是非法位置！'})
        return

    player.act(action)
    game.turn(player)
    emit('act', {'action': {'x': action.x, 'y': action.y}, 'piece': player.piece})
    _logger.info(f'{gid}: {player.nickname} play [{action.x}, {action.y}]')

    # emit('message', {'text': f'{robot.nickname} 正在思考！'})
    action = game.turn(robot)
    if action != INVALID_ACTION:
        emit('act', {'action': {'x': action.x, 'y': action.y}, 'piece': robot.piece})
        _logger.info(f'{gid}: {robot.nickname} play [{action.x}, {action.y}]')

    if game.game_over():
        if game.winner == player.piece:
            emit('alert', {'text': f'{player.nickname} 获胜！'})
        elif game.winner == robot.piece:
            emit('alert', {'text': f'{robot.nickname} 获胜！'})
        else:
            emit('alert', {'text': '平局！'})
        _logger.info(f'{gid} game stopped!')
    else:
        emit('message', {'text': f'{player.nickname}，轮到你啦！'})


@socketio.on('power-up', namespace='/game')
def power_up():
    robot = session.get('robot')
    if robot and isinstance(robot, RobotPlayer):
        robot.with_records()


@socketio.on('power-down', namespace='/game')
def power_down():
    robot = session.get('robot')
    if robot and isinstance(robot, RobotPlayer):
        robot.without_records()


################
# Tests
def test_hello_world():
    print('running test_hello_world()...')
    with app.test_request_context('/hello'):
        response = hello_world()
        assert 'Hello! I am robot.' == response


def test_request_action():
    print('running test_request_action()...')
    data = dict()
    data['robot'] = 'Ace'
    data['sgf'] = '(;SZ[15])'
    with app.test_request_context('/game/robot/play', method='POST', data=data):
        response = request_action()
        assert 'B[hh]' == response


def test_end():
    print('running test_end()...')
    data = dict()
    data['robot'] = 'Ace'
    data['sgf'] = '(;SZ[15])'
    data['gid'] = __name__
    with app.test_request_context('/game/robot/play', method='POST', data=data):
        request_action()
    assert __name__ in RobotFactory._warehouse
    robot = RobotFactory._warehouse.get(__name__)

    data['sgf'] = '(;SZ[15];B[hh];W[hf])'
    with app.test_request_context('/game/robot/play', method='POST', data=data):
        request_action()
    assert len(RobotFactory._warehouse.keys()) == 1
    assert robot == RobotFactory._warehouse.get(__name__)

    data.pop('robot')
    data.pop('sgf')
    with app.test_request_context('/game/robot/end', method='POST', data=data):
        game_over()
    assert __name__ not in RobotFactory._warehouse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-H', '--host', default='0.0.0.0')
    parser.add_argument('-p', '--port', type=int, default=8090)
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    if args.test:
        test_hello_world()
        test_request_action()
        test_end()
    else:
        socketio.run(app, host=args.host, port=args.port, debug=args.verbose)
