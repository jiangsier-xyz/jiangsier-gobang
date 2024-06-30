import os

if os.getenv('PYTHONIDE') is None:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__ + '/..'))
    DEBUG = False
else:
    BASE_DIR = os.getcwd()
    DEBUG = True

VERSION = '1.0'
COLS = 15
ROWS = 15
WIN_NUMBER = 5

STATE_DEPTH = 3
ASYNC_TRAINING = False
SAVE_TRAINING_SET = False
USE_TRAINING_SET = True
PLAY_WITH_TEACHER = False
COLLECT_THREADS = 1
MODEL_FORMAT = 'keras'
STABILIZE_KL_DIVERGENCE = False

MODEL_DIR = BASE_DIR + '/data/models/'
CHECKPOINT_DIR = BASE_DIR + '/data/checkpoints/'
TENSORBOARD_DIR = BASE_DIR + '/data/tensorboard/'
LOG_DIR = BASE_DIR + '/data/logs'
HISTORY_DIR = BASE_DIR + '/data/train/histories/'
SGF_DIR = BASE_DIR + '/data/train/sgf/'
