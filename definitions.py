import os

PROJECT_ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data/')

EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
DRAW = 0
GAME_STILL_RUNNING = 1000
PLAYERS = [PLAYER_X, PLAYER_O]
DEBUG_PRINT = True
ACTIONS = 'actions'
WINNERS = 'winners'
