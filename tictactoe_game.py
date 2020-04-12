import math
import numpy as np

from abc import ABC, abstractmethod
from prettytable import PrettyTable

from definitions import EMPTY, PLAYER_X, PLAYER_O, DRAW, PLAYERS, GAME_STILL_RUNNING


class Game(ABC):

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def play_action(self, action, is_simulation=False):
        pass

    @abstractmethod
    def check_winner(self):
        pass

    @abstractmethod
    def is_finished(self):
        pass

    @abstractmethod
    def print(self):
        pass


class TicTacToe(Game):
    def print(self):
        table = PrettyTable(['', 'Col1', 'Col2', 'Col3'])
        table.add_row(
            ['Row1', stringify_field_value(self.state[0]), stringify_field_value(self.state[1]),
             stringify_field_value(self.state[2])])
        table.add_row(
            ['Row2', stringify_field_value(self.state[3]), stringify_field_value(self.state[4]),
             stringify_field_value(self.state[5])])
        table.add_row(
            ['Row3', stringify_field_value(self.state[6]), stringify_field_value(self.state[7]),
             stringify_field_value(self.state[8])])
        print(table)

    def __init__(self, size, state=None, turn=0, avail_actions=None, winner=GAME_STILL_RUNNING, current_player=None,
                 sum_lines=None,
                 sum_rows=None, sum_diagonals=None):
        self.size = size
        self.turn = turn
        self.winner = winner

        self.diag_1 = np.asarray(range(0, self.size * self.size, self.size + 1))
        self.diag_2 = np.asarray(range(self.size - 1, self.size * (self.size - 1) + 1, self.size - 1))

        if state is None:
            self.state = np.full((size * size), EMPTY)
        else:
            self.state = state

        if avail_actions is None:
            self.avail_actions = np.asarray(list(range(self.size * self.size)))
        else:
            self.avail_actions = avail_actions
        if current_player is None:
            self.current_player = PLAYERS[self.turn % 2]
        else:
            self.current_player = current_player

        if sum_lines is None:
            self.sum_lines = np.zeros(3, dtype=int)
        else:
            self.sum_lines = sum_lines

        if sum_rows is None:
            self.sum_rows = np.zeros(3, dtype=int)
        else:
            self.sum_rows = sum_rows

        if sum_diagonals is None:
            self.sum_diagonals = np.asarray([0, 0])
        else:
            self.sum_diagonals = sum_diagonals

    def clone(self):
        new_game = TicTacToe(self.size, np.copy(self.state), self.turn, np.copy(self.avail_actions), self.winner,
                             self.current_player, np.copy(self.sum_lines), np.copy(self.sum_rows),
                             np.copy(self.sum_diagonals))
        return new_game

    def is_finished(self):
        return self.winner != GAME_STILL_RUNNING

    def play_action(self, action, is_action_in_real_game=False):
        self.state[action] = self.current_player

        self.sum_lines[math.floor(action / self.size)] += self.current_player
        self.sum_rows[action % self.size] += self.current_player
        if action in self.diag_1:
            self.sum_diagonals[0] += self.current_player

        if action in self.diag_2:
            self.sum_diagonals[1] += self.current_player

        self.turn += 1
        self.current_player = PLAYERS[self.turn % 2]
        self.avail_actions = self.avail_actions[self.avail_actions != action]

        if self.turn >= self.size * 2 - 1:
            return self.check_winner()
        else:
            return GAME_STILL_RUNNING

    def check_winner(self):
        for player in ([PLAYER_X, PLAYER_O]):
            sum_player = player * self.size
            if sum_player in self.sum_lines or sum_player in self.sum_rows or sum_player in self.sum_diagonals:
                self.winner = player
                return player

        if self.turn == self.size * self.size:
            self.winner = DRAW
            return DRAW
        else:
            self.winner = GAME_STILL_RUNNING
            return GAME_STILL_RUNNING


def stringify_field_value(number):
    if number == EMPTY:
        return '   '
    elif number == 1:
        return 'X'
    elif number == -1:
        return 'O'
