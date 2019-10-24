import math
import random
import numpy as np
from timeit import default_timer as timer
from abc import ABC, abstractmethod

EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
DRAW = 100000


class Game(ABC):
    #
    # @property
    # @abstractmethod
    # def size(self):
    #     pass
    #
    # @property
    # @abstractmethod
    # def state(self):
    #     pass
    #
    # @property
    # @abstractmethod
    # def turn(self):
    #     pass

    @abstractmethod
    def available_actions(self):
        pass

    @abstractmethod
    def play_action(self, player, index):
        pass

    @abstractmethod
    def check_winner(self):
        pass


class TTTGame(Game):
    def __init__(self, size):
        self.size = size
        self.state = np.full((size * size), EMPTY)
        self.turn = 0

    def available_actions(self):
        actions = np.where(self.state == EMPTY)
        return actions  # or actions[0]?

    def play_action(self, player, index):
        self.state[index] = player
        self.turn += 1
        if self.turn >= 5:  # make 5 a magic number depending on the number of x needed in a row to win ?
            return self.check_winner()
        else:
            return EMPTY

    def check_winner(self):
        # i think both can be done in one , check with tests!!!1

        # lines
        for line in range(self.size):
            sum_line = 0
            for row in range(self.size):
                position = line * self.size + row
                sum_line += self.state[position]
            for player in ([PLAYER_X, PLAYER_O]):
                if sum_line == player * self.size:
                    return player
        # rows
        for row in range(self.size):
            sum_row = 0
            for line in range(self.size):
                position = line * self.size + row
                sum_row += self.state[position]
            for player in ([PLAYER_X, PLAYER_O]):
                if sum_row == player * self.size:
                    return player

        # diagonal1
        sum_diag_1 = 0
        sum_diag_2 = 0
        position_diag_1 = 0
        position_diag_2 = 0
        for index in range(self.size):
            sum_diag_1 += self.state[position_diag_1]
            sum_diag_2 += self.state[position_diag_2]
            position_diag_1 += self.size + 1
            position_diag_2 += self.size - 1
        for player in ([PLAYER_X, PLAYER_O]):
            if sum_diag_1 == player * self.size or sum_diag_2 == player * self.size:
                return player

        if self.turn == self.size * self.size:
            return DRAW
        else:
            return EMPTY


class Bot(ABC):
    @abstractmethod
    def select_action(self, game, avail_actions):
        pass


class RandBot(Bot):

    def select_action(self, game, avail_actions):
        return avail_actions[math.floor(random.random() * avail_actions.size)]


class InOrderBot(Bot):

    def select_action(self, game, avail_actions):
        return avail_actions[0]


class BackwardsInOrderBot(Bot):
    def select_action(self, game, avail_actions):
        return avail_actions[avail_actions.size - 1]


class GoodBot(Bot):

    def select_action(self, game, avail_actions):
        pass


def play_game_till_end(size, bot_x, bot_o):
    ttt = TTTGame(size)
    game_ended = EMPTY
    player = PLAYER_X
    bots = [bot_x, bot_o]
    while game_ended == EMPTY:
        avail_actions = ttt.available_actions()[0]
        action = bots[ttt.turn % 2].select_action(ttt, avail_actions)
        game_ended = ttt.play_action(player, action)
        player = player * -1
    return game_ended


def play_many_games(number, size):
    total_result = 0
    wins_player_x = 0
    wins_player_o = 0
    draws = 0
    bot_x = InOrderBot()
    bot_y = RandBot()
    for index in range(number):
        result = play_game_till_end(size, bot_x, bot_y)
        if result == PLAYER_X:
            wins_player_x += 1
            total_result += result
        elif result == PLAYER_O:
            wins_player_o += 1
            total_result += result
        elif result == DRAW:
            draws += 1
        else:
            print('Shitttttttt')

    print('Wins X: ' + str(wins_player_x) + ' Wins O: ' + str(wins_player_o) + ' And Draws: ' + str(
        draws) + ' Total number of Games: ' + str(number))


def main():
    start = timer()
    play_many_games(10000, 5)
    end = timer()
    print('In Seconds: ' + str(end - start))
    # if game_ended == DRAW:
    #     print('Game ended in a Draw!')
    # elif game_ended == PLAYER_X:
    #     print('Player X won the game')
    # elif game_ended == PLAYER_O:
    #     print('Player O won the game')
    # else:
    #     print('Shit went wrong!!!!')


if __name__ == '__main__':
    main()
