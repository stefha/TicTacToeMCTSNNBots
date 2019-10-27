import math
import random
import numpy as np
from timeit import default_timer as timer
from abc import ABC, abstractmethod

EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
DRAW = 100000
PLAYERS = [PLAYER_X, PLAYER_O]


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

    # @abstractmethod
    # def available_actions(self):
    #     pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def play_action(self, action):
        pass

    @abstractmethod
    def check_winner(self):
        pass


class TicTacToe(Game):
    def __init__(self, size, state=None, turn=0, avail_actions=None, winner=EMPTY, current_player=None):
        self.size = size
        self.turn = turn
        self.winner = winner

        if state is None:
            self.state = np.full((size * size), EMPTY)
        else:
            self.state = state

        if avail_actions is None:
            self.avail_actions = list(range(self.size * self.size))
        else:
            self.avail_actions = avail_actions
        if current_player is None:
            self.current_player = PLAYERS[self.turn % 2]
        else:
            self.current_player = current_player

    def clone(self):
        new_game = TicTacToe(self.size, np.copy(self.state), self.turn, np.copy(self.avail_actions), self.winner,
                             self.current_player)

        return new_game

    #
    # def available_actions(self):
    #     self.avail_actions = np.where(self.state == EMPTY)
    #     return self.avail_actions  # or actions[0]?

    def play_action(self, action):
        self.state[action] = self.current_player
        self.turn += 1
        self.current_player = PLAYERS[self.turn % 2]
        self.avail_actions = np.delete(self.avail_actions, np.argwhere(self.avail_actions == action))
        # self.avail_actions.remove(action)
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
                    self.winner = player
                    return player
        # rows
        for row in range(self.size):
            sum_row = 0
            for line in range(self.size):
                position = line * self.size + row
                sum_row += self.state[position]
            for player in ([PLAYER_X, PLAYER_O]):
                if sum_row == player * self.size:
                    self.winner = player
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
                self.winner = player
                return player

        if self.turn == self.size * self.size:
            self.winner = DRAW
            return DRAW
        else:
            self.winner = EMPTY
            return EMPTY


class Bot(ABC):
    @abstractmethod
    def select_action(self, game):
        pass


class RandBot(Bot):

    def select_action(self, game):
        return game.avail_actions[math.floor(random.random() * len(game.avail_actions))]


class InOrderBot(Bot):

    def select_action(self, game):
        return game.avail_actions[0]


class BackwardsInOrderBot(Bot):
    def select_action(self, game):
        return game.avail_actions[game.avail_actions.size - 1]


class MCTSBot(Bot):

    def __init__(self, iterations=100, exploration=100):
        self.iterations = iterations
        self.exploration = exploration
        self.root = None

    def select_action(self, game):
        self.root = MCTSNode(game, None, None)
        node = self.root
        for iteration in range(self.iterations):
            node = self.traverse_tree(node)
            action = self.select_next_action_extension(node)
            game_clone = node.game.clone()
            game_clone.play_action(action)
            child = MCTSNode(game_clone, action, node)
            node.children.append(child)
            sim_result = self.do_simulation(node)
            self.backpropagate_simulation_result(sim_result)
        best_child = self.select_child_highest_visit_count(self.root)
        return best_child.incoming_action

    def select_child_highest_visit_count(self, root):
        visit_counts = [child.visit_count for child in root.children]
        return root.children[np.argmax(visit_counts)]

    def traverse_tree(self, node):
        if len(node.game.avail_actions) > len(node.children):
            return node
        elif len(node.game.avail_actions) == len(node.children):
            return self.select_best_child_for_traversal(node)
        else:
            print('FUUUUCK Error')
            return None

    def select_best_child_for_traversal(self, node):
        uct_value_array = [self.calculate_uct_value(child) for child in node.children]
        return node.children[np.argmax(uct_value_array)]

    def select_next_action_extension(self, node):
        used_actions = [child.incoming_action for child in node.children]
        interesting_actions = [action for action in node.game.avail_actions if action not in used_actions]
        if len(interesting_actions) == 0:
            print('Deal with 0 avail actions!!!!')
        # Use Extension here ?
        return interesting_actions[math.floor(random.random() * len(interesting_actions))]

    def calculate_uct_value(self, node):
        domain_value = node.wins / node.visit_count
        exploration_value = self.exploration * (
            math.sqrt((2 * math.log(node.parent.visit_count)) / node.visit_count))
        return domain_value + exploration_value


class MCTSNode:
    def __init__(self, game, incoming_action, parent):
        self.game = game
        self.children = []
        self.incoming_action = incoming_action
        self.wins = 0
        self.visit_count = 0
        self.parent = parent


def play_game_till_end(size, bot_x, bot_o):
    ttt = TicTacToe(size)
    game_ended = EMPTY
    player = PLAYER_X
    bots = [bot_x, bot_o]
    while game_ended == EMPTY:
        action = bots[ttt.turn % 2].select_action(ttt)
        game_ended = ttt.play_action(player, action)
        player = player * -1
    return game_ended


def play_many_games(number, size, bot_x=RandBot(), bot_y=RandBot()):
    total_result = 0
    wins_player_x = 0
    wins_player_o = 0
    draws = 0
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


def timeIt(function, args_list):
    start = timer()
    function(*args_list)
    end = timer()
    print('In Seconds: ' + str(end - start))


def main():
    timeIt(play_many_games, [100, 3, MCTSBot(), RandBot()])
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
