import math
import random
import numpy as np
from timeit import default_timer as timer
from abc import ABC, abstractmethod
from graphviz import Digraph

from prettytable import PrettyTable

EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
DRAW = 10
PLAYERS = [PLAYER_X, PLAYER_O]

DEBUG_PRINT = True


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

    def __init__(self, size, state=None, turn=0, avail_actions=None, winner=EMPTY, current_player=None, sum_lines=None,
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
        return self.winner != EMPTY

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
            return EMPTY

    def check_winner(self):
        # i think both can be done in one , check with tests!!!1
        #
        # # lines
        # for line in range(self.size):
        #     sum_line = 0
        #     for row in range(self.size):
        #         position = line * self.size + row
        #         sum_line += self.state[position]
        #     for player in ([PLAYER_X, PLAYER_O]):
        #         if sum_line == player * self.size:
        #             self.winner = player
        #             return player
        # # rows
        # for row in range(self.size):
        #     sum_row = 0
        #     for line in range(self.size):
        #         position = line * self.size + row
        #         sum_row += self.state[position]
        #     for player in ([PLAYER_X, PLAYER_O]):
        #         if sum_row == player * self.size:
        #             self.winner = player
        #             return player
        #
        # # diagonal1
        # sum_diag_1 = 0
        # sum_diag_2 = 0
        # position_diag_1 = 0
        # position_diag_2 = 0
        # for index in range(self.size):
        #     sum_diag_1 += self.state[position_diag_1]
        #     sum_diag_2 += self.state[position_diag_2]
        #     position_diag_1 += self.size + 1
        #     position_diag_2 += self.size - 1
        # for player in ([PLAYER_X, PLAYER_O]):
        #     if sum_diag_1 == player * self.size or sum_diag_2 == player * self.size:
        #         self.winner = player
        #         return player

        for player in ([PLAYER_X, PLAYER_O]):
            sum_player = player * self.size
            if sum_player in self.sum_lines or sum_player in self.sum_rows or sum_player in self.sum_diagonals:
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

    def __init__(self, iterations=1000, exploration=0.5):
        self.iterations = iterations
        self.exploration = exploration
        self.root = None

    def select_action(self, game):
        self.root = MCTSNode(game, None, None)
        for iteration in range(self.iterations):
            node = self.traverse_tree(self.root)
            sim_winner = self.do_simulation(node)
            self.backpropagate_simulation_result(node, sim_winner)
        best_child = self.select_child_highest_visit_count(self.root)
        return best_child.incoming_action

    def select_child_highest_visit_count(self, root):
        visit_counts = [child.visit_count for child in root.children]
        return root.children[np.argmax(visit_counts)]

    def traverse_tree(self, node):
        if node.game.is_finished():
            return node
        elif node.is_fully_extended():
            return self.traverse_tree(self.select_best_child_for_traversal(node))
        else:
            action = self.select_next_action_for_extension(node)
            game_clone = node.game.clone()
            game_clone.play_action(action)
            child = MCTSNode(game_clone, action, node)
            node.children.append(child)
            return child

    def select_best_child_for_traversal(self, node):
        uct_value_array = [self.calculate_uct_value(child) for child in node.children]
        return node.children[np.argmax(uct_value_array)]

    def select_next_action_for_extension(self, node):
        used_actions = [child.incoming_action for child in node.children]
        interesting_actions = [action for action in node.game.avail_actions if action not in used_actions]
        if len(interesting_actions) == 0:
            raise Exception('No Available Actions For Extension')
        rand_action = interesting_actions[math.floor(random.random() * len(interesting_actions))]
        return rand_action

    def calculate_uct_value(self, node):
        domain_value = node.wins / node.visit_count
        exploration_value = self.exploration * (
            math.sqrt((2 * math.log(node.parent.visit_count)) / node.visit_count))
        node.exploration_value = exploration_value
        node.domain_value = domain_value
        return domain_value + exploration_value

    def do_simulation(self, node):
        cloned_game = node.game.clone()
        while cloned_game.winner == EMPTY:
            # Start with rand simulations
            rand_action = cloned_game.avail_actions[math.floor(len(cloned_game.avail_actions) * random.random())]
            cloned_game.play_action(rand_action)
        return cloned_game.winner

    def backpropagate_simulation_result(self, node, sim_winner):
        while node != self.root:
            node.visit_count += 1
            if sim_winner == node.parent.game.current_player:
                node.wins += 1
            node = node.parent
        # Increment root visit count
        node.visit_count += 1


class MCTSNode:
    def __init__(self, game, incoming_action, parent):
        self.game = game
        self.children = []
        self.incoming_action = incoming_action
        self.wins = 0
        self.visit_count = 0
        self.parent = parent
        self.exploration_value = 0
        self.domain_value = 0
        self.total_value = 0

    def is_fully_extended(self):
        return len(self.children) == len(self.game.avail_actions)


def print_mcts_tree(mcts_node):
    g = Digraph('G', filename='hello.gv')
    g.node(list_to_str(mcts_node.game.state))
    add_children_to_tree(0, g, mcts_node, list_to_str(mcts_node.game.state))

    g.view()


def add_children_to_tree(id, tree, node, node_name):
    for child in node.children:
        id += 1
        wins = ''
        if child.game.winner != EMPTY:
            wins = 'W' + stringify_winner(child.game.winner)
        child_name = 'ID' + str(id) + wins + '\n' + list_to_str(child.game.state)  #
        tree.node(child_name)
        edge_label = 'A: ' + str(child.incoming_action) + ' V: ' + str(child.visit_count) + ' D: ' + str(
            child.domain_value)[0:4] + ' UCT: ' + str(child.domain_value + child.exploration_value)[0:4]
        tree.edge(node_name, child_name, edge_label)
        id = add_children_to_tree(id, tree, child, child_name)
    return id


def list_to_str(game_state):
    state_description = stringify_field_value(game_state[0]) + '|' + stringify_field_value(
        game_state[1]) + '|' + stringify_field_value(
        game_state[2]) + '\n' + stringify_field_value(game_state[3]) + '|' + stringify_field_value(
        game_state[4]) + '|' + stringify_field_value(
        game_state[5]) + '\n' + stringify_field_value(game_state[6]) + '|' + stringify_field_value(
        game_state[7]) + '|' + stringify_field_value(
        game_state[8])

    return state_description


def stringify_winner(number):
    if number == EMPTY:
        return ''
    elif number == DRAW:
        return 'D'
    elif number == 1:
        return 'X'
    elif number == -1:
        return 'O'


def stringify_field_value(number):
    if number == EMPTY:
        return '   '
    elif number == 1:
        return 'X'
    elif number == -1:
        return 'O'


def play_game_till_end(size, bot_x, bot_o):
    ttt = TicTacToe(size)
    game_ended = EMPTY
    player = PLAYER_X
    bots = [bot_x, bot_o]
    while game_ended == EMPTY:
        ttt.print()
        action = bots[ttt.turn % 2].select_action(ttt)
        game_ended = ttt.play_action(action, is_action_in_real_game=True)
        player = player * -1
    ttt.print()
    print('Win: ' + str(ttt.winner))
    if DEBUG_PRINT:
        print(' Rows: ' + str(ttt.sum_rows) + ' Lines: ' + str(ttt.sum_lines) + ' Diags: ' + str(ttt.sum_diagonals))
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
    timeIt(play_many_games, [1, 3, MCTSBot(100, 0.1), MCTSBot(100, 0.1)])


if __name__ == '__main__':
    main()
