import math
import random
from abc import ABC, abstractmethod

import numpy as np
from graphviz import Digraph

from tictactoe_game import stringify_field_value
from definitions import DRAW, GAME_STILL_RUNNING


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


class GoodBot(Bot):
    def select_action(self, game):
        me = game.current_player
        he = -1 * me
        avail_a = game.avail_actions
        rand_action = game.avail_actions[math.floor(random.random() * len(game.avail_actions))]
        if game.turn == 1:
            return 4
        elif game.turn == 2:
            if 2 in avail_a:
                return 2
            elif 5 in avail_a:
                return 5
            elif 3 in avail_a:
                return 3
            elif 7 in avail_a:
                return 7
            else:
                return rand_action
        else:
            return rand_action


class MCTSBot(Bot):

    def __init__(self, iterations=1000, exploration=0.3):
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
        while cloned_game.winner == GAME_STILL_RUNNING:
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
    g = Digraph('G', filename='plots/mcts_tree.gv')
    g.node(list_to_str(mcts_node.game.state))
    add_children_to_tree(0, g, mcts_node, list_to_str(mcts_node.game.state))

    g.view()


def add_children_to_tree(id, tree, node, node_name):
    for child in node.children:
        id += 1
        wins = ''
        if child.game.winner != GAME_STILL_RUNNING:
            wins = 'W' + stringify_winner(child.game.winner)
        child_name = 'ID' + str(id) + wins + '\n' + list_to_str(child.game.state)  #
        tree.node(child_name)
        edge_label = 'A: ' + str(child.incoming_action) + 'W: ' + str(child.wins) + ' V: ' + str(
            child.visit_count) + ' D: ' + str(
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
    if number == GAME_STILL_RUNNING:
        return ''
    elif number == DRAW:
        return 'D'
    elif number == 1:
        return 'X'
    elif number == -1:
        return 'O'
