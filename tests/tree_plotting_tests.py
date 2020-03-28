import unittest
import numpy as np

import bots
import tictactoe_game


class tree_Plotting_Tests(unittest.TestCase):

    def setUp(self) -> None:
        self.ttt = tictactoe_game.TicTacToe(3)
        self.mcts_bot = bots.MCTSBot(1000)

    def test_full_tree_plotting(self):
        action = self.mcts_bot.select_action(self.ttt)
        child_visit_counts = [child.visit_count for child in self.mcts_bot.root.children]
        best_child = self.mcts_bot.root.children[np.argmax(child_visit_counts)]
        bots.print_mcts_tree(best_child)

    def test_tree_plotting(self):
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(1)
        self.ttt.play_action(8)
        action = self.mcts_bot.select_action(self.ttt)
        bots.print_mcts_tree(self.mcts_bot.root)

    def test_tree_plotting_final_three_options(self):
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(1)
        self.ttt.play_action(8)
        self.ttt.play_action(7)
        self.ttt.play_action(6)
        action = self.mcts_bot.select_action(self.ttt)
        bots.print_mcts_tree(self.mcts_bot.root)

    def test_tree_plotting_final_two_options(self):
        self.ttt.play_action(0)
        self.ttt.play_action(2)
        self.ttt.play_action(1)
        self.ttt.play_action(4)
        self.ttt.play_action(3)
        self.ttt.play_action(8)
        self.ttt.play_action(5)
        action = self.mcts_bot.select_action(self.ttt)
        bots.print_mcts_tree(self.mcts_bot.root)

    def test_tree_plotting_final_2_win_out_of_3_options(self):
        self.ttt.play_action(0)
        self.ttt.play_action(4)
        self.ttt.play_action(2)
        self.ttt.play_action(5)
        self.ttt.play_action(6)
        self.ttt.play_action(8)
        action = self.mcts_bot.select_action(self.ttt)
        bots.print_mcts_tree(self.mcts_bot.root)

    def test_tree_plotting_final_two_win_options_player_2(self):
        self.ttt.play_action(7)
        self.ttt.play_action(0)
        self.ttt.play_action(4)
        self.ttt.play_action(2)
        self.ttt.play_action(5)
        self.ttt.play_action(6)
        self.ttt.play_action(8)
        action = self.mcts_bot.select_action(self.ttt)
        bots.print_mcts_tree(self.mcts_bot.root)


if __name__ == '__main__':
    # suite = unittest.TestSuite()
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    # suite.addTest(MyTestCase())
    unittest.main()
