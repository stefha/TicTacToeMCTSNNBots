import unittest
from timeit import timeit

import numpy as np

import bots
import tictactoe_game


class TestCheckWinner(unittest.TestCase):

    def setUp(self) -> None:
        self.ttt = tictactoe_game.TicTacToe(3)

    def test_diag_1_win(self):
        self.ttt.play_action(0)
        self.ttt.play_action(1)
        self.ttt.play_action(4)
        self.ttt.play_action(7)
        self.ttt.play_action(8)

        x_is_winner = self.ttt.winner == 1
        self.assertEqual(x_is_winner, True)

    def test_diag_2_win(self):
        self.ttt.play_action(2)
        self.ttt.play_action(1)
        self.ttt.play_action(4)
        self.ttt.play_action(7)
        self.ttt.play_action(6)

        x_is_winner = self.ttt.winner == 1
        self.assertEqual(x_is_winner, True)

    def test_line_1_win(self):
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(1)
        self.ttt.play_action(4)
        self.ttt.play_action(2)

        x_is_winner = self.ttt.winner == 1
        self.assertEqual(x_is_winner, True)

    def test_row_1_win(self):
        self.ttt.play_action(0)
        self.ttt.play_action(1)
        self.ttt.play_action(3)
        self.ttt.play_action(4)
        self.ttt.play_action(6)

        x_is_winner = self.ttt.winner == 1
        self.assertEqual(x_is_winner, True)

    def test_draw(self):
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(1)
        self.ttt.play_action(4)
        self.ttt.play_action(6)
        self.ttt.play_action(2)
        self.ttt.play_action(5)
        self.ttt.play_action(7)
        self.ttt.play_action(8)

        draw = self.ttt.winner == 10
        self.assertEqual(draw, True)


class TestCheckMCTS(unittest.TestCase):

    def setUp(self) -> None:
        self.ttt = tictactoe_game.TicTacToe(3)
        self.mcts_bot = bots.MCTSBot(1000)

    def test_simple_win_player_1_line(self):
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(1)
        self.ttt.play_action(6)
        action = 0
        for count in range(10):
            action += self.mcts_bot.select_action(self.ttt)
        self.assertTrue(action == 20)

    def test_simple_win_player_1_row(self):
        self.ttt.play_action(0)
        self.ttt.play_action(1)
        self.ttt.play_action(3)
        self.ttt.play_action(7)
        action = 0
        for count in range(10):
            action += self.mcts_bot.select_action(self.ttt)
        self.assertTrue(action == 60)

    def test_simple_win_player_1_diagonal(self):
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(4)
        self.ttt.play_action(2)
        action = 0
        for count in range(10):
            action += self.mcts_bot.select_action(self.ttt)
        self.assertTrue(action == 80)

    def test_simple_win_player_2_line(self):
        self.ttt.play_action(5)
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(1)
        self.ttt.play_action(6)
        only_right_actions = True
        for count in range(10):
            action = self.mcts_bot.select_action(self.ttt)
        if action not in [2, 4]:
            only_right_actions = False
        self.assertTrue(only_right_actions)

    def test_simple_win_player_2_row(self):
        self.ttt.play_action(5)
        self.ttt.play_action(0)
        self.ttt.play_action(1)
        self.ttt.play_action(3)
        self.ttt.play_action(7)
        only_right_actions = True
        for count in range(10):
            action = self.mcts_bot.select_action(self.ttt)
            only_right_actions = only_right_actions and action == 6
        self.assertTrue(only_right_actions)

    def test_simple_win_player_2_diagonal(self):
        self.ttt.play_action(6)
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(4)
        self.ttt.play_action(2)
        only_right_action = True
        for count in range(10):
            action = self.mcts_bot.select_action(self.ttt)
            only_right_action = only_right_action and action == 8
        self.assertTrue(only_right_action)

    def test_simple_not_lose(self):
        self.ttt.play_action(4)
        self.ttt.play_action(3)
        self.ttt.play_action(7)
        self.ttt.play_action(5)
        self.ttt.play_action(6)
        only_right_action = True
        for count in range(10):
            action = self.mcts_bot.select_action(self.ttt)
            only_right_action = only_right_action and (action == 1 or action == 2 or action == 8)
        self.assertTrue(only_right_action)


class TestBots(unittest.TestCase):
    def test_good_bot(self):
        tictactoe_game.play_many_games, [1, 3, bots.GoodBot(), bots.RandBot()]


if __name__ == '__main__':
    # suite = unittest.TestSuite()
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    # suite.addTest(MyTestCase())
    unittest.main()
