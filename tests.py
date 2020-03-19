import unittest

import TicTacToeGame


class TestCheckWinner(unittest.TestCase):

    def setUp(self) -> None:
        self.ttt = TicTacToeGame.TicTacToe(3)

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
        self.ttt = TicTacToeGame.TicTacToe(3)
        self.mcts_bot = TicTacToeGame.MCTSBot(1000)

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
        action = 0
        for count in range(10):
            action += self.mcts_bot.select_action(self.ttt)
        self.assertTrue(action == 20)

    def test_simple_win_player_2_row(self):
        self.ttt.play_action(5)
        self.ttt.play_action(0)
        self.ttt.play_action(1)
        self.ttt.play_action(3)
        self.ttt.play_action(7)
        action = 0
        for count in range(10):
            action += self.mcts_bot.select_action(self.ttt)
        self.assertTrue(action == 60)

    def test_simple_win_player_2_diagonal(self):
        self.ttt.play_action(6)
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(4)
        self.ttt.play_action(2)
        action = 0
        for count in range(10):
            action += self.mcts_bot.select_action(self.ttt)
        self.assertTrue(action == 80)

    def test_tree_plotting(self):
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(1)
        self.ttt.play_action(8)
        action = self.mcts_bot.select_action(self.ttt)
        TicTacToeGame.print_mcts_tree(self.mcts_bot.root)

    def test_tree_plotting_final_three_options(self):
        self.ttt.play_action(0)
        self.ttt.play_action(3)
        self.ttt.play_action(1)
        self.ttt.play_action(8)
        self.ttt.play_action(7)
        self.ttt.play_action(6)
        action = self.mcts_bot.select_action(self.ttt)
        TicTacToeGame.print_mcts_tree(self.mcts_bot.root)

    def test_tree_plotting_final_two_options(self):
        self.ttt.play_action(0)
        self.ttt.play_action(2)
        self.ttt.play_action(1)
        self.ttt.play_action(4)
        self.ttt.play_action(3)
        self.ttt.play_action(8)
        self.ttt.play_action(5)
        action = self.mcts_bot.select_action(self.ttt)
        TicTacToeGame.print_mcts_tree(self.mcts_bot.root)

if __name__ == '__main__':
    # suite = unittest.TestSuite()
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    # suite.addTest(MyTestCase())
    unittest.main()
