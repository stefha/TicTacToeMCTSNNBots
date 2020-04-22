import unittest

from games import tictactoe
from definitions import ACTIONS, WINNERS
from nn.nn_model_and_training import load_model
import numpy as np


class TestNN(unittest.TestCase):

    def setUp(self) -> None:
        self.ttt = tictactoe.TicTacToe(3)

        self.model_actions = load_model(15, ACTIONS)
        self.model_winners = load_model(15, WINNERS)

    def test_start_action(self):
        self.nn_input = np.asarray([np.asarray(self.ttt.state)])
        predictions = self.model_actions.predict(self.nn_input)
        pred = np.argmax(predictions)
        print(pred)
        self.assertTrue(pred == 4)
