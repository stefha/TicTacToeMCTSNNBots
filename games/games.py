from abc import ABC, abstractmethod


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
