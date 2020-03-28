import os

import tictactoe_game
from bots import MCTSBot, RandBot
from definitions import EMPTY, PLAYER_X, PLAYER_O, DRAW, PLAYERS, DEBUG_PRINT
from timeit import default_timer as timer
import numpy as np
import pandas as pd


def produce_data_actions(number_of_games, bot):
    state_list = list()
    action_list = list()
    winner_list = list()
    for count in range(number_of_games):
        game = tictactoe_game.TicTacToe(3)
        winner = EMPTY
        while winner == EMPTY:
            action = bot.select_action(game)
            state = np.copy(game.state)
            state_list.append(state)
            action_list.append(action)
            winner = game.play_action(action)

        winner_list.extend([winner] * game.turn)

    next_data_directory = find_and_create_next_data_directory_name()

    df_states = pd.DataFrame({"state": state_list})
    df_actions = pd.DataFrame({"action": action_list})
    df_winners = pd.DataFrame({"winner": winner_list})

    df_all_data = pd.DataFrame(list(zip(action_list, winner_list, state_list)),
                               columns=['Action', 'Winner', 'State'])

    df_all_data.to_csv(next_data_directory + 'all_data.csv', index=False)
    df_states.to_csv(next_data_directory + 'states.csv', index=False)
    df_actions.to_csv(next_data_directory + 'actions.csv', index=False)
    df_winners.to_csv(next_data_directory + 'winners.csv', index=False)


def find_and_create_next_data_directory_name():
    os.chdir('data')
    data_directory_list = os.listdir()
    next_directory_number = np.max([int(i) for i in data_directory_list]) + 1
    next_data_directory = '../data/' + str(next_directory_number) + '/'
    os.mkdir(next_data_directory)
    return next_data_directory


def play_game_till_end(size, bot_x, bot_o):
    ttt = tictactoe_game.TicTacToe(size)
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


if __name__ == '__main__':
    timeIt(produce_data_actions, [100, MCTSBot()])
