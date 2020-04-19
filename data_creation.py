import os

import tictactoe_game
from bots import MCTSBot, RandBot
from definitions import PLAYER_X, PLAYER_O, DRAW, PLAYERS, DEBUG_PRINT, GAME_STILL_RUNNING
from timeit import default_timer as timer
import numpy as np
import pandas as pd


def produce_data_actions(number_of_games, bot):
    # state_list = list()
    state_list0 = list()
    state_list1 = list()
    state_list2 = list()
    state_list3 = list()
    state_list4 = list()
    state_list5 = list()
    state_list6 = list()
    state_list7 = list()
    state_list8 = list()
    action_list = list()
    winner_list = list()

    for count in range(number_of_games):
        if count % 100 == 0:
            print(str(count))
        game = tictactoe_game.TicTacToe(3)
        winner = GAME_STILL_RUNNING
        temp_winner_list = list()
        while winner == GAME_STILL_RUNNING:
            action = bot.select_action(game)
            state = np.copy(game.state).tolist()
            perspective_indicator = 1
            if game.turn % 2 == 1:
                state = np.multiply(state, -1)  # always show the game state form perspective of player 1
                perspective_indicator = perspective_indicator * -1
            #           state_list.append(state)
            state_list0.append(state[0])
            state_list1.append(state[1])
            state_list2.append(state[2])
            state_list3.append(state[3])
            state_list4.append(state[4])
            state_list5.append(state[5])
            state_list6.append(state[6])
            state_list7.append(state[7])
            state_list8.append(state[8])
            action_list.append(action)
            temp_winner_list.append(perspective_indicator)
            winner = game.play_action(action)

        temp_winner_list = [perspective_indicator * winner for perspective_indicator in temp_winner_list]
        winner_list.extend(temp_winner_list)

    next_data_directory = find_and_create_next_data_directory_name()

    df_states = pd.DataFrame(
        {"state0": state_list0, "state1": state_list1, "state2": state_list2, "state3": state_list3,
         "state4": state_list4, "state5": state_list5, "state6": state_list6, "state7": state_list7,
         "state8": state_list8})
    df_actions = pd.DataFrame({"action": action_list})
    df_winners = pd.DataFrame({"winner": winner_list})

    total_data_list = list(
        zip(action_list, winner_list, state_list0, state_list1, state_list2, state_list3, state_list4, state_list5,
            state_list6, state_list7, state_list8))
    df_all_data = pd.DataFrame(total_data_list,
                               columns=['Action', 'Winner', 'State0', 'State1', 'State2', 'State3', 'State4', 'State5',
                                        'State6', 'State7', 'State8'])

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
    game_ended = GAME_STILL_RUNNING
    player = PLAYER_X
    bots = [bot_x, bot_o]
    while game_ended == GAME_STILL_RUNNING:
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
    timeIt(produce_data_actions, [10000, MCTSBot(iterations=2000)])
