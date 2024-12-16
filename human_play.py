from datetime import datetime
import numpy as np
from search_tree import SearchTree
from connect_game import ConnectGame, ConnectFour, Gomoku, TicTacToe
from mcts import MCTS
from minimax import Minimax
from alpha_connect_zero import AlphaConnectZero
from search_tree import play_for_win_rate
from loguru import logger

def print_data(data):
    for i in range(3):
        print(data[i])
if __name__ == "__main__":  
    logger.level("INFO")
    np.set_printoptions(precision=3, suppress=True)

    game = ConnectFour()
    # game = ConnectGame(9, 4)

    # tree = AlphaConnectZero(game, iterations=40)
    
    tree = MCTS(game, iterations=40)
    # tree = Minimax(game)
    tree.play('O')
    tree.play('X')
    start_time = datetime.now()

    # win_rate = play_for_win_rate(game, tree, tree, 2)
    # win_rate = play_for_win_rate(game, tree, Minimax(game), 2)
    # zero_tree = AlphaConnectZero(game, iterations=5)
    # win_rate = play_for_win_rate(game, zero_tree, tree, 2)
    # print(f"Win rate: {win_rate}")
    data = tree.self_play(2, enhance=True)
    # tree.paralle_self_play(10)

    print(f"Time taken: {datetime.now() - start_time}")
    