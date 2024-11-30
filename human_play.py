from datetime import datetime
import numpy as np
from search_tree import SearchTree
from connect_game import ConnectGame
from mcts import MCTS
from minimax import Minimax
from alpha_connect_zero import AlphaConnectZero
from search_tree import play_for_win_rate
from loguru import logger

logger.level("INFO")

if __name__ == "__main__":  
    game = ConnectGame(9, 4)
    # game = ConnectGame(3, 3)

    # tree = AlphaConnectZero(game)
    
    tree = MCTS(game, iterations=1000)
    # tree = Minimax(game, verbose)
    # tree.play('O')
    # tree.play('X')

    start_time = datetime.now()
    # win_rate = play_for_win_rate(game, tree, Minimax(game), 2)
    zero_tree = AlphaConnectZero(game, iterations=5)
    win_rate = play_for_win_rate(game, zero_tree, tree, 2)
    print(f"Win rate: {win_rate}")
    # tree.self_play(10)
    # tree.paralle_self_play(10)
    print(f"Time taken: {datetime.now() - start_time}")
    