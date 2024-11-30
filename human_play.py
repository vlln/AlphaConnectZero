from datetime import datetime
import numpy as np
from search_tree import SearchTree
from connect_game import ConnectGame
from mcts import MCTS
from minimax import Minimax
from alpha_connect_zero import AlphaConnectZero
from search_tree import play_for_win_rate

if __name__ == "__main__":  
    # game = ConnectGame(9, 4)
    game = ConnectGame(3, 3)

    verbose = False
    debug = True
    # tree = AlphaConnectZero(game, verbose=verbose, debug=debug)
    
    tree = MCTS(game, iterations=5000, verbose=verbose, debug=debug)
    # tree = Minimax(game, verbose)
    # tree.play('O')
    # tree.play('X')

    start_time = datetime.now()
    win_rate = play_for_win_rate(game, tree, Minimax(game, verbose), 2)
    # win_rate = play_for_win_rate(game, Minimax(game, verbose), tree, 2)
    print(f"Win rate: {win_rate}")
    # tree.self_play(10)
    # tree.paralle_self_play(10)
    print(f"Time taken: {datetime.now() - start_time}")
    