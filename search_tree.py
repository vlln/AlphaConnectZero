from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from abc import abstractmethod
import numpy as np
from connect_game import ConnectGame, GameState

class TreeNode:  
    def __init__(self, state, move, parent=None):  
        self.state = state
        self.move = move  # wichch move led to this state
        self.parent = parent  
        self.children = []  
        self.score = 0
        self.visits = 0
    
    def expand(self, state, move):
        child_class = type(self) 
        child = child_class(state, move, parent=self) 
        self.children.append(child)
        return child

    def is_leaf(self):
        return len(self.children) == 0

class SearchTree:

    def __init__(self, game: ConnectGame, verbose=False, debug=False):
        """
        Args:
            game: game object
        """
        self.verbose = verbose
        self.debug = debug
        self.game = game
        self.root = None
    
    @abstractmethod
    def search(self, state: GameState, maxmize=True):
        """
        Args:
            state: current game state
            maxmize: if True, search for the best move for the current player, otherwise search for the best move for the opponent
        Returns:
            best_move (np.ndarray): the best move for the current player.  If the game is over, return None. shape: (2,)
            act_prob (np.ndarray): the probability of taking the best move. shape: (n, n)
        """
        self.act_player = 1 if maxmize else -1
        self.root = TreeNode(state, None)  
        ...
    
    def print_tree(self, node, deepth=1):
        if node is None or deepth == 0:
            return
        self._print_tree_dfs(node, deepth)
    
    def _print_tree_dfs(self, node, deepth, indent_num=0):
        print(f"{indent_num * '--'}Move: {node.move} visits: {node.visits} score: {node.score}")
        deepth -= 1
        if deepth != 0:
            for child in node.children:
                self._print_tree_dfs(child, deepth, indent_num+1)
    
    def play(self, act_player='X'):
        """
        Args:
            ai_player: 'X' or 'O', if not specified, play as first player ('X')
        """
        
        np.set_printoptions(precision=3, suppress=True)
        state = self.game.reset()
        maxmize = False
        if act_player == 'X':
            maxmize = True
            best_move, act_prob = self.search(state, maxmize)
            state = self.game.make_move(state, best_move)

        while not self.game.is_over(state):
            # human player
            self.game.print_state(state)
            self.game.print_turn(state)
            move = input("Enter your move (row col): ").split()
            move = (int(move[0]), int(move[1]))
            state = self.game.make_move(state, move)
            print("-" * 30)
            if self.game.is_over(state):
                break

            # computer player
            best_move, act_prob = self.search(state, maxmize)
            if self.debug:
                self.print_tree(self.root)
            state = self.game.make_move(state, best_move)
            print(f"Computer move: {best_move} Action prob: \n{act_prob}")

        self.game.print_state(state)
        self.game.print_winner(state)

    def self_play(self, total_games=1000, enhance=False):
        maxmize = True
        collected_data = []
        for n in range(total_games):
            print(f"Game {n+1}, time: {datetime.now()}")
            states, act_probs, current_players = [], [], []
            state = self.game.reset()
            while not self.game.is_over(state):
                move, probs = self.search(state, maxmize)
                state = self.game.make_move(state, move)
                maxmize = not maxmize
                states.append(state.board)
                act_probs.append(probs)
                current_players.append(state.current_player)
            reward = self.game.get_reward(state)
            # transform data
            game_stack = self._transform_data(states, act_probs, current_players, reward)
            collected_data.extend(game_stack)
            # enhance data
            if enhance:
                game_stack = self._enhance_data(game_stack, skip_front=1)
                collected_data.extend(game_stack)
        return collected_data

    def paralle_self_play(self, total_games=1000, enhance=False, processes=10):
        games_per_process = total_games // processes
        results = []
        with ProcessPoolExecutor(max_workers=processes) as executor:  
            futures = {executor.submit(self.self_play, games_per_process, enhance=True): i for i in range(processes)}  
            for future in as_completed(futures):  
                game_results = future.result()
                results.extend(game_results)
        return results

    def _enhance_data(self, batch_data, skip_front=0):
        """Based on the 8 symmetries of the game board that is a square
        Args:
            batch_data (list): list of tuples (state, act_prob, reward)
            skip_front (int): number of states to skip from the front of the batch
        Returns:
            enhanced_data (list): list of tuples (state, act_prob, reward).
                Does not include the original batch_data
        """
        enhanced_data = []
        for n in range(skip_front, len(batch_data)):
            state, act_prob, reward = batch_data[n]
            state_t = np.transpose(state, axes=(0, 2, 1))
            act_prob_t = np.transpose(act_prob, axes=(1, 0))
            enhanced_data.append((state_t, act_prob_t, reward))
            for r in range(3):
                state = np.rot90(state, axes=(2, 1))
                act_prob = np.rot90(act_prob)
                enhanced_data.append((state, act_prob, reward))
                state = np.transpose(state, axes=(0, 2, 1))
                act_prob = np.transpose(act_prob, axes=(1, 0))
                enhanced_data.append((state, act_prob, reward))
        return enhanced_data

    def _transform_data(self, states, act_probs, current_players, reward):
        players = np.array(current_players)[:, None]   # (n, 1)
        states = np.array(states)[:, None]   # (n, c, h, w)
        states = states * players[:, :, None, None]     # convert to same player
        act_probs = np.array(act_probs)     # (h, w)
        rewards = players * reward          # (n, 1)
        stack = list(zip(states, act_probs, rewards))   # [((c, h, w), (h, w), (1,)), ...] 
        return stack

def play_for_win_rate(game: ConnectGame, player: SearchTree, oponent: SearchTree, total_games=10):
    """
    Returns:
        win_rate (float): percentage of games won by player
    """
    first_play = True
    win_count = 0
    for n in range(total_games):
        state = game.reset()
        while not game.is_over(state):
            if first_play:
                move, _ = player.search(state, maxmize=first_play)
                print(f"{player.__class__} move: {move}")
            else:
                move, _ = oponent.search(state, maxmize=first_play)
                print(f"{oponent.__class__} move: {move}")
            first_play = not first_play
            state = game.make_move(state, move)
            # game.print_state(state)
        if game.get_reward(state) > 0:
            win_count += 1
    return win_count / total_games


    