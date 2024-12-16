import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from abc import abstractmethod

import numpy as np
from loguru import logger

from connect_game import ConnectGame, GameState

class TreeNode:  
    def __init__(self, state, move, parent=None):  
        self.state = state
        self.move = move    # wichch move led to this state
        self.parent = parent  
        self.children = []  
        self.score = 0
        self.visits = 0
        self.p = 1     # probability
    
    def expand(self, state, move):
        child_class = type(self) 
        child = child_class(state, move, parent=self) 
        self.children.append(child)
        return child

    def is_leaf(self):
        return len(self.children) == 0

class SearchTree:

    def __init__(self, game: ConnectGame, *args, **kwargs):
        """
        Args:
            game: game object
        """
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
    
    def tree2str(self, node, deepth=1):  
        if node is None or deepth == 0:  
            return ""  
        return self._tree2str_dfs(node, deepth)  

    def _tree2str_dfs(self, node, deepth, indent_num=0):  
        output = f"{indent_num * '--'}Move: {node.move} visits: {node.visits} score: {node.score}\n"  
        deepth -= 1  
        if deepth != 0:  
            for child in node.children:  
                output += self._tree2str_dfs(child, deepth, indent_num+1)  
        return output
    
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
            start_time = time.time()
            best_move, act_prob = self.search(state, maxmize)
            logger.info(f"Search time: {time.time() - start_time}")
            state = self.game.make_move(state, best_move)

        while not self.game.is_over(state):
            # human player
            self.game.print_state(state)
            self.game.print_turn(state)
            move = self.game.input_move(state)
            state = self.game.make_move(state, move)
            print("-" * 30)
            if self.game.is_over(state):
                break

            # computer player
            best_move, act_prob = self.search(state, maxmize)
            print(self.tree2str(self.root))
            state = self.game.make_move(state, best_move)
            print(f"Computer move: {best_move} Action prob: \n{act_prob}")

        self.game.print_state(state)
        self.game.print_winner(state)

    def self_play(self, total_games=1000, enhance=False, noise_steps=2, noise_alpha=0.03, temperature=0.5):
        start_time = time.time()
        maxmize = True
        collected_data = []
        for n in range(total_games):
            logger.trace(f"Self-Play game begins. Game {n}/{total_games}")
            states, act_probs, current_players = [], [], []
            state = self.game.reset()
            current_step = 0
            while not self.game.is_over(state):
                move, probs = self.search(state, maxmize)
                if current_step < noise_steps:
                    current_alpha = noise_alpha * (1 - current_step / noise_steps)  
                    dirichlet_noise = np.random.dirichlet([current_alpha] * len(probs))  
                    probs = (1 - current_alpha) * probs + current_alpha * dirichlet_noise  
                    # normalize
                    # temperature
                    probs = np.log(probs) / temperature
                    exp_probs = np.exp(probs- np.max(probs))
                    probs = exp_probs / np.sum(exp_probs)
                    temperature = max(0.01, temperature * 0.99)
                    move = np.random.choice(self.game.get_possible_moves(state), p=probs)
                states.append(state.board)
                act_probs.append(probs)
                current_players.append(state.current_player)
                # turn
                state = self.game.make_move(state, move)
                maxmize = not maxmize
                current_step += 1
            reward = self.game.get_reward(state)
            # transform data
            game_stack = self._transform_data(states, act_probs, current_players, reward)
            collected_data.extend(game_stack)
            # enhance data
            if enhance:
                game_stack = self.game.enhance_data(game_stack)
                collected_data.extend(game_stack)
        logger.trace(f"Self-play finished in {time.time() - start_time:.2f}s.")
        return collected_data

    def paralle_self_play(self, total_games=1000, enhance=False, processes=10):
        """Parallel self-play using multiprocessing on CPU"""
        games_per_process = total_games // processes
        if total_games < processes:
            processes = total_games
            games_per_process = 1
        results = []
        with ProcessPoolExecutor(max_workers=processes) as executor:  
            futures = {executor.submit(self.self_play, games_per_process, enhance=enhance): i for i in range(processes)}  
            for future in as_completed(futures):  
                game_results = future.result()
                results.extend(game_results)
        return results

    def _transform_data(self, states, act_probs, current_players, reward):
        """Transform data to be used for training
           All data are converted to First Player perspective
        """
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
            else:
                move, _ = oponent.search(state, maxmize=first_play)
            first_play = not first_play
            state = game.make_move(state, move)
        if game.get_reward(state) > 0:
            win_count += 1
    return win_count / total_games


    