#%%
from loguru import logger
import numpy as np
#%%
class GameState:
    
    def __init__(self, board, current_player):  
        self.board = board
        self.current_player = current_player    # the current_player not yet in action
        self.is_over = False
        self.winner = 0     # 0: no winner, 1: player one wins, -1: player two wins

class ConnectGame:  
            
    def __init__(self, size, connect_num=4):  
        assert connect_num <= size, 'Connect number must be smaller than the smaller dimension'
        self.size = size
        self.connect_num = connect_num
        self.symbols = {1: 'X', -1: 'O', 0: '_'}    # 'X' first player, 'O' second player
        self.action_shape = (size, size)
        self.name = 'connect_game'
    
    @property
    def board_size(self):
        return self.size, self.size
    
    def action2position(self, action):
        """Convert action number to position"""
        return np.array((action // self.size, action % self.size))
    
    def reset(self):  
        board = np.zeros(self.board_size, dtype=int)      # 0: empty
        return GameState(board, 1)  # 1: first player 'X'

    def make_move(self, state: GameState, move):  
        """
        Args:
            state: the current state
            move: the move to be made, a tuple (row, col)
        """
        # assert state.board[move[0], move[1]] == 0, 'Invalid move'
        if state.board[move[0], move[1]] != 0:
            logger.error(f"Invalid move:{move}\n Possible moves: \n{self.get_possible_moves(state).T}")
            raise Exception('Invalid move')
        new_state = GameState(state.board.copy(), state.current_player)
        new_state.board[move[0], move[1]] = new_state.current_player
        if self._check_win(new_state, move):
            new_state.is_over = True
            new_state.winner = state.current_player
        if np.all(new_state.board != 0):
            new_state.is_over = True
            new_state.winner = 0
        new_state.current_player = -state.current_player    # change player
        return new_state

    def _check_win(self, state: GameState, move):  
        row, col = move  
        symbol = state.board[row, col]  
        board_shape = state.board.shape  

        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:  
            count = 1  
            for direction in [-1, 1]:  
                x, y = row + direction * dx, col + direction * dy  
                while 0 <= x < board_shape[0] and 0 <= y < board_shape[1] and state.board[x, y] == symbol:  
                    count += 1  
                    x += direction * dx  
                    y += direction * dy  
                    if count >= self.connect_num:  
                        return True  
        return False  

    def enhance_data(self, batch_data, skip_front=0):
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

    def get_legal_moves(self, state:GameState) -> np.ndarray:
        """
        Returns:
            (np.ndarray): shape is (board_size, board_size)
        """
        return state.board == 0

    def get_possible_moves(self, state:GameState) -> np.ndarray:
        """
        Returns:
            (np.ndarray): shape is (n, 2), where n < board_size * board_size
        """
        # TODO consider symmetry of the board to reduce search space
        return np.concatenate([np.where(state.board == 0)], axis=1).T
    
    def get_act_prob(self, moves, move_prob):
        act_prob = np.zeros(self.board_size, dtype=np.float32)
        act_prob [moves[:, 0], moves[:, 1]] = move_prob    # shape: (board_rows, board_cols)
        return act_prob

    def get_reward(self, state:GameState) -> int:
        # return state.winner * (self.get_possible_moves(state).shape[0] + 1)
        return state.winner
        # return 1 if state.winner == 1 else -1   # avoid draw
    
    def get_turn(self, state:GameState) -> str:
        return self.symbols[state.current_player]
    
    @property
    def first_player(self) -> str:
        return self.symbols[1]

    @property
    def second_player(self) -> str:
        return self.symbols[-1]

    def is_over(self, state:GameState):
        return state.is_over
    
    def print_winner(self, state:GameState):
        if not self.is_over(state):
            print('Game not over')
            return
        if state.winner == 0:
            print('Draw')
        else:
            print(f'Player "{self.symbols[state.winner]}" wins')
    
    def print_state(self, state:GameState):
        rows, cols = self.board_size
        print(' ', '_' * (cols*2+1))
        for i in range(rows):
            print(f'{i}|', ' '.join(self.symbols[state.board[i, j]] for j in range(cols)), '|')
        print('  ', ' '.join(str(i) for i in range(cols)))
    
    def print_turn(self, state:GameState):
        print(f'Player "{self.symbols[state.current_player]}" turn')
    
    def input_move(self, state:GameState):
        str_input = input("Enter your move (row col): ").split()
        move = np.array((int(str_input[0]), int(str_input[1])))
        return move
    
    def __str__(self):
        return self.name + f'_{self.size}x{self.size}_{self.connect_num}'

class ConnectFour(ConnectGame):
    def __init__(self):
        super().__init__(9, 4)
        self.action_shape = (1, self.size)
        self.name = 'connect_four'

    def make_move(self, state: GameState, col):
        """Move is a column number(int)"""
        # col = int(col)
        arr = state.board[:, col]
        row = (arr == 0)[::-1].argmax()  
        row = len(arr) - 1 - row
        move = np.array((row, col))
        return super().make_move(state, move)
    
    def enhance_data(self, batch_data, skip_front=0):
        """Based on the vertical symmetries 
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
            state_t = state[:, :, ::-1]     # vertical symmetry
            act_prob_t = act_prob[::-1]
            enhanced_data.append((state_t, act_prob_t, reward))
        return enhanced_data
    
    # def make_move(self, state: GameState, col):
    #     """Move is a column number(int)"""
    #     arr = state.board[:,col]
    #     row = (arr == 0)[::-1].argmax()  
    #     row = len(arr) - 1 - row
    #     move = np.array((row, col))
    #     return super().make_move(state, move)
    
    def get_legal_moves(self, state):
        return np.any(state.board == 0, axis=0)
    
    def get_possible_moves(self, state):
        return np.where(np.any(state.board == 0, axis=0))[0]
    
    def get_act_prob(self, moves, move_prob):
        act_prob = np.zeros(self.size, dtype=np.float32)
        act_prob[moves] = move_prob
        return act_prob

    def input_move(self, state:GameState):
        str_input = input("Enter your move (col): ")
        move = int(str_input)
        return move

class Gomoku(ConnectGame):
    def __init__(self):
        super().__init__(15, 5)
        self.name = 'gomoku'

class TicTacToe(ConnectGame):
    def __init__(self):
        super().__init__(3, 3)
        self.name = 'tic_tac_toe'

if __name__ == '__main__':
    pass
    #%%
    game = ConnectFour()
    state = game.reset()
    game.print_state(state)
    while not game.is_over(state):
        game.print_turn(state)
        move = game.input_move(state)
        state = game.make_move(state, move)
        game.print_state(state)
    print(f'Player {game.symbols[state.winner]} wins')
