
#%%
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
            print(self.get_possible_moves(state))
            print(f"Invalid move: {move}")
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

if __name__ == '__main__':
    pass
    #%%
    game = ConnectGame(3, 3, 3)
    state = game.reset()
    game.print_state(state)
    while not game.is_over(state):
        game.print_turn(state)
        move = input("Enter your move (row col): ").split()
        move = (int(move[0]), int(move[1]))
        state = game.make_move(state, move)
        game.print_state(state)
    print(f'Player {game.symbols[state.winner]} wins')
