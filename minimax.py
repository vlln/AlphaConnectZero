import numpy as np
from loguru import logger

from search_tree import SearchTree, TreeNode  

class Minimax(SearchTree):  

    def search(self, *args, **kwargs):  
        super().search(*args, **kwargs)
        self.count = 0  
        move, value = self._search(self.root, True, float('-inf'), float('inf'))  
        moves = np.array([child.move for child in self.root.children])
        move_prob = np.zeros((moves.shape[0]))
        move_prob[np.where(moves == move)[0][0]] = 1
        act_prob = self.game.get_act_prob(moves, move_prob)
        # act_prob = np.zeros((self.game.size, self.game.size))
        # act_prob[move[0], move[1]] = 1
        logger.debug(f"Number of nodes visited: {self.count}")  
        return move, act_prob
    
    def _search(self, node:TreeNode, is_maximizing, alpha, beta):  
        self.count += 1  
        if self.game.is_over(node.state):  
            return None, self.game.get_reward(node.state) * self.act_player
        
        best_value = float('-inf') if is_maximizing else float('inf')  
        best_move = None  
        
        for move in self.game.get_possible_moves(node.state):  
            child_state = self.game.make_move(node.state, move)  
            child_node = node.expand(child_state, move)
            _, value= self._search(child_node, not is_maximizing, alpha, beta)  
            
            if (is_maximizing and value > best_value) or (not is_maximizing and value < best_value):  
                best_value = value  
                best_move = move  
            
            if is_maximizing:  
                alpha = max(alpha, value)  
            else:  
                beta = min(beta, value)  
            node.score = best_value
            if beta <= alpha:  
                break  
        
        return best_move, best_value