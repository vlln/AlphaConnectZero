from datetime import datetime
import random  
import math  
from search_tree import SearchTree, TreeNode
import numpy as np

class MCTS(SearchTree):  

    def __init__(self, *args, c_puc=50, iterations=1000, **kwargs):
        """
        Initialize the MCTS class with the given parameters.
        Args:
            c_puc (float): exploration parameter.
            iterations (int): number of iterations to run the MCTS search.
        """
        super().__init__(*args, **kwargs)
        self.c_puct = c_puc
        self.iterations = iterations   

    def search(self, *args, **kwargs):  
        super().search(*args, **kwargs)
        for _ in range(self.iterations):  
            # select
            node = self._select_child(self.root)  
            # expand
            if not self.game.is_over(node.state):
                node = self._expand(node)
            # simulate
            reward = self._simulate(node.state)  
            # backpropagate
            self._backpropagate(node, reward)  
        # compute move probabilities
        act_score = np.array([child.score / (child.visits + 1) for child in self.root.children])
        moves = np.array([child.move for child in self.root.children])
        act_prob = np.zeros(self.game.board_size, dtype=np.float32)
        exp_x = np.exp(act_score - np.max(act_score))   # for safe softmax
        move_prob = exp_x / np.sum(exp_x)
        act_prob [moves[:, 0], moves[:, 1]] = move_prob    # shape: (board_rows, board_cols)
        best_move = moves[np.argmax(act_score)]
        return best_move, act_prob

    def _select_child(self, node: TreeNode) -> TreeNode:  
        """Select a leaf node"""
        best_child = node
        while not best_child.is_leaf():  
            best_child = self._get_best_child(best_child)
        return best_child
    
    def _expand(self, node: TreeNode):  
        """Expand the tree by adding new child nodes for all possible moves"""
        for move in self.game.get_possible_moves(node.state):  
            new_state = self.game.make_move(node.state, move)  
            child = node.expand(new_state, move)
            if self.game.is_over(new_state):
                return child
        if node.is_leaf():
            return node
        return random.choice(node.children)
    
    def _simulate(self, start_state):  
        state = start_state
        while not self.game.is_over(state):
            move = random.choice(self.game.get_possible_moves(state))  
            state = self.game.make_move(state, move)  
        return self.game.get_reward(state) * self.act_player
        # return 1 if self.game.get_reward(state) * self.act_player else 0

    def _backpropagate(self, node:TreeNode, reward):  
        node.visits += 1  
        node.score += reward
        if node.parent:  
            self._backpropagate(node.parent, reward)  

    def _get_best_child(self, node: TreeNode) -> TreeNode:  
        def ucb1(child):
            return child.score / (child.visits + 1) + self.c_puct * math.sqrt(node.visits) / (child.visits + 1)
        best_child = node.children[0]

        if self.act_player == node.state.current_player:
            best_child = max(node.children, key=ucb1)
        else:
            best_child = min(node.children, key=ucb1)

        # best_score = float('inf') * self.act_player
        # for child in node.children:  
        #     exploration_factor = self.c_puct * math.sqrt(math.log(node.visits) / (child.visits + 1))  
        #     score = child.score / (child.visits + 1) + exploration_factor  
        #     if score > best_score:  
        #         best_score = score  
                # best_child = child  
        # print(node.state.board)
        # print(f"best_child: {best_child.move}, best_score: {best_score}")

        return best_child
    