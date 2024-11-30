#%%
import pathlib
from datetime import datetime
import time
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed  

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger

from mcts import MCTS
from connect_game import ConnectGame, GameState
from model import ZeroModel
from search_tree import play_for_win_rate

np.set_printoptions(precision=3, suppress=True)
#%%
REPLAY_BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlphaConnectZero(MCTS):

    def __init__(self, 
                 *args, 
                 expand_n=10, 
                 train_steps=100,
                 self_play_games=10,
                 lr=0.001,
                 save_dir='./checkpoints', 
                 **kwargs):
        """
        Args:
            expand_n: number of expanding leaves in MCTS
            train_steps: number of training steps
            save_dir: directory to save model
        """
        super().__init__(*args, **kwargs)
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.expand_n = expand_n    # number of expanding leaves in MCTS
        self.train_steps = train_steps
        self.train_search_iterations = self.iterations
        self.self_play_games = self_play_games
        self.model = ZeroModel(self.game.size)
        self.load_model()
        self.model.eval()
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def load_model(self, path=None):
        if path is None:    # load best model
            path = self.save_dir / 'model_best.pth'
        if not path.exists():
            logger.warning('Best model not found!')
            self.save_model()
            self.model.to(DEVICE)
            return
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.to(DEVICE)
    
    def save_model(self, tag=None):
        if tag is None:
            path = self.save_dir / 'model_best.pth'
        else:
            path = self.save_dir / f'model_{str(tag)}.pth'
        torch.save(self.model.state_dict(), path)
    
    def train(self, epochs=1000):

        for epoch in range(epochs):

            # self play
            self.model.eval()

            # TODO virtual self play
            # self_play_data = self.self_play(self.self_play_games, enhance=True)
            self_play_data = self.paralle_self_play(self.self_play_games, enhance=True)
            self.replay_buffer.extend(self_play_data)

            # sample from replay buffer
            minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
            states, act_probs, values = self._array2tensor(minibatch)

            # update model
            start_time = time.time()
            self.model.train()
            mean_loss = 0
            for i in range(self.train_steps):
                mean_loss += self._train_step(states, act_probs, values)
            logger.info(f"Epoch [{epoch+1}/{epochs}], mean loss: [{mean_loss / self.train_steps:.3f}], " + 
                        f"train elapsed: {time.time() - start_time: .2f}s.")

            # evaluate model
            if epoch % 10 == 0:    # evaluate every 10 epochs

                # save model
                self.save_model(tag=epoch)
                logger.info(f"The checkpoint for epoch [{epoch+1}] has been saved.")

                # evaluate
                win_rate = self.evaluate()
                logger.info(f"Win rate: {win_rate:.3f}.")
                if win_rate > 0.5:
                    self.save_model()
                    logger.success("New best model is saved!")

    def evaluate(self, oponent_num=3, game_num=1, search_iterations=3):  
        win_rate = 0  
        self.iterations = search_iterations     # deduce search time in evaluation
        with ProcessPoolExecutor() as executor:  
            futures = []  
            for _ in range(oponent_num):  
                oponent = AlphaConnectZero(self.game, expand_n=self.expand_n, iterations=search_iterations)  
                # TODO random select oponent  
                oponent.load_model()  
                futures.append(executor.submit(play_for_win_rate, self.game, self, oponent, game_num))  

            for future in as_completed(futures):  
                win_rate += future.result()  
        
        self.iterations = self.train_search_iterations
        return win_rate / oponent_num  
    
    def _array2tensor(self, batch_data):
        """
        Args:
            batch_data: list of (state, action, value) tuples
        """
        states, act_probs, values = list(zip(*batch_data))
        states = torch.tensor(np.stack(states), dtype=torch.float32).to(DEVICE)
        act_probs = torch.tensor(np.stack(act_probs), dtype=torch.float32).to(DEVICE)
        values = torch.tensor(np.stack(values), dtype=torch.float32).to(DEVICE)
        return states, act_probs, values
    
    def _train_step(self, state_batch, action_batch, value_batch):
        pred_act, pred_value = self.model(state_batch)
        loss_policy = F.kl_div(torch.log(pred_act), action_batch, reduction='batchmean')
        loss_value = F.mse_loss(pred_value, value_batch)
        loss = loss_policy + loss_value
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _expand(self, node):
        if self.game.is_over(node.state):
            return node
        board = torch.tensor(node.state.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            act_prob, self.value = self.model(board)       # preserve the value for backpropagation
        act_prob = F.softmax(act_prob, dim=1)
        legal_mask = self.game.get_legal_moves(node.state)
        legal_mask = torch.tensor(legal_mask, dtype=torch.float32).to(DEVICE)
        act_prob = act_prob * legal_mask
        action_set = set()
        expand_n = min(self.expand_n, legal_mask.sum())
        while len(action_set) < expand_n:
            action = torch.multinomial(act_prob.view(-1), 1).item()
            if action not in action_set:
                action_set.add(action)
            move = self.game.action2position(action)
            state = self.game.make_move(node.state, move)
            child_node = node.expand(state, move)
        return random.choice(node.children)
    
    def _simulate(self, state):
        return self.value.detach().item()


if __name__ == '__main__':
    ...
#%%
    logger.add('logs/alpha_connect_zero.log', level='INFO')
    game = ConnectGame(9, 4)
    tree = AlphaConnectZero(game, iterations=100, train_steps=4, expand_n=5)
    tree.train(epochs=2)