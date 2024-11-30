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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlphaConnectZero(MCTS):

    def __init__(self, 
                 *args, 
                 train_epochs=10,
                 train_steps=100,
                 self_play_games=10,
                 lr=0.001,
                 replay_buffer_size=1000000,
                 batch_size=1024,
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
        self.train_epochs = train_epochs
        self.train_steps = train_steps
        self.train_search_iterations = self.iterations
        self.self_play_games = self_play_games
        self.batch_size = batch_size
        self.model = ZeroModel(self.game.size)
        self.load_model()
        self.model.eval()
        self.replay_buffer = deque(maxlen=int(replay_buffer_size))
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
    
    def train(self):
        for epoch in range(self.train_epochs):

            # self play
            self.model.eval()

            self_play_data = self.self_play(self.self_play_games, enhance=True)
            # self_play_data = self.paralle_self_play(self.self_play_games, enhance=True)
            self.replay_buffer.extend(self_play_data)

            # sample from replay buffer
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            states, act_probs, values = self._array2tensor(minibatch)

            # update model
            start_time = time.time()
            self.model.train()
            mean_loss = 0
            for i in range(self.train_steps):
                mean_loss += self._train_step(states, act_probs, values)
            logger.info(f"Epoch [{epoch+1}/{self.train_epochs}], mean loss: [{mean_loss / self.train_steps:.3f}], " + 
                        f"train elapsed: {time.time() - start_time: .2f}s.")

            # evaluate model
            if epoch % 10 == 0:    # evaluate every 10 epochs

                # save model
                self.save_model(tag=epoch)
                logger.info(f"The checkpoint for epoch [{epoch+1}] has been saved.")

                # evaluate
                win_rate = self.evaluate()
                # win_rate = self.paralle_evaluate()
                logger.info(f"Win rate: {win_rate:.3f}.")
                if win_rate > 0.5:
                    self.save_model()
                    logger.success("New best model is saved!")
    
    def evaluate(self, oponent_num=3, game_num=1, search_iterations=3):
        win_rate = 0
        self.iterations = search_iterations     # deduce search time in evaluation
        # TODO random select oponent  
        oponent = AlphaConnectZero(self.game, iterations=search_iterations)
        oponent.load_model()
        for _ in range(oponent_num):
            win_rate += play_for_win_rate(self.game, self, oponent, game_num)
        self.iterations = self.train_search_iterations
        return win_rate / oponent_num

    def paralle_evaluate(self, oponent_num=3, game_num=1, search_iterations=3):  
        win_rate = 0  
        self.iterations = search_iterations     # deduce search time in evaluation
        with ProcessPoolExecutor() as executor:  
            futures = []  
            for _ in range(oponent_num):  
                oponent = AlphaConnectZero(self.game, iterations=search_iterations)  
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
        act_prob = F.softmax(act_prob, dim=1).squeeze(0)
        best_move = self.game.action2position(torch.multinomial(act_prob.view(-1), 1).item())
        legal_mask = self.game.get_legal_moves(node.state)
        act_prob = act_prob.detach().cpu().numpy() * legal_mask    # shape: (n, n)

        moves = np.argwhere(act_prob > 0)
        best_node = node
        for move in moves:
            state = self.game.make_move(node.state, move)
            child_node = node.expand(state, move)
            child_node.p = act_prob[move[0], move[1]]
            if np.array_equal(best_move, move):
                best_node = child_node
        return best_node

    def _simulate(self, state):
        return self.value.detach().item()


REPLAY_BUFFER_SIZE = 1000000
BATCH_SIZE = 16
if __name__ == '__main__':
    ...
#%%
    logger.add('logs/alpha_connect_zero.log', level='INFO')
    game = ConnectGame(9, 4)
    tree = AlphaConnectZero(
        game, 
        iterations=1, 
        train_epochs=1,
        train_steps=4, 
        self_play_games=10,
        replay_buffer_size=REPLAY_BUFFER_SIZE, 
        batch_size=BATCH_SIZE
        )
    tree.train()