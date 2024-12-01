import os  
import pathlib  
import time  
import random  
from collections import deque  
import torch  
import torch.nn.functional as F  
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np  
from loguru import logger  
try:  
    import torch_sdaa  
    torch.cuda = torch.sdaa
    DEVICE_TYPE = 'sdaa'
except ImportError:  
    DEVICE_TYPE = 'cuda'

from mcts import MCTS  
from model import ZeroModel  
from search_tree import play_for_win_rate  
from connect_game import ConnectGame

class Config(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    @property
    def args(self):
        return self.__dict__

class AlphaConnectZero(MCTS):  

    def __init__(self,   
                 *args,   
                 train_epochs=10,  
                 train_steps=100,  
                 self_play_games=10,  
                 lr=0.001,  
                 replay_buffer_size=100000,  
                 batch_size=1024,  
                 save_dir='./checkpoints',   
                 device_id=-1,  # -1 for cpu, 0 for cuda:0
                 rank=0,
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
        self.device_id = device_id
        self.rank = rank

        self.model = ZeroModel(self.game.size)  
        self.load_model()  
        self.model.eval()  
        self.replay_buffer = deque(maxlen=int(replay_buffer_size))  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  
    
    @property
    def device(self):
        if self.device_id == -1:
            return torch.device('cpu')
        else:
            return torch.device(f'{DEVICE_TYPE}:{self.device_id}')
    
    def load_model(self, path=None):  
        if path is None:    # load best model  
            path = self.save_dir / 'model_best.pth'  
        if path.exists():  
            self.model.load_state_dict(torch.load(path, weights_only=True))  
        else:
            logger.warning('Best model not found!')  
            self.save_model()  
        self.model.to(self.device)  
        # self.model = DDP(self.model, device_ids=[self.device_id])
    
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

            start_time = time.time()  
            self_play_data = self.self_play(self.self_play_games, enhance=True)  
            logger.trace(f"Self play elapsed: {time.time() - start_time: .2f}s.")  
            self.replay_buffer.extend(self_play_data)  
            logger.trace(f"Self play data size: [{len(self_play_data)}], replay buffer size: [{len(self.replay_buffer)}].")  

            # sample from replay buffer  
            minibatch = random.sample(self.replay_buffer, self.batch_size)  
            states, act_probs, values = self._array2tensor(minibatch)  

            # update model  
            start_time = time.time()  
            self.model.train()  
            mean_loss = 0  
            for i in range(self.train_steps):  
                logger.trace(f"Epoch [{epoch+1}/{self.train_epochs}], step [{i+1}/{self.train_steps}]")  
                mean_loss += self._train_step(states, act_probs, values)  
            logger.trace(f"Train elapsed: {time.time() - start_time: .2f}s.")  

            if self.rank == 0:
                logger.info(f"Epoch [{epoch+1}/{self.train_epochs}], mean loss: [{mean_loss / self.train_steps:.3f}].")  

                # evaluate model  
                if epoch % 10 == 0:    # evaluate every 10 epochs  

                    # save model  
                    self.save_model(tag=epoch)  
                    logger.info(f"The checkpoint for epoch [{epoch+1}] has been saved.")  

                    # evaluate  
                    start_time = time.time()  
                    win_rate = self.evaluate()  
                    logger.info(f"Evaluate elapsed: {time.time() - start_time: .2f}s.")  
                    logger.info(f"Win rate: {win_rate:.3f}.")  
                    if win_rate > 0.5:  
                        self.save_model()  
                        logger.success("New best model is saved!")  

    def evaluate(self, oponent_num=3, game_num=1, search_iterations=3):  
        win_rate = 0  
        self.iterations = search_iterations     # for deduce search time in evaluation  
        # TODO random select oponent  
        oponent = AlphaConnectZero(self.game, iterations=search_iterations, \
            device_id=self.device_id, rank=self.rank, save_dir=self.save_dir)  
        oponent.load_model()  
        for _ in range(oponent_num):  
            win_rate += play_for_win_rate(self.game, self, oponent, game_num)  
        self.iterations = self.train_search_iterations  
        return win_rate / oponent_num  
   
    def _array2tensor(self, batch_data):  
        """  
        Args:  
            batch_data: list of (state, action, value) tuples  
        """  
        states, act_probs, values = list(zip(*batch_data))  
        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)  
        act_probs = torch.tensor(np.stack(act_probs), dtype=torch.float32).to(self.device)  
        values = torch.tensor(np.stack(values), dtype=torch.float32).to(self.device)  
        return states, act_probs, values  

    # def _train_step(self, state_batch, action_batch, value_batch):
    #     dist.barrier()
    #     pred_act, pred_value = self.model(state_batch)
    #     loss_policy = F.kl_div(torch.log(pred_act), action_batch, reduction='batchmean')
    #     loss_value = F.mse_loss(pred_value, value_batch)
    #     loss = loss_policy + loss_value
    #     self.optimizer.zero_grad()
    #     loss.backward()     # backward before all_reduce, otherwise the gradients will not be autogradiented
    #     dist.all_reduce(loss, op=dist.ReduceOp.SUM) 
    #     loss /= dist.get_world_size()
    #     for param in self.model.parameters():
    #         dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)  
    #         param.grad.data /= dist.get_world_size()
    #     self.optimizer.step()
    #     return loss.item()
    
    def _train_step(self, state_batch, action_batch, value_batch):
        self.optimizer.zero_grad()
        pred_act, pred_value = self.model(state_batch)
        loss_policy = F.kl_div(torch.log(pred_act), action_batch, reduction='batchmean')
        loss_value = F.mse_loss(pred_value, value_batch)
        loss = loss_policy + loss_value
        loss.backward()     # backward before all_reduce, otherwise the gradients will not be autogradiented
        self.optimizer.step()
        return loss.item()

    def _expand(self, node):
        if self.game.is_over(node.state):
            return node
        board = torch.tensor(node.state.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
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

if __name__ == '__main__':
    logger.add("./logs/test_train.log", level="TRACE")
    config = Config(
        game=ConnectGame(9, 4),
        iterations=3,
        train_epochs=2,
        train_steps=10,
        self_play_games=10,
        lr=0.001,
        replay_buffer_size=100000,
        batch_size=32,
        save_dir='./checkpoints',
        device_id=0,
        rank=0,
    )
    tree = AlphaConnectZero(**config.args)
    tree.train()

