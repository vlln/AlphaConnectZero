import os  
import copy
import pathlib  
import time  
import random  
from collections import deque  
from torch import multiprocessing as mp
import torch  
import torch.nn.functional as F  
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np  
from loguru import logger  
try:  
    import torch_sdaa  
    torch.nn.SyncBatchNorm = torch_sdaa.nn.SyncBatchNorm
    torch.cuda = torch.sdaa
    DEVICE_TYPE = 'sdaa'
except ImportError:  
    DEVICE_TYPE = 'cuda'

from mcts import MCTS  
from model import ZeroModel  
from search_tree import play_for_win_rate  
from connect_game import ConnectGame, ConnectFour

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
        self.model_save_dir = self.save_dir / str(self.game)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)  
        self.train_epochs = train_epochs  
        self.train_steps = train_steps  
        self.train_search_iterations = self.iterations  
        self.self_play_games = self_play_games  
        self.batch_size = batch_size  
        self.device_id = device_id
        self.rank = rank

        np.random.seed(self.rank)
        # torch.manual_seed(self.rank)  # may cause sdaa error
        if self.device_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'{DEVICE_TYPE}:{self.device_id}')

        self.model = ZeroModel(self.game.size, self.game.action_shape, 1)
        self.load_model()  
        self.model.eval()  
        self.replay_buffer = deque(maxlen=int(replay_buffer_size))  
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  

    def load_model(self, path=None):  
        if path is None:  # load best model  
            path = self.model_save_dir / 'model_best.pth'  
        if path.exists():  
            self.model.load_state_dict(torch.load(path, weights_only=True))  
        elif self.rank == 0:  
            logger.warning(f"rank_{self.rank}: Best model not found!")  
            self.save_model()  
        
        self.model.to(self.device)  

        """Using [DDP] method"""
        # DDP don't auto sync batchnorm  
        # self.ddp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        # self.ddp_model = DDP(self.ddp_model, device_ids=[self.device_id])
        # self.model = self.ddp_model

    def save_model(self, tag=None):  
        if tag is None:  
            path = self.model_save_dir / 'model_best.pth'  
        else:  
            path = self.model_save_dir / f'model_{str(tag)}.pth'  
        if isinstance(self.model, DDP):
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)  

    def train(self):  
        if self.device_id == -1:
            logger.error('CPU training is not supported yet!')
            raise NotImplementedError

        for epoch in range(self.train_epochs):  

            # self play  
            self.model.eval()  
            start_time = time.time()  
            self_play_data = self.self_play(self.self_play_games, enhance=True)  
            logger.trace(f"rank_{self.rank}: Self play elapsed: {time.time() - start_time: .2f}s.")  
            self.replay_buffer.extend(self_play_data)  
            logger.trace(f"rank_{self.rank}: Self play data size: [{len(self_play_data)}], replay buffer size: [{len(self.replay_buffer)}].")  

            # sample from replay buffer  
            minibatch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
            states, act_probs, values = self._array2tensor(minibatch)  

            # update model  
            start_time = time.time()  
            self.model.train()  
            mean_loss = 0  
            for i in range(self.train_steps):  
                # logger.trace(f"rank_{self.rank}: Epoch [{epoch+1}/{self.train_epochs}], step [{i+1}/{self.train_steps}]")  
                mean_loss += self._train_step(states, act_probs, values)  
            logger.trace(f"rank_{self.rank}: Train elapsed: {time.time() - start_time: .2f}s.")  

            logger.trace(f"rank_{self.rank}: Epoch [{epoch+1}/{self.train_epochs}], mean loss: [{mean_loss / self.train_steps:.3f}].")  

            # save epoch model  
            if epoch % 1 == 0 and self.rank == 0:
                self.save_model(tag=epoch)  
                logger.info(f"rank_{self.rank}: The checkpoint for epoch [{epoch+1}] has been saved.")  
            
            # evaluate  
            if epoch % 1 == 0:
                start_time = time.time()  
                win_rate = self.evaluate()  # win_rate is a float number
                logger.trace(f"rank_{self.rank}: Evaluate elapsed: {time.time() - start_time: .2f}s.")

                """Using [gather] method"""
                # win_rate = torch.tensor(win_rate, device=self.device, dtype=torch.float32)
                # gathered_win_rates = [torch.zeros_like(win_rate) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_win_rates, win_rate)  # 收集所有进程的 win_rate
                # average_win_rate = torch.mean(torch.stack(gathered_win_rates))
                # dist.barrier()
                # if self.rank == 0 and average_win_rate > 0.5:  
                #     logger.info(f"rank_{self.rank}: Win rate: {average_win_rate:.3f}.")  
                #     self.save_model()
                #     logger.success(f"rank_{self.rank}: New best model is saved!")


                """Using [reduce] method to get the average win rate"""
                win_rate = torch.tensor(win_rate, device=self.device, dtype=torch.float32)
                dist.all_reduce(win_rate, op=dist.ReduceOp.SUM)  
                win_rate /= dist.get_world_size()
                dist.barrier()
                if self.rank == 0 and win_rate > 0.5:  
                    logger.info(f"rank_{self.rank}: Win rate: {win_rate:.3f}.")  
                    self.save_model()
                    logger.success(f"rank_{self.rank}: New best model is saved!")  

    def evaluate(self, oponent_num=3, game_num=1, search_iterations=3):  
        win_rate = 0  
        self.iterations = search_iterations     # for deduce search time in evaluation  

        # TODO random select oponent  
        # oponent = copy.deepcopy(self)     # may crash sdaa bug
        oponent = AlphaConnectZero(**self.__dict__)
        oponent.model = ZeroModel(self.game.size, self.game.action_shape, 1)
        oponent.model.load_state_dict(torch.load(self.model_save_dir / 'model_best.pth', weights_only=True))  
        oponent.model.to(self.device)

        """Using [DDP] method need"""
        # self.model = ZeroModel(self.game.size, self.game.action_shape, 1)
        # Avoid assigning values to DDP models. DDP model state dict may differ from the normal model's state dict.
        # self.model.load_state_dict(self.ddp_model.module.state_dict())
        # self.model.to(self.device)

        for _ in range(oponent_num):  
            win_rate += play_for_win_rate(self.game, self, oponent, game_num)  

        self.iterations = self.train_search_iterations  

        """Using [DDP] method need"""
        # self.model = self.ddp_model

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
    
    """Using [gather] method"""
    # def _train_step(self, state_batch, action_batch, value_batch):
    #     pred_act, pred_value = self.model(state_batch)
    #     loss_policy = F.kl_div(torch.log(pred_act), action_batch, reduction='batchmean')
    #     loss_value = F.mse_loss(pred_value, value_batch)
    #     loss = loss_policy + loss_value
    #     loss = loss.unsqueeze(0)
    #     self.optimizer.zero_grad()
    #     loss.backward()  # Compute gradients
    #     for param in self.model.parameters():
    #         gathered_grads = [torch.zeros_like(param.grad.data) for _ in range(dist.get_world_size())]
    #         dist.all_gather(gathered_grads, param.grad.data)  # Gather gradients from all processes
    #         param.grad.data = torch.mean(torch.stack(gathered_grads), dim=0)  # Average gradients
    #     self.optimizer.step()
    #     gathered_losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
    #     dist.all_gather(gathered_losses, loss)  # Gather losses from all processes
    #     loss = torch.mean(torch.stack(gathered_losses))  # Average loss
    #     return loss.item()

    """Using [reduce] method"""
    def _train_step(self, state_batch, action_batch, value_batch):
        dist.barrier()
        pred_act, pred_value = self.model(state_batch)
        loss_policy = F.kl_div(pred_act.squeeze(1).log(), action_batch, reduction='batchmean')
        # loss_policy = F.cross_entropy(pred_act, action_batch)
        loss_value = F.mse_loss(pred_value, value_batch)
        loss = loss_policy + loss_value
        self.optimizer.zero_grad()
        loss.backward()     # backward before all_reduce, otherwise the gradients will not be autogradiented
        dist.all_reduce(loss, op=dist.ReduceOp.SUM) 
        loss /= dist.get_world_size()
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)  
            param.grad.data /= dist.get_world_size()
        self.optimizer.step()
        return loss.item()
    
    """Using [DDP] method"""
    # def _train_step(self, state_batch, action_batch, value_batch):
    #     self.optimizer.zero_grad()
    #     pred_act, pred_value = self.model(state_batch)
    #     loss_policy = F.kl_div(torch.log(pred_act), action_batch, reduction='batchmean')
    #     loss_value = F.mse_loss(pred_value, value_batch)
    #     loss = loss_policy + loss_value
    #     loss.backward()    
    #     self.optimizer.step()  
    #     return loss.item()

    def _expand(self, node):
        if self.game.is_over(node.state):
            return node
        board = torch.tensor(node.state.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        board *= self.act_player    # Model trained on First Player
        with torch.no_grad():
            act_prob, self.value = self.model(board)       # preserve the value for backpropagation
            # act_prob = F.softmax(act_prob, dim=1)
        legal_mask = torch.tensor(self.game.get_legal_moves(node.state)).to(self.device)
        act_prob = act_prob.squeeze(0) * legal_mask
        best_act = torch.multinomial(act_prob.view(-1), 1).item()
        possible_moves = np.argwhere(np.ones(legal_mask.shape, dtype=int))
        best_move = possible_moves[best_act]
        act_prob = np.reshape(act_prob.detach().cpu().numpy(), -1)
        moves = possible_moves[act_prob > 0].reshape(-1)

        best_node = node
        for i, move in enumerate(moves):
            state = self.game.make_move(node.state, move)
            child_node = node.expand(state, move)
            child_node.p = act_prob[i]
            if np.array_equal(best_move, move):
                best_node = child_node
        return best_node

    def _simulate(self, state):
        return self.value.detach().item() * self.act_player

if __name__ == '__main__':
    logger.add("./logs/test_train.log", level="TRACE")
    config = Config(
        # game=ConnectGame(9, 4),
        game = ConnectFour(),
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
