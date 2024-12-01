from datetime import datetime
import argparse
from loguru import logger
from alpha_connect_zero import AlphaConnectZero
from connect_game import ConnectGame

argparser = argparse.ArgumentParser()
argparser.add_argument('--iterations', type=int, default=100)
argparser.add_argument('--train_epochs', type=int, default=100)
argparser.add_argument('--train_steps', type=int, default=100)
argparser.add_argument('--self_play_games', type=int, default=10)
argparser.add_argument('--replay_buffer_size', type=int, default=100000)
argparser.add_argument('--batch_size', type=int, default=1024)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--save_dir', type=str, default='./checkpoints')
args = argparser.parse_args()

logger.add(f'./logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log', level='TRACE')
logger.success(f"Arguments used: \n{args}")
game = ConnectGame(9, 4)
tree = AlphaConnectZero(game, **args.__dict__)
tree.train()
