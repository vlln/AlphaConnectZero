from alpha_connect_zero import AlphaConnectZero
from connect_game import ConnectGame
from loguru import logger
import argparse

REPLAY_BUFFER_SIZE = 1000000
BATCH_SIZE = 16

argparser = argparse.ArgumentParser()
argparser.add_argument('--iterations', type=int, default=1000)
argparser.add_argument('--train_epochs', type=int, default=10)
argparser.add_argument('--train_steps', type=int, default=100)
argparser.add_argument('--self_play_games', type=int, default=10)
argparser.add_argument('--replay_buffer_size', type=int, default=100000)
argparser.add_argument('--batch_size', type=int, default=1024)
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--save_dir', type=str, default='./checkpoints')
args = argparser.parse_args()
print(args)

logger.add('logs/alpha_connect_zero.log', level='INFO')
game = ConnectGame(9, 4)
tree = AlphaConnectZero(game, **args.__dict__)
tree.train()
