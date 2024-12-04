import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
try:  
    import torch_sdaa  
    torch.cuda = torch.sdaa
    DEVICE_TYPE = 'sdaa'
except ImportError:  
    DEVICE_TYPE = 'cuda'
from datetime import datetime
from loguru import logger
from alpha_connect_zero import AlphaConnectZero, Config
from connect_game import ConnectGame, ConnectFour

def init_process(rank, backend, world_size, device_ids, log_filename, config:Config):  
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '12317'
    logger.add(log_filename, level='TRACE', enqueue=True)
    torch.cuda.set_device(device_ids[rank])     # if not set, may cause sdaa error
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    tree = AlphaConnectZero(**config.args, rank=rank, device_id=device_ids[rank])
    logger.success(f"Process [{rank}] is initialized.")
    tree.train()

if __name__ == '__main__':  
    log_filename = f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logger.add(log_filename, level='TRACE', enqueue=True)

    if not dist.is_available():
        raise RuntimeError("PyTorch distributed is not available.")
        
    if "torch_sdaa" in globals() and torch_sdaa.distributed.is_available():
        backend = 'tccl'
    elif dist.is_nccl_available():
        backend = 'nccl'
    elif dist.is_gloo_available():
        backend = 'gloo'
    else:
        raise RuntimeError("No available backend.")

    # Prameters
    device_count = torch.cuda.device_count()
    game = ConnectFour()
    process_per_device = 1
    world_size = device_count * process_per_device
    device_ids = [i for i in range(device_count)] * process_per_device
    # train config
    config = Config(
        game = game,
        iterations = 100,
        train_epochs = 100,
        train_steps = 1000,
        self_play_games = 20,
        lr = 0.001,
        replay_buffer_size = 50000,
        batch_size = 2048,
        save_dir = './checkpoints'
    )
    # test mini size
    config = Config(
        game = game,
        iterations = 3,
        train_epochs = 40, 
        train_steps = 10,
        self_play_games = 10,
        lr = 0.001, 
        replay_buffer_size = 100000,
        batch_size = 32,
        save_dir = './checkpoints'  
    )
    
    logger.info(f"Prameters: {config.args}")
    logger.info(f"Backend: {backend}, Device count: {device_count}, Process per device: {process_per_device}, World size: {world_size}")
    logger.success("Start training")
    mp.spawn(init_process, args=(backend, world_size, device_ids, log_filename, config), nprocs=world_size, join=True)
    logger.success("Training finished!")  
