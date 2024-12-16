# file_name: test_spawn.py

import torch
import torch_sdaa
import torch.distributed as dist
import torch.multiprocessing as mp

# 定义训练函数
def fn(rank, ws, nums):
    torch.sdaa.set_device(rank)
    # 初始化通讯进程
    dist.init_process_group('tccl', init_method='tcp://127.0.0.1:28765',
                            rank=rank, world_size=ws)
    rank = dist.get_rank()
    print(f"rank = {rank} is initialized")
    torch.sdaa.set_device(rank)
    tensor = torch.tensor(nums).sdaa()
    print(tensor)
# 定义主函数，args参数是一个元组，包含了所有需要传递给训练函数的参数。
if __name__ == "__main__":
    ws = 4
    # mp.spawn(fn, nprocs=ws, args=(ws, [1, 2, 3, 4]))
    
    print(torch_sdaa.distributed.is_available())