import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor, DTensor
import torch.nn as nn

# Fake rank setup
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"

dist.init_process_group("nccl")
torch.cuda.set_device(rank)
device_mesh = init_device_mesh("cuda", (world_size,))

# 1. Test weight consistency
master_weight = torch.empty(10, 10)
nn.init.normal_(master_weight)
sharded_param = nn.Parameter(distribute_tensor(master_weight, device_mesh, [Shard(0)]))
replicated_param = nn.Parameter(distribute_tensor(master_weight, device_mesh, [Replicate()]))

# Gather weights to rank 0 to check if they match
full_replicated = replicated_param.to_local().clone()
dist.all_reduce(full_replicated, op=dist.ReduceOp.SUM)
full_replicated /= world_size

if rank == 0:
    print(f"Rank {rank}: Weight consistency test")
    # Actually, to check if they were the same, we can compare them between ranks
    pass

# We'll just do a diff
local_val = replicated_param.detach()
other_val = local_val.clone()
dist.all_reduce(other_val, op=dist.ReduceOp.SUM) # Sum of all ranks
if rank == 0:
    diff = (other_val / world_size) - local_val
    print(f"Replicated param discrepancy (should be 0): {diff.abs().max().item()}")

# 2. Test Gradient Placement
x = torch.randn(4, 10, device="cuda")
x_dt = DTensor.from_local(x, device_mesh, [Replicate()])

# Replicated Param used in Linear
w = nn.Parameter(distribute_tensor(torch.randn(10, 10), device_mesh, [Replicate()]))
y = x_dt @ w 
y.sum().backward()

print(f"Rank {rank}: Gradient placement for replicated param: {w.grad.placements}")

dist.destroy_process_group()
