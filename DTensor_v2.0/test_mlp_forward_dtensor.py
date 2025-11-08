import dtensor
import torch
import torch.nn.functional as F
import numpy as np
from mpi4py import MPI

# ----------------------------------------------------
# Init MPI + NCCL
# ----------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

print(f"[Rank {rank}] Starting NCCL init...")
dtensor.init()

if rank == 0:
    nccl_id = dtensor.get_unique_id()
else:
    nccl_id = None
nccl_id = comm.bcast(nccl_id, root=0)

pg = dtensor.ProcessGroup(rank=rank, world_size=world_size, device=rank, nccl_id=nccl_id)
print(f"[Rank {rank}] ProcessGroup initialized ")

# ----------------------------------------------------
# Dummy input + model weights
# ----------------------------------------------------
torch.manual_seed(42 + rank)
x = torch.randn(32, 128, device=f"cuda:{rank}")

# Each rank gets its own local shard of weights
W1 = torch.randn(128, 64, device=f"cuda:{rank}") / np.sqrt(128)
b1 = torch.zeros(64, device=f"cuda:{rank}")
W2 = torch.randn(64, 10, device=f"cuda:{rank}") / np.sqrt(64)
b2 = torch.zeros(10, device=f"cuda:{rank}")

# ----------------------------------------------------
# Forward Pass
# ----------------------------------------------------
h = F.relu(x @ W1 + b1)
out_local = h @ W2 + b2

# Simulate distributed all-reduce to sum outputs
out_np = out_local.detach().cpu().numpy()
out_sum = np.zeros_like(out_np)
comm.Allreduce(out_np, out_sum, op=MPI.SUM)
out_sum /= world_size  # average

if rank == 0:
    print("\n[Rank 0] MLP Forward Pass (Distributed Avg Output):")
    print(out_sum[:5])  # show first few rows

print(f"[Rank {rank}] Forward pass complete ")
