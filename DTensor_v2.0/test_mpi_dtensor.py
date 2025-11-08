from mpi4py import MPI
import dtensor

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

print(f"[Rank {rank}] Starting NCCL init...")

# Step 1: Rank 0 generates the real NCCL unique ID
if rank == 0:
    nccl_id = dtensor.get_unique_id()
else:
    nccl_id = bytearray(128)

# Step 2: Broadcast NCCL ID to all ranks
comm.Bcast(nccl_id, root=0)

# Step 3: Initialize process group
pg = dtensor.ProcessGroup(rank, world_size, device=rank, nccl_id=bytes(nccl_id))

# Step 4: Initialize DTensor symbols
dtensor.init()

print(f"[Rank {rank}] ProcessGroup initialized ")
