import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.utils.data import Dataset, DataLoader
# Correct import for DistributedSampler
from torch.utils.data.distributed import DistributedSampler 
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from log_utils import rank_log, get_logger, verify_min_gpu_count

# ---- 1. Initialize Distributed Environment First ----
_min_gpu_count = 2
if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus. Exiting.")
    sys.exit()

# These must be defined before the DataLoader uses them
_world_size = int(os.environ.get("WORLD_SIZE", 1))
# Get accelerator (cuda/cpu/mps)
accelerator = torch.accelerator.current_accelerator()
device_type = accelerator.type if accelerator else "cuda" 

device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()

logger = get_logger()
rank_log(_rank, logger, f"Starting TP example. World size: {_world_size}")

# ---- 2. Dataset Definition ----
class NpyShardDataset(Dataset):
    def __init__(self, shard_paths, input_dim=384):
        # Only store paths to avoid opening all files at once in the main process
        self.shard_paths = shard_paths
        self.input_dim = input_dim
        # Calculate lengths once
        self.lengths = []
        for p in shard_paths:
            m = np.load(p, mmap_mode='r')
            self.lengths.append(len(m))
        self.total_len = sum(self.lengths)
        # Cache for opened shards to prevent reopening constantly
        self.current_shard_idx = -1
        self.current_shard_data = None

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        shard_idx = 0
        temp_idx = idx
        for length in self.lengths:
            if temp_idx < length:
                break
            temp_idx -= length
            shard_idx += 1

        if self.current_shard_idx != shard_idx:
            self.current_shard_data = np.load(self.shard_paths[shard_idx], mmap_mode='r')
            self.current_shard_idx = shard_idx
    
        assert self.current_shard_data is not None, "Shard data failed to load"
        
        # Now Pylance knows this is not None
        raw_row = self.current_shard_data[temp_idx]
        
        sample = torch.from_numpy(raw_row[:self.input_dim].astype(np.float32))
        return sample


# --- Configuration ---
BATCH_SIZE = 8
shard_files = [
    "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/Data_Loader/Edufineweb_numpy/edufineweb_train_000001.npy",
    "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/Data_Loader/Edufineweb_numpy/edufineweb_train_000002.npy",
    "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/Data_Loader/Edufineweb_numpy/edufineweb_train_000003.npy",
    "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/Data_Loader/Edufineweb_numpy/edufineweb_train_000004.npy"
] 

# --- Configuration for Stability ---
BATCH_SIZE = 4 # Reduced further to guarantee stability during testing
dataset = NpyShardDataset(shard_files, input_dim=384)

# Use shuffle=False and drop_last=True for more predictable memory usage
sampler = DistributedSampler(dataset, num_replicas=_world_size, rank=_rank, shuffle=False)
train_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,
    num_workers=0, # Keep at 0 to avoid multi-process memory multiplication
    pin_memory=False
)


dataset = NpyShardDataset(shard_files)
sampler = DistributedSampler(
    dataset, num_replicas=_world_size, rank=_rank, shuffle=False
)


train_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,
    num_workers=0,      # Start with 0 to keep everything in the main memory
    pin_memory=False    # Set to False if RAM is extremely tight
)


# ---- 3. Model Definition ----
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        # Ensure input features (10) matches your .npy data dimensions!
        self.in_proj = nn.Linear(384,1536) 
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(1536, 384)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))

# Move model to device and parallelize
tp_model = ToyModel().to(device_type)
tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(),
        "out_proj": RowwiseParallel(),
    },
)

optimizer = torch.optim.AdamW(tp_model.parameters(), lr=1, foreach=True)

# ---- 4. Training Loop with Real Data ----
rank_log(_rank, logger, "Tensor Parallel training starting with .npy shards...")

for epoch in range(1): # Example 1 epoch
    sampler.set_epoch(epoch)
    for i, batch_data in enumerate(train_loader):
        # batch_data comes from your .npy files
        inp = batch_data.to(device_type)
        
        optimizer.zero_grad()
        output = tp_model(inp)
        
        # Loss (Dummy target for demo)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            rank_log(_rank, logger, f"Iter {i} completed")

rank_log(_rank, logger, "Training completed!")

if dist.is_initialized():
    dist.destroy_process_group()
