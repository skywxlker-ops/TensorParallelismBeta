
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from log_utils import rank_log, get_logger, verify_min_gpu_count

# ---- GPU check ------------
_min_gpu_count = 2

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

# Recommended official path
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh


class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(32, 5)

    def forward(self, x):
        return self.out_proj(self.relu(self.in_proj(x)))


"""
Main body of the demo of a basic version of tensor parallel by using
PyTorch native APIs.
"""
logger = get_logger()

# create a device mesh based on the given world_size.
_world_size = int(os.environ["WORLD_SIZE"])
device_type = torch.accelerator.current_accelerator().type
device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()


print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {_world_size} gpus"

rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")

# create model and move it to GPU - initdevice_type_mesh has already mapped GPU ids.
tp_model = ToyModel().to(device_type)


# Custom parallelization plan for the model
tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(),
        "out_proj": RowwiseParallel(),
    },
)

# Create an optimizer for the parallelized module.
lr = 0.25
optimizer = torch.optim.AdamW(tp_model.parameters(), lr=lr, foreach=True)


# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
num_iters = 10
rank_log(_rank, logger, "Tensor Parallel training starting...")

for i in range(num_iters):
    # For TP, input needs to be same across all TP ranks.
    # Setting the random seed is to mimic the behavior of dataloader.
    torch.manual_seed(i)
    inp = torch.rand(20, 10, device=device_type)
    output = tp_model(inp)
    output.sum().backward()
    optimizer.step()
    rank_log(_rank, logger, f"Tensor Parallel iter {i} completed")

rank_log(_rank, logger, "Tensor Parallel training completed!")

if dist.is_initialized():
    dist.destroy_process_group()