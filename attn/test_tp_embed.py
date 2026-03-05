import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
import torch.distributed as dist

dist.init_process_group("nccl")
tp_world_size = dist.get_world_size()
mesh = init_device_mesh("cuda", (tp_world_size,))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(10, 8)
        self.head = nn.Linear(8, 10)

def count(model):
    for n, p in model.named_parameters():
        print(f"{n} {p.shape}")

m1 = Model().cuda()
parallelize_module(m1, mesh, {"wte": RowwiseParallel(), "head": ColwiseParallel()})
print("Rowwise wte:", m1.wte.weight.shape)

m2 = Model().cuda()
parallelize_module(m2, mesh, {"wte": ColwiseParallel(), "head": ColwiseParallel()})
print("Colwise wte:", m2.wte.weight.shape)
