import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
import torch.distributed as dist

dist.init_process_group("nccl")
tp_world_size = dist.get_world_size()
mesh = init_device_mesh("cuda", (tp_world_size,))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(384, 3 * 384)

m = Model().cuda()
parallelize_module(m, mesh, {"c_attn": ColwiseParallel()})
x = torch.randn(8, 64, 384).cuda()
out = m.c_attn(x)
print("Out tuple:", [t.shape for t in out.split(384, dim=2)])
