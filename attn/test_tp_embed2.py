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

m1 = Model().cuda()
parallelize_module(m1, mesh, {"wte": RowwiseParallel()})
out = m1.wte(torch.tensor([[1, 2]]).cuda())
print("Rowwise wte out local:", out._local_tensor.shape)

m2 = Model().cuda()
parallelize_module(m2, mesh, {"wte": ColwiseParallel()})
out2 = m2.wte(torch.tensor([[1, 2]]).cuda())
print("Colwise wte out local:", out2._local_tensor.shape)
