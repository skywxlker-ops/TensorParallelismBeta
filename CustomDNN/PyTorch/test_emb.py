import os
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel, ColwiseParallel

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["WORLD_SIZE"] = "2"
os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")

torch.distributed.init_process_group("nccl")
device = f"cuda:{os.environ['RANK']}"
torch.cuda.set_device(device)
mesh = init_device_mesh("cuda", (2,))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 8)
    def forward(self, x):
        return self.emb(x)

m = Model().to(device)
print(f"Rank {mesh.get_rank()} Original weight:", m.emb.weight.shape)
m = parallelize_module(m, mesh, {"emb": RowwiseParallel()})
print(f"Rank {mesh.get_rank()} Parallel weight:", m.emb.weight.shape)

x = torch.tensor([[1, 5, 9]]).to(device)
out = m(x)
print(f"Rank {mesh.get_rank()} Output shape: {out.shape}, type: {type(out)}")
