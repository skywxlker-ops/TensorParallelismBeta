import torch
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
torch.distributed.init_process_group("gloo", world_size=1, rank=0)
x = torch.tensor([1.0], requires_grad=True)
loss = x * 2.0
print("Requires grad before:", loss.requires_grad)
torch.distributed.all_reduce(loss)
print("Requires grad after:", loss.requires_grad)
loss.backward()
print("Grad:", x.grad)
