import torch
from torch.nn import functional as F

B, H, T, D = 4, 6, 32, 64
scale = 1.0 / (D ** 0.5)

q = torch.randn(B, H, T, D, requires_grad=True)
k = torch.randn(B, H, T, D, requires_grad=True)
v = torch.randn(B, H, T, D, requires_grad=True)

# Std Attention
scores = torch.matmul(q * scale, k.transpose(-2, -1))
mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
scores = scores.masked_fill(mask == 0, float('-inf'))
probs = F.softmax(scores, dim=-1)
out = torch.matmul(probs, v)

# Grad
dout = torch.ones_like(out)
out.backward(dout)

print(f"STD: q_grad norm: {q.grad.norm().item()}")
print(f"STD: k_grad norm: {k.grad.norm().item()}")
print(f"STD: v_grad norm: {v.grad.norm().item()}")
