import torch
from torch.nn import functional as F

B, H, T, D = 2, 4, 16, 32
scale = 1.0 / (D ** 0.5)

q = torch.randn(B, H, T, D, requires_grad=True, dtype=torch.float64)
k = torch.randn(B, H, T, D, requires_grad=True, dtype=torch.float64)
v = torch.randn(B, H, T, D, requires_grad=True, dtype=torch.float64)

# Forward exact
scores = torch.matmul(q * scale, k.transpose(-2, -1))
# Assume self-chunk (causal)
mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
scores = scores.masked_fill(mask == 0, float('-inf'))
max_scores = scores.max(dim=-1, keepdim=True).values
max_scores = max_scores.masked_fill(max_scores == float('-inf'), 0.0)
exp_scores = torch.exp(scores - max_scores)
sum_exp = exp_scores.sum(dim=-1, keepdim=True)
lse = max_scores + torch.log(sum_exp)

probs = F.softmax(scores, dim=-1)
out = torch.matmul(probs, v)

dout = torch.randn_like(out)
out.backward(dout)

ref_dq, ref_dk, ref_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

# Manual backward
q_d = q.detach()
k_d = k.detach()
v_d = v.detach()
out_d = out.detach()
dout_d = dout.detach()
lse_d = lse.detach()

# Pretend merged_lse == lse, so lse_diff = 0
lse_diff = torch.zeros_like(lse_d)
weight = torch.exp(lse_diff)

P_local = F.softmax(scores.detach(), dim=-1)
P_global = P_local * weight

P_global_t = P_global.transpose(-2, -1)
dV = torch.matmul(P_global_t, dout_d)

dP_global = torch.matmul(dout_d, v_d.transpose(-2, -1))
D_global = (dout_d * out_d).sum(dim=-1, keepdim=True)

dS = P_global * (dP_global - D_global)
dS_scaled = dS * scale

dQ = torch.matmul(dS_scaled, k_d)
dK = torch.matmul(dS_scaled.transpose(-2, -1), q_d)

print(f"dQ max diff: {(dQ - ref_dq).abs().max().item()}")
print(f"dK max diff: {(dK - ref_dk).abs().max().item()}")
print(f"dV max diff: {(dV - ref_dv).abs().max().item()}")

