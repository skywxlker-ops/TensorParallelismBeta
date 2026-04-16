import torch
T = 16
N = 4
S = T // N
half = S // 2
global_indices = torch.zeros(T, dtype=torch.long)
for p in range(T):
    i = p // S
    r = p % S
    if r < half:
        global_indices[p] = i * half + r
    else:
        global_indices[p] = T - (i + 1) * half + (r - half)
print("PyTorch Layout: ", global_indices.tolist())
