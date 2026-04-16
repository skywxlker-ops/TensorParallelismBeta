import torch
T = 16
N = 4
S = T // N
half = S // 2
unloaded = torch.zeros(T, dtype=torch.long)
for d in range(T):
    if d < T // 2:
        i = d // half
        k = d % half
        src_seq = i * S + k
    else:
        i = (T - d - 1) // half
        k = d - (T - (i + 1) * half)
        src_seq = i * S + half + k
    unloaded[d] = src_seq
print("Unloaded Layout: ", unloaded.tolist())
