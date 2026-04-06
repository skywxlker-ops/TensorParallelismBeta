"""
pytorch_attn_bench.py
Benchmark for PyTorch native SDPA simulating a full 2-rank CP ring
attention pass (2 ring steps: local K/V block + one received K/V block).
No MPI communication — compute cost only, matching the SDPA work done
by our C++ ring and Megatron ring per GPU.
"""
import torch
import torch.nn.functional as F

device = torch.device("cuda", 0)

B        = 4
T_local  = 512
n_head   = 6
head_dim = 64

qkv_shape = (B, n_head, T_local, head_dim)
NWARM  = 5
NITERS = 20


def ring_fwd_no_grad(q, k0, v0, k1, v1):
    """2-step ring attention forward (compute only, no comm).
    Step 0: local K/V with causal mask.
    Step 1: received remote K/V without causal mask.
    Outputs are summed to force both SDPA ops into the graph."""
    with torch.backends.cuda.sdp_kernel(
        enable_math=True, enable_mem_efficient=False, enable_flash=False
    ):
        o0 = F.scaled_dot_product_attention(q, k0, v0, is_causal=True)
        o1 = F.scaled_dot_product_attention(q, k1, v1, is_causal=False)
    return o0 + o1


# -----------------------------------------------------------------------
# Warm-up: 2-step ring fwd+bwd with fresh tensors
# -----------------------------------------------------------------------
for _ in range(NWARM):
    q  = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    k0 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    v0 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    k1 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    v1 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    out = ring_fwd_no_grad(q, k0, v0, k1, v1)
    out.sum().backward()
torch.cuda.synchronize()

# -----------------------------------------------------------------------
# Timed: forward-only (2-step ring)
# -----------------------------------------------------------------------
ts_fwd = torch.cuda.Event(enable_timing=True)
te_fwd = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
ts_fwd.record()
for _ in range(NITERS):
    q  = torch.randn(qkv_shape, dtype=torch.float32, device=device)
    k0 = torch.randn(qkv_shape, dtype=torch.float32, device=device)
    v0 = torch.randn(qkv_shape, dtype=torch.float32, device=device)
    k1 = torch.randn(qkv_shape, dtype=torch.float32, device=device)
    v1 = torch.randn(qkv_shape, dtype=torch.float32, device=device)
    with torch.no_grad():
        ring_fwd_no_grad(q, k0, v0, k1, v1)
te_fwd.record()
torch.cuda.synchronize()
ms_fwd = ts_fwd.elapsed_time(te_fwd) / NITERS

# -----------------------------------------------------------------------
# Re-warm backward kernels: the forward-only loop evicts backward kernel
# data from L2.  A brief fwd+bwd warmup restores the cache state.
# -----------------------------------------------------------------------
for _ in range(NWARM):
    q  = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    k0 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    v0 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    k1 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    v1 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    out = ring_fwd_no_grad(q, k0, v0, k1, v1)
    out.sum().backward()
torch.cuda.synchronize()

# -----------------------------------------------------------------------
# Timed: fwd+bwd (2-step ring)
# -----------------------------------------------------------------------
ts_tot = torch.cuda.Event(enable_timing=True)
te_tot = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
ts_tot.record()
for _ in range(NITERS):
    q  = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    k0 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    v0 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    k1 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    v1 = torch.randn(qkv_shape, dtype=torch.float32, device=device, requires_grad=True)
    out = ring_fwd_no_grad(q, k0, v0, k1, v1)
    out.sum().backward()
te_tot.record()
torch.cuda.synchronize()
ms_tot = ts_tot.elapsed_time(te_tot) / NITERS
ms_bwd = ms_tot - ms_fwd

print(f"PT_FWD_MS={ms_fwd:.4f}")
print(f"PT_BWD_MS={ms_bwd:.4f}")
