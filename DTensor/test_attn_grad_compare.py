"""
test_attn_grad_compare.py

Compares attention-layer param and gradient values between:
  A) PyTorch CP reference  (gpt2_cp_headtail_fp32.py / gpt2_context_parallel_fp32.py)
  B) Manual ring attention  (mirrors ContextParallel.h logic in Python)

Run modes:
  No load balance (matches gpt2_context_parallel_fp32.py):
    torchrun --standalone --nnodes=1 --nproc-per-node=2 \
      DTensor/test_attn_grad_compare.py --lb=0

  HeadTail load balance (matches gpt2_cp_headtail_fp32.py + C++ gpt2_cp_test):
    torchrun --standalone --nnodes=1 --nproc-per-node=2 \
      DTensor/test_attn_grad_compare.py --lb=1

Output: for each section (INIT WEIGHTS, Q/K/V, ATTN_OUT, PROJ_OUT, GRADS)
  both models print mean/std/min/max and first-8 values on rank 0.
  Mismatches here pinpoint the divergence point.

Compare with C++ output by uncommenting TEMP DEBUGGING in gpt2_cp_test.cpp
and adding display() calls in CPAttention::forward + ContextParallelBackward::apply.
"""

import os
import sys
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import _cp_options


# ============================================================================
# Distributed init
# ============================================================================

dist.init_process_group(backend="nccl")

local_rank    = int(os.environ.get("LOCAL_RANK", 0))
world_size    = dist.get_world_size()
rank          = dist.get_rank()
device        = torch.device("cuda", local_rank)
torch.cuda.set_device(local_rank)

cp_mesh   = init_device_mesh("cuda", (world_size,))
cp_group  = dist.group.WORLD

# ============================================================================
# Config
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--lb", type=int, default=0, help="1=enable HeadTail load balance")
args = parser.parse_args()

USE_LB = bool(args.lb)

B         = 4
T         = 1024          # full sequence length
T_local   = T // world_size
N_EMBD    = 384
N_HEAD    = 6
N_LAYERS  = 3              # used only for proj_std
HEAD_DIM  = N_EMBD // N_HEAD
SCALE     = 1.0 / math.sqrt(HEAD_DIM)
SEED_ATTN = 1434           # matches C++: base_seed=1234, layer_0 offset=200
SEED_PROJ = 1435           # SEED_ATTN + 1

PROJ_STD  = 0.02 / math.sqrt(2.0 * N_LAYERS)


# ============================================================================
# display() -- matches C++ Tensor::display() style output
# ============================================================================

def display(name: str, t: torch.Tensor, max_vals: int = 8) -> None:
    if rank != 0:
        return
    flat = t.detach().float().cpu().flatten()
    n    = flat.numel()
    first_vals = flat[:max_vals].tolist()
    vals_str   = "  ".join(f"{v:+.6f}" for v in first_vals)
    print(
        f"[{name}]  shape={list(t.shape)}  dtype={t.dtype}  "
        f"mean={flat.mean().item():+.6f}  std={flat.std().item():.6f}  "
        f"min={flat.min().item():+.6f}  max={flat.max().item():+.6f}\n"
        f"  first {min(n, max_vals)} values: {vals_str}"
    )


def section(title: str) -> None:
    if rank == 0:
        print(f"\n{'='*72}")
        print(f"  {title}")
        print('='*72)


# ============================================================================
# Weight initialisation (mirrors C++ init_linear_gpt2)
# ============================================================================

def _init_weight_seeded(weight: torch.Tensor, std: float, seed: int) -> None:
    g = torch.Generator()
    g.manual_seed(seed)
    nn.init.normal_(weight, std=std, generator=g)


def _init_bias_zero(bias: torch.Tensor) -> None:
    nn.init.zeros_(bias)


# ============================================================================
# HeadTail permutation  (mirrors DTensor/tensor/dtensor.cpp HeadTail::loadbalance)
#
# The C++ kernel interleaves first half and (reversed) second half of the
# sequence: output[2*i]   = input[i]
#           output[2*i+1] = input[T-1-i]   for i in 0..T/2-1
# ============================================================================

def headtail_permute(x: torch.Tensor) -> torch.Tensor:
    """x: [B, T, C] or [B, H, T, D] -- permutes the sequence/T dim."""
    seq_dim = 2 if x.ndim == 4 else 1
    T_seq   = x.shape[seq_dim]
    half    = T_seq // 2
    front   = x.narrow(seq_dim, 0, half)
    back    = x.narrow(seq_dim, half, half).flip(seq_dim)
    interleaved = torch.stack([front, back], dim=seq_dim + 1)
    flat_shape  = list(x.shape)
    flat_shape[seq_dim] = T_seq
    return interleaved.reshape(flat_shape)


def headtail_unpermute(x: torch.Tensor) -> torch.Tensor:
    """Inverse of headtail_permute."""
    seq_dim = 2 if x.ndim == 4 else 1
    T_seq   = x.shape[seq_dim]
    half    = T_seq // 2
    even    = x[..., ::2, :] if x.ndim == 4 else x[:, ::2, :]
    odd     = x[..., 1::2, :] if x.ndim == 4 else x[:, 1::2, :]
    front   = even
    back    = odd.flip(seq_dim)
    return torch.cat([front, back], dim=seq_dim)


# ============================================================================
# Manual ring attention  (mirrors ContextParallel::forward_cp)
#
# No load balance variant: contiguous T/n chunks.
# With load balance: headtail_permute before sharding, headtail_unpermute after.
#
# q/k/v are [B, H, T, D] full-sequence tensors on each rank (pre-sharding).
# Returns [B, H, T, D] full output tensor with grad_fn for backward.
# ============================================================================

def _ring_rotate(tensor: torch.Tensor, pg) -> torch.Tensor:
    """AlltoAll-based ring shift (all ranks send to next, receive from prev)."""
    buf = torch.empty_like(tensor)
    ops = [dist.P2POp(dist.isend, tensor, (rank + 1) % world_size, pg),
           dist.P2POp(dist.irecv, buf,    (rank - 1) % world_size, pg)]
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()
    return buf


def manual_ring_forward(
    q_full: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    use_lb: bool,
    pg,
    scale: float,
) -> torch.Tensor:
    """
    Manual ring attention matching ContextParallel::forward_cp.

    Inputs: [B, H, T, D] full-sequence tensors (all identical across ranks).
    Returns: [B, H, T, D] attention output (differentiable).
    """
    q_work = q_full.contiguous()
    k_work = k_full.contiguous()
    v_work = v_full.contiguous()

    if use_lb:
        q_work = headtail_permute(q_work)
        k_work = headtail_permute(k_work)
        v_work = headtail_permute(v_work)

    local_q = q_work[:, :, rank * T_local : (rank + 1) * T_local, :].contiguous()
    local_k = k_work[:, :, rank * T_local : (rank + 1) * T_local, :].contiguous()
    local_v = v_work[:, :, rank * T_local : (rank + 1) * T_local, :].contiguous()

    running_out: torch.Tensor = None
    running_lse: torch.Tensor = None

    curr_k = local_k.detach().clone()
    curr_v = local_v.detach().clone()

    for i in range(world_size):
        source_rank = (rank - i) % world_size

        if use_lb:
            if source_rank == rank:
                mask_type = "causal"
            else:
                mask_type = "none"
        else:
            if source_rank == rank:
                mask_type = "causal"
            elif source_rank > rank:
                if i < world_size - 1:
                    curr_k = _ring_rotate(curr_k, pg)
                    curr_v = _ring_rotate(curr_v, pg)
                continue
            else:
                mask_type = "none"

        q_off = rank * T_local
        k_off = source_rank * T_local

        if mask_type == "causal":
            attn_mask = None
            is_causal  = True
        else:
            attn_mask = None
            is_causal  = False

        partial_out = F.scaled_dot_product_attention(
            local_q, curr_k, curr_v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=0.0,
            scale=scale,
        )

        T_q = local_q.shape[2]
        lse_shape = (local_q.shape[0], local_q.shape[1], T_q, 1)
        partial_lse = _compute_lse(local_q, curr_k, curr_v,
                                    is_causal, scale, mask_type,
                                    q_off, k_off)

        if running_out is None:
            running_out = partial_out
            running_lse = partial_lse
        else:
            running_out, running_lse = _merge_sdpa(
                running_out, running_lse, partial_out, partial_lse
            )

        if i < world_size - 1:
            curr_k = _ring_rotate(curr_k, pg)
            curr_v = _ring_rotate(curr_v, pg)

    out_chunks = [torch.empty_like(running_out) for _ in range(world_size)]
    dist.all_gather(out_chunks, running_out, group=pg)
    full_out = torch.cat(out_chunks, dim=2)

    if use_lb:
        full_out = headtail_unpermute(full_out)

    return full_out


def _compute_lse(q, k, v, is_causal, scale, mask_type, q_off, k_off):
    """
    Compute log-sum-exp [B,H,T_q,1] for given q,k in float32.
    Uses a softmax trick: LSE = log(sum(exp(scores))).
    """
    with torch.no_grad():
        qf = q.float()
        kf = k.float()
        scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
        T_q, T_k = scores.shape[-2], scores.shape[-1]
        if is_causal:
            causal_mask = torch.tril(
                torch.ones(T_q, T_k, device=q.device, dtype=torch.bool),
                diagonal=k_off - q_off
            )
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    return lse


def _merge_sdpa(out1, lse1, out2, lse2):
    """Online softmax merge (mirrors SDPAMerger::step, numerical only)."""
    with torch.no_grad():
        lse_diff  = (lse2 - lse1).float()
        sig       = torch.sigmoid(lse_diff)
        out_diff  = (out1 - out2).float()
        correction = sig * out_diff
        new_out   = out1 - correction
        log_sig   = -F.softplus(-lse_diff)
        new_lse   = lse1 - log_sig
    return new_out.to(out1.dtype), new_lse.to(lse1.dtype)


# ============================================================================
# Reference attention  (PyTorch context_parallel)
# ============================================================================

class RefAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln     = nn.LayerNorm(N_EMBD)
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=True)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD,     bias=True)
        self._init_weights()

    def _init_weights(self):
        _init_weight_seeded(self.c_attn.weight, 0.02, SEED_ATTN)
        _init_bias_zero(self.c_attn.bias)
        _init_weight_seeded(self.c_proj.weight, PROJ_STD, SEED_PROJ)
        _init_bias_zero(self.c_proj.bias)
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def forward(self, x):
        B_in, T_in, C = x.shape
        h   = self.ln(x)
        qkv = self.c_attn(h)
        q, k, v = qkv.split(N_EMBD, dim=2)
        q = q.view(B_in, T_in, N_HEAD, HEAD_DIM).transpose(1, 2)
        k = k.view(B_in, T_in, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B_in, T_in, N_HEAD, HEAD_DIM).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B_in, T_in, C)
        return x + self.c_proj(y)


# ============================================================================
# Manual attention  (ring attention in Python, shares weights with Ref)
# ============================================================================

class ManualAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln     = nn.LayerNorm(N_EMBD)
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=True)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD,     bias=True)
        self._init_weights()

    def _init_weights(self):
        _init_weight_seeded(self.c_attn.weight, 0.02, SEED_ATTN)
        _init_bias_zero(self.c_attn.bias)
        _init_weight_seeded(self.c_proj.weight, PROJ_STD, SEED_PROJ)
        _init_bias_zero(self.c_proj.bias)
        nn.init.ones_(self.ln.weight)
        nn.init.zeros_(self.ln.bias)

    def forward(self, x_local, pg):
        """
        x_local: [B, T_local, C]  (already sharded T/n chunk)
        Returns: [B, T_local, C]  residual output (local chunk only)
        """
        B_in, T_in, C = x_local.shape
        h   = self.ln(x_local)
        qkv = self.c_attn(h)
        q, k, v = qkv.split(N_EMBD, dim=2)
        q = q.view(B_in, T_in, N_HEAD, HEAD_DIM).transpose(1, 2)
        k = k.view(B_in, T_in, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B_in, T_in, N_HEAD, HEAD_DIM).transpose(1, 2)

        # Gather full Q/K/V across ranks for ring attention
        q_chunks = [torch.empty_like(q) for _ in range(world_size)]
        k_chunks = [torch.empty_like(k) for _ in range(world_size)]
        v_chunks = [torch.empty_like(v) for _ in range(world_size)]
        dist.all_gather(q_chunks, q.contiguous(), group=pg)
        dist.all_gather(k_chunks, k.contiguous(), group=pg)
        dist.all_gather(v_chunks, v.contiguous(), group=pg)
        q_full = torch.cat(q_chunks, dim=2)
        k_full = torch.cat(k_chunks, dim=2)
        v_full = torch.cat(v_chunks, dim=2)

        attn_full = manual_ring_forward(q_full, k_full, v_full, USE_LB, pg, SCALE)

        attn_local = attn_full[:, :, rank * T_local : (rank + 1) * T_local, :]
        y = attn_local.transpose(1, 2).contiguous().view(B_in, T_in, C)
        return x_local + self.c_proj(y)


# ============================================================================
# Fixed synthetic batch  (same across both models)
# ============================================================================

torch.manual_seed(999)
x_full = torch.randn(B, T, N_EMBD, device=device)
dist.broadcast(x_full, src=0)

x_local = x_full[:, rank * T_local : (rank + 1) * T_local, :].contiguous()
x_local.requires_grad_(True)


# ============================================================================
# Build models
# ============================================================================

_cp_options.enable_load_balance = USE_LB
_cp_options.convert_to_f32      = True

ref_model    = RefAttention().to(device)
manual_model = ManualAttention().to(device)
manual_model.c_attn.weight.data.copy_(ref_model.c_attn.weight.data)
manual_model.c_attn.bias.data.copy_(ref_model.c_attn.bias.data)
manual_model.c_proj.weight.data.copy_(ref_model.c_proj.weight.data)
manual_model.c_proj.bias.data.copy_(ref_model.c_proj.bias.data)
manual_model.ln.weight.data.copy_(ref_model.ln.weight.data)
manual_model.ln.bias.data.copy_(ref_model.ln.bias.data)


# ============================================================================
# Section 0: Initial weight display
# ============================================================================

section("INITIAL WEIGHTS  (rank 0)")
display("REF   c_attn.weight", ref_model.c_attn.weight)
display("REF   c_proj.weight", ref_model.c_proj.weight)
display("MANUAL c_attn.weight", manual_model.c_attn.weight)
display("MANUAL c_proj.weight", manual_model.c_proj.weight)


# ============================================================================
# Section 1: Reference forward + backward  (context_parallel)
# ============================================================================

section("REFERENCE (PyTorch context_parallel) FORWARD")

buffers         = [x_full]
buffer_seq_dims = [1]
no_restore      = {x_full}

with context_parallel(
    cp_mesh,
    buffers=buffers,
    buffer_seq_dims=buffer_seq_dims,
    no_restore_buffers=no_restore,
):
    x_cp_local = x_full                     # sharded in-place by context_parallel
    ref_out_local = ref_model(x_cp_local)   # [B, T_local, C]

    # Simple proxy loss: mean of all outputs (sum-reduced across ranks)
    loss_ref = ref_out_local.mean()
    dist.all_reduce(loss_ref, op=dist.ReduceOp.SUM)
    loss_ref = loss_ref / world_size

display("REF   output_local", ref_out_local)

section("REFERENCE BACKWARD")
loss_ref.backward()

display("REF   c_attn.weight.grad", ref_model.c_attn.weight.grad)
display("REF   c_attn.bias.grad",   ref_model.c_attn.bias.grad)
display("REF   c_proj.weight.grad", ref_model.c_proj.weight.grad)
display("REF   c_proj.bias.grad",   ref_model.c_proj.bias.grad)


# ============================================================================
# Section 2: Manual forward + backward  (ring attention mirroring C++)
# ============================================================================

section("MANUAL (ring attention) FORWARD")

manual_out_local = manual_model(x_local, cp_group)   # [B, T_local, C]

loss_manual = manual_out_local.mean()
dist.all_reduce(loss_manual, op=dist.ReduceOp.SUM)
loss_manual = loss_manual / world_size

display("MANUAL output_local", manual_out_local)

section("MANUAL BACKWARD")
loss_manual.backward()

display("MANUAL c_attn.weight.grad", manual_model.c_attn.weight.grad)
display("MANUAL c_attn.bias.grad",   manual_model.c_attn.bias.grad)
display("MANUAL c_proj.weight.grad", manual_model.c_proj.weight.grad)
display("MANUAL c_proj.bias.grad",   manual_model.c_proj.bias.grad)


# ============================================================================
# Section 3: Diff summary
# ============================================================================

section("DIFF SUMMARY  (manual - ref,  rank 0 only)")

def _diff(name, ref_t, manual_t):
    if ref_t is None or manual_t is None:
        if rank == 0:
            print(f"  {name}: MISSING GRAD")
        return
    diff   = (manual_t - ref_t).float().abs()
    rel    = diff / (ref_t.float().abs().mean() + 1e-8)
    if rank == 0:
        print(f"  {name:35s}  abs_max={diff.max().item():.6e}  "
              f"abs_mean={diff.mean().item():.6e}  "
              f"rel_mean={rel.mean().item():.4f}")

if rank == 0:
    print()
_diff("output_local",        ref_out_local.detach(),                  manual_out_local.detach())
_diff("c_attn.weight.grad",  ref_model.c_attn.weight.grad,            manual_model.c_attn.weight.grad)
_diff("c_attn.bias.grad",    ref_model.c_attn.bias.grad,              manual_model.c_attn.bias.grad)
_diff("c_proj.weight.grad",  ref_model.c_proj.weight.grad,            manual_model.c_proj.weight.grad)
_diff("c_proj.bias.grad",    ref_model.c_proj.bias.grad,              manual_model.c_proj.bias.grad)

if rank == 0:
    print("\nDone.  Run with --lb=0 (no LB) first, then --lb=1 (HeadTail).")
    print("Compare REF vs MANUAL to find where our C++ diverges.\n")

dist.destroy_process_group()
