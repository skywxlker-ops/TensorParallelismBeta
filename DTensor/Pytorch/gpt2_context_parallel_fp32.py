"""
Context Parallel GPT-2 — Pure FP32 Manual Ring Attention
=========================================================
Identical to gpt2_context_parallel.py EXCEPT:
  • Uses SDPBackend.MATH (pure FP32, no BF16 casting)
  • Implements ring attention manually via NCCL send/recv
  • Does NOT depend on torch.distributed.tensor.experimental.context_parallel
  • Merges partial attention outputs using online softmax (LSE correction)

This is a direct FP32 counterpart to the C++ CP implementation.

Launch:
  torchrun --standalone --nnodes=1 --nproc-per-node=<N> gpt2_context_parallel_fp32.py
"""

import os
import sys
import math
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
# NVTX markers via torch.cuda.nvtx (built-in, no import needed)

from torch.nn.attention import sdpa_kernel, SDPBackend


# ══════════════════════════════════════════════════════════════════════════════
# Distributed init
# ══════════════════════════════════════════════════════════════════════════════

torch.distributed.init_process_group(backend="nccl")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

cp_world_size = torch.distributed.get_world_size()
cp_rank       = torch.distributed.get_rank()
cp_group      = torch.distributed.group.WORLD

device         = torch.device("cuda", local_rank)
master_process = (cp_rank == 0)

torch.manual_seed(1234)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
nsys_report = False

@dataclass
class GPTConfig:
    n_embd:       int  = 384
    block_size:   int  = 1024
    vocab_size:   int  = 50304
    n_layer:      int  = 3
    n_head:       int  = 6
    weight_tying: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# CUDA Timer
# ══════════════════════════════════════════════════════════════════════════════

class CudaTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def elapsed_seconds(self):
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event) / 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# Manual FP32 SDPA with LSE (for ring attention merging)
# ══════════════════════════════════════════════════════════════════════════════

def sdpa_with_lse(q, k, v, is_causal=False, q_offset=0, k_offset=0):
    """
    Scaled dot-product attention in FP32, also returns log-sum-exp for
    online softmax merging across ring attention steps.

    Args:
        q: [B, H, T_q, D]
        k: [B, H, T_k, D]
        v: [B, H, T_k, D]
        is_causal: apply causal mask (only when Q and K are from same chunk)
        q_offset: global sequence offset of Q chunk (for cross-chunk causal)
        k_offset: global sequence offset of K chunk (for cross-chunk causal)

    Returns:
        out: [B, H, T_q, D]
        lse: [B, H, T_q, 1]  -- log-sum-exp for merging
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    # [B, H, T_q, T_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        T_q, T_k = scores.size(-2), scores.size(-1)
        # Build causal mask using global positions
        q_pos = torch.arange(q_offset, q_offset + T_q, device=scores.device)
        k_pos = torch.arange(k_offset, k_offset + T_k, device=scores.device)
        # mask[i,j] = True where q_pos[i] >= k_pos[j] (allowed to attend)
        mask = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)  # [T_q, T_k]
        scores = scores.masked_fill(~mask, float('-inf'))

    # LSE for merging: [B, H, T_q, 1]
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)

    # Attention weights and output
    weights = torch.softmax(scores, dim=-1)
    # Replace NaN from all-masked rows (future chunks) with 0
    weights = torch.nan_to_num(weights, nan=0.0)
    out = torch.matmul(weights, v)

    return out, lse


def merge_attention(out_old, lse_old, out_new, lse_new):
    """
    Online softmax merge of two partial attention results.

    Given two partial results (out_old, lse_old) and (out_new, lse_new),
    computes the combined result as if attention was computed over both
    key-value sets simultaneously.

    Args:
        out_old: [B, H, T, D]
        lse_old: [B, H, T, 1]
        out_new: [B, H, T, D]
        lse_new: [B, H, T, 1]

    Returns:
        merged_out: [B, H, T, D]
        merged_lse: [B, H, T, 1]
    """
    # Numerically stable merge using max trick
    max_lse = torch.maximum(lse_old, lse_new)
    # Handle -inf (all-masked rows)
    max_lse = torch.clamp(max_lse, min=-1e30)

    exp_old = torch.exp(lse_old - max_lse)
    exp_new = torch.exp(lse_new - max_lse)
    sum_exp = exp_old + exp_new

    # Weighted combination
    merged_out = (exp_old * out_old + exp_new * out_new) / sum_exp
    merged_lse = max_lse + torch.log(sum_exp)

    return merged_out, merged_lse


# ══════════════════════════════════════════════════════════════════════════════
# Ring Attention (manual NCCL send/recv)
# ══════════════════════════════════════════════════════════════════════════════

def ring_attention(q, k, v, world_size, rank, group):
    """
    Manual ring attention implementation using NCCL send/recv.

    Each rank holds the local Q chunk. K,V rotate around the ring.
    For causal attention: skip future chunks, apply causal mask on self-chunk.

    Uses batch_isend_irecv with P2POp to avoid the NCCL communicator crash
    that occurs with raw isend/irecv on lazy-initialized process groups.

    Args:
        q: [B, H, T_local, D] -- local query
        k: [B, H, T_local, D] -- local key
        v: [B, H, T_local, D] -- local value
        world_size: number of CP ranks
        rank: current CP rank
        group: process group

    Returns:
        out: [B, H, T_local, D] -- merged attention output
    """
    T_local = q.size(2)

    # Running accumulators for online softmax merge
    merged_out = None
    merged_lse = None

    # Current K, V being processed (starts with local, make contiguous)
    curr_k = k.contiguous()
    curr_v = v.contiguous()

    for step in range(world_size):
        # Determine which rank's K,V we currently have
        source_rank = (rank - step) % world_size

        # --- Async send/recv for next step's K,V (overlap with compute) ---
        recv_k = None
        recv_v = None
        reqs = None
        if step < world_size - 1:
            # Contiguous recv buffers
            recv_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
            recv_v = torch.empty(v.shape, dtype=v.dtype, device=v.device)
            send_to   = (rank + 1) % world_size
            recv_from = (rank - 1) % world_size

            # Use batched P2P ops to avoid per-op NCCL communicator creation
            p2p_ops = [
                torch.distributed.P2POp(torch.distributed.isend, curr_k, send_to, group),
                torch.distributed.P2POp(torch.distributed.irecv, recv_k, recv_from, group),
                torch.distributed.P2POp(torch.distributed.isend, curr_v, send_to, group),
                torch.distributed.P2POp(torch.distributed.irecv, recv_v, recv_from, group),
            ]
            reqs = torch.distributed.batch_isend_irecv(p2p_ops)

        # --- Determine causal behavior for this ring step ---
        if source_rank == rank:
            # Self-chunk: apply causal mask
            use_causal = True
        elif source_rank > rank:
            # Future chunk: all K,V positions > all Q positions, skip
            # Wait for communication to complete before continuing
            if reqs is not None:
                for req in reqs:
                    req.wait()
            curr_k = recv_k if recv_k is not None else curr_k
            curr_v = recv_v if recv_v is not None else curr_v
            continue
        else:
            # Past chunk: full attention (no mask)
            use_causal = False

        # --- Compute SDPA with LSE ---
        q_offset = rank * T_local
        k_offset = source_rank * T_local

        step_out, step_lse = sdpa_with_lse(
            q, curr_k, curr_v,
            is_causal=use_causal,
            q_offset=q_offset,
            k_offset=k_offset,
        )

        # --- Merge into running accumulator ---
        if merged_out is None:
            merged_out = step_out
            merged_lse = step_lse
        else:
            merged_out, merged_lse = merge_attention(
                merged_out, merged_lse, step_out, step_lse
            )

        # --- Wait for communication and swap buffers ---
        if reqs is not None:
            for req in reqs:
                req.wait()
        if recv_k is not None:
            curr_k = recv_k
            curr_v = recv_v

    return merged_out


# ══════════════════════════════════════════════════════════════════════════════
# Attention (pure FP32 manual ring attention)
# ══════════════════════════════════════════════════════════════════════════════

class CPAttention(nn.Module):
    """
    Context-parallel causal self-attention — pure FP32.

    Uses manual ring attention loop with NCCL send/recv instead of
    PyTorch's context_parallel() dispatcher. Everything stays in fp32.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.ln     = nn.LayerNorm(config.n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)

        residual_std = 0.02 / math.sqrt(2.0 * config.n_layer)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=True)

        nn.init.normal_(self.c_attn.weight, std=0.02)
        nn.init.zeros_(self.c_attn.bias)
        nn.init.normal_(self.c_proj.weight, std=residual_std)
        nn.init.zeros_(self.c_proj.bias)

        self.t_attn = 0.0
        self._ts = torch.cuda.Event(enable_timing=True)
        self._te = torch.cuda.Event(enable_timing=True)

    def reset_t_attn(self):
        self.t_attn = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_local, C = x.size()

        h   = self.ln(x)
        qkv = self.c_attn(h)
        q, k, v = qkv.split(self.n_embd, dim=2)

        def to_heads(t):
            return t.view(B, T_local, self.n_head, self.head_dim).transpose(1, 2)

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        # ── FP32 Ring Attention via manual NCCL send/recv ──
        self._ts.record()
        if cp_world_size > 1:
            y = ring_attention(q, k, v, cp_world_size, cp_rank, cp_group)
        else:
            # Single-GPU: standard SDPA with MATH backend
            with sdpa_kernel(SDPBackend.MATH):
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
                )
        self._te.record()
        torch.cuda.synchronize()
        self.t_attn += self._ts.elapsed_time(self._te) / 1000.0

        # Merge heads -> [B, T_local, C]
        y = y.transpose(1, 2).contiguous().view(B, T_local, C)

        return x + self.c_proj(y)


# ══════════════════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        residual_std = 0.02 / math.sqrt(2.0 * config.n_layer)

        self.ln     = nn.LayerNorm(config.n_embd)
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)

        nn.init.normal_(self.c_fc.weight,   std=0.02)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.normal_(self.c_proj.weight, std=residual_std)
        nn.init.zeros_(self.c_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.c_fc(h)
        h = self.gelu(h)
        h = self.c_proj(h)
        return x + h


# ══════════════════════════════════════════════════════════════════════════════
# Transformer Block
# ══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CPAttention(config)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.cuda.nvtx.range_push("Attention")
        x = self.attn(x)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("MLP")
        x = self.mlp(x)
        torch.cuda.nvtx.range_pop()
        return x


# ══════════════════════════════════════════════════════════════════════════════
# GPT Model
# ══════════════════════════════════════════════════════════════════════════════

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.t_tok_emb = 0.0
        self.t_pos_emb = 0.0
        self.t_attn    = 0.0
        self.t_mlp     = 0.0
        self.t_ln_f    = 0.0
        self.t_lm_head = 0.0

        self.transformer = nn.ModuleDict({
            'wte':  nn.Embedding(config.vocab_size, config.n_embd),
            'wpe':  nn.Embedding(config.block_size, config.n_embd),
            'ln_f': nn.LayerNorm(config.n_embd),
            'h':    nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.weight_tying:
            self.lm_head.weight = self.transformer['wte'].weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.transformer['wte'].weight, std=0.02)
        nn.init.normal_(self.transformer['wpe'].weight, std=0.02)
        if not self.config.weight_tying:
            nn.init.normal_(self.lm_head.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def collect_attn_timing(self):
        self.t_attn = sum(b.attn.t_attn for b in self.transformer['h'])

    def reset_timing(self):
        self.t_tok_emb = 0.0
        self.t_pos_emb = 0.0
        self.t_attn    = 0.0
        self.t_mlp     = 0.0
        self.t_ln_f    = 0.0
        self.t_lm_head = 0.0
        for b in self.transformer['h']:
            b.attn.reset_t_attn()

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.size()
        assert T % cp_world_size == 0, \
            f"T={T} must be divisible by cp_world_size={cp_world_size}"
        chunk = T // cp_world_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos = pos.unsqueeze(0).expand(B, -1)

        _t  = torch.cuda.Event(enable_timing=True)
        _t2 = torch.cuda.Event(enable_timing=True)

        # ---- token embedding ----
        torch.cuda.nvtx.range_push("TokEmb")
        _t.record()
        tok_emb = self.transformer['wte'](idx)
        _t2.record(); torch.cuda.synchronize()
        self.t_tok_emb += _t.elapsed_time(_t2) / 1000.0
        torch.cuda.nvtx.range_pop()

        # ---- positional embedding ----
        torch.cuda.nvtx.range_push("PosEmb")
        _t.record()
        pos_emb = self.transformer['wpe'](pos)
        _t2.record(); torch.cuda.synchronize()
        self.t_pos_emb += _t.elapsed_time(_t2) / 1000.0
        torch.cuda.nvtx.range_pop()

        x = tok_emb + pos_emb

        # ---- scatter sequence to this rank's local chunk ----
        x_local = x[:, cp_rank * chunk:(cp_rank + 1) * chunk, :].contiguous()

        # ---- transformer blocks (FP32 ring attention) ----
        _t.record()
        for i, block in enumerate(self.transformer['h']):
            torch.cuda.nvtx.range_push(f"Block_{i}")
            x_local = block(x_local)
            torch.cuda.nvtx.range_pop()
        _t2.record(); torch.cuda.synchronize()
        self.t_mlp += _t.elapsed_time(_t2) / 1000.0

        # ---- final layer norm ----
        torch.cuda.nvtx.range_push("LN_Final")
        _t.record()
        x_local = self.transformer['ln_f'](x_local)
        _t2.record(); torch.cuda.synchronize()
        self.t_ln_f += _t.elapsed_time(_t2) / 1000.0
        torch.cuda.nvtx.range_pop()

        # ---- lm head ----
        torch.cuda.nvtx.range_push("LMHead")
        _t.record()
        logits = self.lm_head(x_local)
        _t2.record(); torch.cuda.synchronize()
        self.t_lm_head += _t.elapsed_time(_t2) / 1000.0
        torch.cuda.nvtx.range_pop()

        loss = None
        if targets is not None:
            targets_local = targets[:, cp_rank * chunk:(cp_rank + 1) * chunk].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets_local.view(-1),
            )
            torch.distributed.all_reduce(
                loss, op=torch.distributed.ReduceOp.SUM, group=cp_group
            )
            loss = loss / cp_world_size

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict     = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Data Loader
# ══════════════════════════════════════════════════════════════════════════════

def load_tokens(filename):
    npt = np.fromfile(filename, dtype=np.uint16).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:

    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        data_root = "/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/Data"
        shards = sorted(
            os.path.join(data_root, s)
            for s in os.listdir(data_root)
            if split in s
        )
        assert len(shards) > 0, f"No '{split}' shards found in {data_root}"
        self.shards = shards
        self.reset()

    def reset(self):
        self.current_shard    = 0
        self.tokens           = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y


# ══════════════════════════════════════════════════════════════════════════════
# Training Setup
# ══════════════════════════════════════════════════════════════════════════════

B = 4
T = 1024
total_batch_size = 65536

assert T % cp_world_size == 0, f"T={T} must be divisible by cp_world_size={cp_world_size}"
grad_accum_steps = total_batch_size // (B * T)

config = GPTConfig(vocab_size=50304, n_layer=3, n_head=6, weight_tying=False)

model = GPT(config)
model.to(device)

num_params         = sum(p.numel() for p in model.parameters())
num_params_per_gpu = num_params

# nsys profile -t cuda -o my_report ./your_executable && nsys stats --report cuda_gpu_kern_sum:base --format csv -o my_custom_report my_report.nsys-rep


max_steps    = 6768

if nsys_report == True:
    max_steps = 1

warmup_steps = max_steps // 10

max_lr = 6e-4
min_lr = max_lr * 0.1

VAL_FREQ = 100

if master_process:
    print("=== GPT-2 Context Parallel Training Script (FP32 Manual Ring Attention) ===")
    print(f"Configuration:")
    print(f"  vocab_size:     {config.vocab_size}")
    print(f"  context_length: {config.block_size}")
    print(f"  n_embd:         {config.n_embd}")
    print(f"  n_layers:       {config.n_layer}")
    print(f"  n_heads:        {config.n_head}")
    print(f"  B={B}, T={T}")
    print(f"  cp_world_size:  {cp_world_size}")
    print(f"  global_batch:   {total_batch_size}")
    print(f"  grad_accum_steps: {grad_accum_steps}")
    print(f"  Weight Tying:   {'ENABLED' if config.weight_tying else 'DISABLED'}")
    print(f"  Parameters:          {num_params}")
    print(f"  Parameters per GPU:  {num_params_per_gpu}")
    print(f"  max_steps:      {max_steps}")
    print(f"  warmup_steps:   {warmup_steps}")
    print(f"  SDPA Backend:   MATH (pure FP32)")

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr)


# ══════════════════════════════════════════════════════════════════════════════
# LR Schedule
# ══════════════════════════════════════════════════════════════════════════════

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ══════════════════════════════════════════════════════════════════════════════
# Data Loaders
# ══════════════════════════════════════════════════════════════════════════════

train_loader = DataLoaderLite(B, T, "train")
val_loader   = DataLoaderLite(B, T, "val")


# ══════════════════════════════════════════════════════════════════════════════
# CSV Logging Setup
# ══════════════════════════════════════════════════════════════════════════════

log_file        = None
log_filename    = ""
config_filename = ""

if master_process:
    os.makedirs("Pytorch_CP_FP32_Training_logs", exist_ok=True)
    log_idx = 1
    while True:
        log_filename = f"Pytorch_CP_FP32_Training_logs/Pytorch_CP_FP32_Training_log{log_idx}.csv"
        if not os.path.exists(log_filename):
            break
        log_idx += 1

    print(f"Saving logs to: {log_filename}")

    config_filename = f"Pytorch_CP_FP32_Training_logs/Pytorch_CP_FP32_Training_log{log_idx}_config.txt"
    with open(config_filename, 'w') as cf:
        cf.write("Configuration:\n")
        cf.write(f"  Batch_size: {B}\n")
        cf.write(f"  context_length: {config.block_size}\n")
        cf.write(f"  n_embd: {config.n_embd}\n")
        cf.write(f"  n_heads: {config.n_head}\n")
        cf.write(f"  vocab_size: {config.vocab_size}\n")
        cf.write(f"  n_layers: {config.n_layer}\n")
        cf.write(f"  global_batch: {total_batch_size}\n")
        cf.write(f"  grad_accum_steps: {grad_accum_steps}\n")
        cf.write(f"  cp_world_size: {cp_world_size}\n")
        cf.write(f"  Parameters: {num_params}\n")
        cf.write(f"  Parameters per GPU: {num_params_per_gpu}\n")
        cf.write(f"  Max Learning Rate: {max_lr}\n")
        cf.write(f"  Min Learning Rate: {min_lr}\n")
        cf.write(f"  max_steps: {max_steps}\n")
        cf.write(f"  warmup_steps: {warmup_steps}\n")
        cf.write(f"  SDPA Backend: MATH (pure FP32)\n")

    log_file = open(log_filename, 'w', newline='')
    log_file.write(
        "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec,"
        "timer_data,timer_fwd,timer_loss,timer_bwd,timer_clip,timer_optim,"
        "timer_tok_emb,timer_pos_emb,timer_attn_cp,timer_mlp,timer_ln_f,timer_lm_head\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step Timers
# ══════════════════════════════════════════════════════════════════════════════

timer_step       = CudaTimer()
timer_data       = CudaTimer()
timer_fwd        = CudaTimer()
timer_loss_timer = CudaTimer()
timer_bwd        = CudaTimer()
timer_clip       = CudaTimer()
timer_optim      = CudaTimer()


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

if master_process:
    print("\nStarting training...")

val_loss_accum_log = -1.0

for step in range(max_steps):

    timer_step.start()

    # ---- Validation ----
    if step % VAL_FREQ == 0 or step == max_steps - 1:
        model.eval()
        val_loader.reset()
        val_loss_accum = 0.0
        val_loss_steps = 5

        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x = x.to(device)
                y = y.to(device)
                _, loss = model(x, y)
                val_loss_accum += loss.item() / val_loss_steps

        if master_process:
            print(f"validation loss: {val_loss_accum:.4f}")
        val_loss_accum_log = val_loss_accum

    # ---- Training ----
    model.train()
    optimizer.zero_grad()

    loss_accum    = 0.0
    time_data     = 0.0
    time_forward  = 0.0
    time_loss     = 0.0
    time_backward = 0.0
    model.reset_timing()

    for micro_step in range(grad_accum_steps):

        torch.cuda.nvtx.range_push("DataLoad")
        timer_data.start()
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        time_data += timer_data.elapsed_seconds()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Forward")
        timer_fwd.start()
        logits, _ = model(x, None)
        time_forward += timer_fwd.elapsed_seconds()
        torch.cuda.nvtx.range_pop()

        # ---- local loss + all_reduce ----
        torch.cuda.nvtx.range_push("Loss")
        timer_loss_timer.start()
        chunk   = T // cp_world_size
        y_local = y[:, cp_rank * chunk:(cp_rank + 1) * chunk].contiguous()
        loss    = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_local.view(-1),
        )
        torch.distributed.all_reduce(
            loss, op=torch.distributed.ReduceOp.SUM, group=cp_group
        )
        loss = loss / cp_world_size
        time_loss += timer_loss_timer.elapsed_seconds()
        torch.cuda.nvtx.range_pop()

        loss = loss / grad_accum_steps
        loss_accum += loss.detach().item()

        torch.cuda.nvtx.range_push("Backward")
        timer_bwd.start()
        loss.backward()
        time_backward += timer_bwd.elapsed_seconds()
        torch.cuda.nvtx.range_pop()

    model.collect_attn_timing()

    # ---- Gradient Clipping ----
    torch.cuda.nvtx.range_push("GradClip")
    timer_clip.start()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    time_clip = timer_clip.elapsed_seconds()
    torch.cuda.nvtx.range_pop()

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # ---- Optimizer Step ----
    torch.cuda.nvtx.range_push("Optimizer")
    timer_optim.start()
    optimizer.step()
    time_optim = timer_optim.elapsed_seconds()
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    dt = timer_step.elapsed_seconds()

    tokens_processed = B * T * grad_accum_steps
    tokens_per_sec   = tokens_processed / dt

    total_sec = int((max_steps - step) * dt)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60

    if master_process:
        print(
            f"step {step:5d} | loss: {loss_accum:.6f} | lr {lr:.4e} "
            f"| norm: {norm:.4f} | dt: {dt * 1000.0:.2f}ms "
            f"| tok/sec: {tokens_per_sec:.1f} "
            f"| Time Left: {h:02d} hrs : {m:02d} mins"
        )
        print(
            f"  [TIMING] data: {time_data * 1000.0:.1f}ms"
            f" | fwd: {time_forward * 1000.0:.1f}ms"
            f" | loss: {time_loss * 1000.0:.1f}ms"
            f" | bwd: {time_backward * 1000.0:.1f}ms"
            f" | clip: {time_clip * 1000.0:.1f}ms"
            f" | optim: {time_optim * 1000.0:.1f}ms"
        )
        print(
            f"  [LAYER] tok_emb: {model.t_tok_emb * 1000.0:.1f}ms"
            f" | pos_emb: {model.t_pos_emb * 1000.0:.1f}ms"
            f" | attn_cp: {model.t_attn * 1000.0:.1f}ms"
            f" | mlp: {model.t_mlp * 1000.0:.1f}ms"
            f" | ln_f: {model.t_ln_f * 1000.0:.1f}ms"
            f" | lm_head: {model.t_lm_head * 1000.0:.1f}ms"
        )

        if log_file:
            log_file.write(
                f"{step},{loss_accum:.6f},{val_loss_accum_log:.6f},"
                f"{lr},{norm},{dt * 1000.0},{tokens_per_sec},"
                f"{time_data * 1000.0},{time_forward * 1000.0},"
                f"{time_loss * 1000.0},{time_backward * 1000.0},"
                f"{time_clip * 1000.0},{time_optim * 1000.0},"
                f"{model.t_tok_emb * 1000.0},{model.t_pos_emb * 1000.0},"
                f"{model.t_attn * 1000.0},{model.t_mlp * 1000.0},"
                f"{model.t_ln_f * 1000.0},{model.t_lm_head * 1000.0}\n"
            )
            log_file.flush()

    val_loss_accum_log = -1.0


# ══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════════════════════════════════════

if master_process:
    if log_file:
        log_file.close()
    print(f"\nTraining log saved to: {log_filename}")
    print("\n=== FP32 Context Parallel Training Complete ===")
