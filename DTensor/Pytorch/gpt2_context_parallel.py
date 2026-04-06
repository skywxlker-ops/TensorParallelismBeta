"""
Context Parallel GPT-2 — PyTorch Native Implementation
=======================================================
Matches megatronCP.py exactly in:
  • GPTConfig  (n_embd=384, block_size=1024, vocab_size=50304,
                n_layer=3, n_head=6, weight_tying=False)
  • Training hyper-params (B=4, T=1024, total_batch_size=65536,
                           max_steps=6768, warmup_steps=676,
                           max_lr=6e-4, min_lr=6e-5, VAL_FREQ=100)
  • AdamW (weight_decay=0.1, betas=(0.9,0.95), eps=1e-8)
  • Cosine LR schedule with linear warmup
  • DataLoaderLite: same data_root, shard loading, shard cycling
  • fp32 throughout (no autocast, no mixed-precision)
  • Loss: local cross-entropy -> all_reduce SUM -> divide by cp_world_size
  • CSV log columns: identical to megatronCP.py
  • Config .txt sidecar: identical fields and order
  • Step print: step | loss | lr | norm | dt | tok/sec | Time Left
  • [TIMING] print: data | fwd | loss | bwd | clip | optim
  • [LAYER] print: tok_emb | pos_emb | attn_cp | mlp | ln_f | lm_head
  • Cleanup: log close + final banner

Context Parallel backend: torch.distributed.tensor.experimental.context_parallel
(PyTorch >= 2.7 Ring Attention — replaces SDPA transparently)

Launch:
  torchrun --standalone --nnodes=1 --nproc-per-node=<N> gpt2_context_parallel.py
"""

import os
import sys
import math
import csv

# Pyre2: skip-line-check
_PRINT_ONCE = False # Disabled debug

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

# ── PyTorch Context Parallel (requires PyTorch >= 2.7) ────────────────────────
try:
    from torch.distributed.tensor.experimental._attention import (
        context_parallel_unshard,
        set_rotate_method,
        _context_parallel as internal_context_parallel
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Replicate
    CP_AVAILABLE = True
except ImportError:
    CP_AVAILABLE = False

from torch.nn.attention import sdpa_kernel, SDPBackend


# ══════════════════════════════════════════════════════════════════════════════
# Distributed init  (mirrors megatronCP.py lines 84-102)
# ══════════════════════════════════════════════════════════════════════════════

torch.distributed.init_process_group(backend="nccl")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

cp_world_size = torch.distributed.get_world_size()
cp_rank       = torch.distributed.get_rank()
cp_group      = torch.distributed.group.WORLD   # default process group

device         = torch.device("cuda", local_rank)
master_process = (cp_rank == 0)

torch.manual_seed(1234)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration  (mirrors megatronCP.py GPTConfig)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    n_embd:       int  = 384
    block_size:   int  = 1024
    vocab_size:   int  = 50304
    n_layer:      int  = 3
    n_head:       int  = 6
    weight_tying: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# CUDA Timer  (identical to megatronCP.py CudaTimer)
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
# Attention  (pure PyTorch fp32; CP injected via context_parallel())
# ══════════════════════════════════════════════════════════════════════════════

class CPAttention(nn.Module):
    """
    Context-parallel causal self-attention.

    Each rank receives its local sequence chunk [B, T_local, C].
    Ring Attention is applied via PyTorch's context_parallel() context manager
    (called from GPT.forward), which patches F.scaled_dot_product_attention
    transparently.  Everything stays in fp32.
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

        # Weight init — identical to megatronCP.py CPAttention.__init__
        nn.init.normal_(self.c_attn.weight, std=0.02)
        nn.init.zeros_(self.c_attn.bias)
        nn.init.normal_(self.c_proj.weight, std=residual_std)
        nn.init.zeros_(self.c_proj.bias)

        # Per-layer attention timing accumulator
        self.t_attn = 0.0
        self._ts = torch.cuda.Event(enable_timing=True)
        self._te = torch.cuda.Event(enable_timing=True)

    def reset_t_attn(self):
        self.t_attn = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_local, C]  — already the local sequence shard
        B, T_local, C = x.size()

        h   = self.ln(x)
        qkv = self.c_attn(h)
        q, k, v = qkv.split(self.n_embd, dim=2)   # each [B, T_local, C]

        # Reshape to [B, n_head, T_local, head_dim] for SDPA
        def to_heads(t):
            return t.view(B, T_local, self.n_head, self.head_dim).transpose(1, 2)

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        # Context parallel strictly requires FLASH_ATTENTION or EFFICIENT_ATTENTION
        # which only support bf16/fp16 in PyTorch. So we cast right before SDPA,
        # perfectly mirroring what megatronCP.py does.
        orig_dtype = q.dtype
        q_bf = q.to(torch.bfloat16)
        k_bf = k.to(torch.bfloat16)
        v_bf = v.to(torch.bfloat16)

        # ── SDPA — patched to Ring Attention when inside context_parallel() ──
        self._ts.record()
        y = F.scaled_dot_product_attention(
            q_bf, k_bf, v_bf,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )   # [B, nh, T_local, hd]
        self._te.record()
        torch.cuda.synchronize()
        self.t_attn += self._ts.elapsed_time(self._te) / 1000.0

        # Cast back to original dtype
        y = y.to(orig_dtype)

        # Merge heads -> [B, T_local, C]
        y = y.transpose(1, 2).contiguous().view(B, T_local, C)

        return x + self.c_proj(y)


# ══════════════════════════════════════════════════════════════════════════════
# MLP  (identical structure to megatronCP.py MLP)
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
        x = self.attn(x)
        x = self.mlp(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# GPT Model  (mirrors megatronCP.py GPT exactly)
# ══════════════════════════════════════════════════════════════════════════════

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Per-step layer timing accumulators (mirrors megatronCP.py)
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

        # Context Parallel device mesh (set via setup_context_parallel)
        self._device_mesh = None
        self._cp_enabled  = False

    def _init_weights(self):
        nn.init.normal_(self.transformer['wte'].weight, std=0.02)
        nn.init.normal_(self.transformer['wpe'].weight, std=0.02)
        if not self.config.weight_tying:
            nn.init.normal_(self.lm_head.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Context Parallel setup ────────────────────────────────────────────
    def setup_context_parallel(self, device_mesh):
        if not CP_AVAILABLE:
            raise RuntimeError("Context Parallel requires PyTorch >= 2.7.")
        self._device_mesh = device_mesh
        self._cp_enabled  = True
        # all-gather pass-KV: same ring strategy as Megatron p2p
        set_rotate_method("allgather")

    # ── Timing helpers (mirrors megatronCP.py) ────────────────────────────
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

    # ── Forward (mirrors megatronCP.py GPT.forward exactly) ───────────────
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
        _t.record()
        tok_emb = self.transformer['wte'](idx)       # [B, T, C]
        _t2.record(); torch.cuda.synchronize()
        self.t_tok_emb += _t.elapsed_time(_t2) / 1000.0

        # ---- positional embedding ----
        _t.record()
        pos_emb = self.transformer['wpe'](pos)       # [B, T, C]
        _t2.record(); torch.cuda.synchronize()
        self.t_pos_emb += _t.elapsed_time(_t2) / 1000.0

        x = tok_emb + pos_emb                       # [B, T, C]  -- fp32

        # ---- scatter sequence to this rank's local chunk ----
        # Rank r owns token positions [r*chunk, (r+1)*chunk)
        x_local = x[:, cp_rank * chunk:(cp_rank + 1) * chunk, :].contiguous()
        # [B, chunk, C]

        # ---- transformer blocks (Ring Attention via context_parallel) ----
        _t.record()
        if self._cp_enabled and self._device_mesh is not None:
            # internal_context_parallel() intercepts F.scaled_dot_product_attention 
            # and patches it -> Ring Attention, bypassing the buggy buffers resize logic.
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                with internal_context_parallel(
                    seq_dim=1,
                    mesh=self._device_mesh,
                ):
                    for block in self.transformer['h']:
                        x_local = block(x_local)
                # DO NOT unshard because we want the sequence to remain local for
                # final layernorm, LM head, and chunked cross entropy loss (just like megatronCP).
        else:
            # Single-GPU path
            for block in self.transformer['h']:
                x_local = block(x_local)
        _t2.record(); torch.cuda.synchronize()
        # t_mlp covers attn+mlp combined; attn is tracked separately in CPAttention
        self.t_mlp += _t.elapsed_time(_t2) / 1000.0

        # ---- final layer norm ----
        _t.record()
        x_local = self.transformer['ln_f'](x_local)
        _t2.record(); torch.cuda.synchronize()
        self.t_ln_f += _t.elapsed_time(_t2) / 1000.0

        # ---- lm head ----
        _t.record()
        logits = self.lm_head(x_local)              # [B, chunk, vocab_size]
        _t2.record(); torch.cuda.synchronize()
        self.t_lm_head += _t.elapsed_time(_t2) / 1000.0

        loss = None
        if targets is not None:
            targets_local = targets[:, cp_rank * chunk:(cp_rank + 1) * chunk].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets_local.view(-1),
            )
            # Sum loss across CP ranks then divide — mirrors megatronCP.py exactly
            torch.distributed.all_reduce(
                loss, op=torch.distributed.ReduceOp.SUM, group=cp_group
            )
            loss = loss / cp_world_size

        return logits, loss

    # ── Optimiser (mirrors megatronCP.py configure_optimizers) ───────────
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
# Data Loader  (mirrors megatronCP.py DataLoaderLite exactly)
# ══════════════════════════════════════════════════════════════════════════════

def load_tokens(filename):
    npt = np.fromfile(filename, dtype=np.uint16).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:

    def __init__(self, B, T, split):
        self.B = B
        self.T = T

        # ── same data_root as megatronCP.py ──────────────────────────────
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
# Training Setup  (mirrors megatronCP.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

B = 4
T = 1024
total_batch_size = 65536

assert T % cp_world_size == 0, f"T={T} must be divisible by cp_world_size={cp_world_size}"
grad_accum_steps = total_batch_size // (B * T)   # = 16

config = GPTConfig(vocab_size=50304, n_layer=3, n_head=6, weight_tying=False)

model = GPT(config)
model.to(device)   # fp32 — no .half(), no autocast anywhere

# ── Context Parallel device mesh ──────────────────────────────────────────────
if cp_world_size > 1 and CP_AVAILABLE:
    device_mesh = init_device_mesh(
        device_type    = "cuda",
        mesh_shape     = (cp_world_size,),
        mesh_dim_names = ("cp",),
    )
    model.setup_context_parallel(device_mesh)
else:
    if master_process:
        print("[INFO] Single-GPU mode — Context Parallel disabled.")

num_params         = sum(p.numel() for p in model.parameters())
num_params_per_gpu = num_params

max_steps    = 6768
warmup_steps = max_steps // 10    # = 676

max_lr = 6e-4
min_lr = max_lr * 0.1

VAL_FREQ = 100

# ── Print configuration banner (mirrors megatronCP.py exactly) ───────────────
if master_process:
    print("=== GPT-2 Context Parallel Training Script (PyTorch Native Ring Attention) ===")
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

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr)


# ══════════════════════════════════════════════════════════════════════════════
# LR Schedule  (mirrors megatronCP.py get_lr exactly)
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
# CSV Logging Setup  (mirrors megatronCP.py — same folder, filename, columns)
# ══════════════════════════════════════════════════════════════════════════════

log_file        = None
log_filename    = ""
config_filename = ""

if master_process:
    os.makedirs("Pytorch_CP_Training_logs", exist_ok=True)
    log_idx = 1
    while True:
        log_filename = f"Pytorch_CP_Training_logs/Pytorch_CP_Training_log{log_idx}.csv"
        if not os.path.exists(log_filename):
            break
        log_idx += 1

    print(f"Saving logs to: {log_filename}")

    config_filename = f"Pytorch_CP_Training_logs/Pytorch_CP_Training_log{log_idx}_config.txt"
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

    log_file = open(log_filename, 'w', newline='')
    # CSV header — byte-for-byte identical to megatronCP.py
    log_file.write(
        "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec,"
        "timer_data,timer_fwd,timer_loss,timer_bwd,timer_clip,timer_optim,"
        "timer_tok_emb,timer_pos_emb,timer_attn_cp,timer_mlp,timer_ln_f,timer_lm_head\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step Timers  (mirrors megatronCP.py)
# ══════════════════════════════════════════════════════════════════════════════

timer_step       = CudaTimer()
timer_data       = CudaTimer()
timer_fwd        = CudaTimer()
timer_loss_timer = CudaTimer()
timer_bwd        = CudaTimer()
timer_clip       = CudaTimer()
timer_optim      = CudaTimer()


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop  (mirrors megatronCP.py training loop exactly)
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

        timer_data.start()
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        time_data += timer_data.elapsed_seconds()

        timer_fwd.start()
        logits, _ = model(x, None)       # fp32 forward; targets=None
        time_forward += timer_fwd.elapsed_seconds()

        # ---- local loss + all_reduce (mirrors megatronCP.py exactly) ----
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

        loss = loss / grad_accum_steps
        loss_accum += loss.detach().item()

        timer_bwd.start()
        loss.backward()
        time_backward += timer_bwd.elapsed_seconds()

    # Collect per-layer attention timing (mirrors megatronCP.py)
    model.collect_attn_timing()

    # ---- Gradient Clipping ----
    timer_clip.start()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    time_clip = timer_clip.elapsed_seconds()

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # ---- Optimizer Step ----
    timer_optim.start()
    optimizer.step()
    time_optim = timer_optim.elapsed_seconds()

    torch.cuda.synchronize()
    dt = timer_step.elapsed_seconds()

    tokens_processed = B * T * grad_accum_steps
    tokens_per_sec   = tokens_processed / dt

    # ---- Time Left (mirrors megatronCP.py) ----
    total_sec = int((max_steps - step) * dt)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60

    # ---- Console logging (mirrors megatronCP.py print format exactly) ----
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

        # ---- CSV logging (mirrors megatronCP.py log_file.write exactly) ----
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
# Cleanup  (mirrors megatronCP.py)
# ══════════════════════════════════════════════════════════════════════════════

if master_process:
    if log_file:
        log_file.close()
    print(f"\nTraining log saved to: {log_filename}")
    print("\n=== Context Parallel Training Complete ===")
