#!/usr/bin/env python3
"""
GPT-2 Tensor Parallel Training — PyTorch Reference

Mirrors gpt2_tp_test.cpp architecture exactly:
  - Megatron-style TP: column-parallel QKV/fc1, row-parallel c_proj/fc2
  - PaLM-style parallel blocks: x = x + attn(ln1(x)) + mlp(ln2(x))
  - Weight tying: lm_head shares wte.weight
  - Same hyperparameters, optimizer, LR schedule

Launch:
  torchrun --nproc_per_node=2 gpt2_tp_pytorch.py
"""

import os, math, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# =============================================================================
# Configuration (matches C++ exactly)
# =============================================================================

CONFIG = dict(
    vocab_size     = 50304,
    context_length = 1024,
    n_embd         = 384,
    n_head         = 6,
    n_layers       = 3,
)

B               = 8
T               = 1024
GLOBAL_BATCH    = 65536
GRAD_ACCUM      = GLOBAL_BATCH // (B * T)   # 8
MAX_LR          = 6e-4
MIN_LR          = MAX_LR * 0.1
WARMUP_STEPS    = 174
MAX_STEPS       = 1738
DATA_ROOT       = "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Data"

# =============================================================================
# CUDA Timer (matches C++ CudaTimer — synchronizes on query)
# =============================================================================

class CudaTimer:
    """
    Records a CUDA event on start(), then on elapsed_sec() records a stop
    event and calls cudaEventSynchronize — identical to the C++ CudaTimer.

    This DOES introduce a GPU sync on every elapsed query, serializing the
    pipeline.  It is intentional so the per-component numbers are directly
    comparable to the C++ output.  See the note at the bottom of this file
    for an event-only approach that avoids mid-forward syncs.
    """
    def __init__(self):
        self.t0 = torch.cuda.Event(enable_timing=True)
        self.t1 = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.t0.record()

    def elapsed_ms(self):
        self.t1.record()
        self.t1.synchronize()            # <-- blocks CPU until GPU catches up
        return self.t0.elapsed_time(self.t1)

    def elapsed_sec(self):
        return self.elapsed_ms() / 1000.0

# =============================================================================
# Tensor-Parallel Autograd Primitives
# =============================================================================

class _CopyToParallelRegion(torch.autograd.Function):
    """Forward: identity.  Backward: AllReduce(sum)."""
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad

class _ReduceFromParallelRegion(torch.autograd.Function):
    """Forward: AllReduce(sum).  Backward: identity."""
    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x
    @staticmethod
    def backward(ctx, grad):
        return grad

# =============================================================================
# Column-Parallel & Row-Parallel Linear
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """Shard output dim across GPUs.  weight: [out_local, in]"""
    def __init__(self, in_features, out_features, bias=True, init_std=0.02):
        super().__init__()
        ws = dist.get_world_size()
        self.out_local = out_features // ws
        self.weight = nn.Parameter(torch.empty(self.out_local, in_features, device='cuda'))
        nn.init.normal_(self.weight, std=init_std)
        self.bias = nn.Parameter(torch.zeros(self.out_local, device='cuda')) if bias else None

    def forward(self, x):
        x = _CopyToParallelRegion.apply(x)
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Shard input dim across GPUs.  weight: [out, in_local]"""
    def __init__(self, in_features, out_features, bias=True, init_std=0.02):
        super().__init__()
        ws = dist.get_world_size()
        self.in_local = in_features // ws
        self.weight = nn.Parameter(torch.empty(out_features, self.in_local, device='cuda'))
        nn.init.normal_(self.weight, std=init_std)
        self.bias = nn.Parameter(torch.zeros(out_features, device='cuda')) if bias else None

    def forward(self, x):
        out = F.linear(x, self.weight)
        out = _ReduceFromParallelRegion.apply(out)
        if self.bias is not None:
            out = out + self.bias
        return out

# =============================================================================
# Attention (column-parallel QKV, row-parallel output projection)
# =============================================================================

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ws = dist.get_world_size()
        n_embd, n_head, n_layers = cfg['n_embd'], cfg['n_head'], cfg['n_layers']
        self.n_head_local = n_head // ws
        self.head_dim = n_embd // n_head
        res_scale = 1.0 / math.sqrt(2.0 * n_layers)

        self.c_attn = ColumnParallelLinear(n_embd, 3 * n_embd, bias=True, init_std=0.02)
        self.c_proj = RowParallelLinear(n_embd, n_embd, bias=True, init_std=0.02 * res_scale)

    def forward(self, x):
        B, T, _ = x.shape
        nh, hd = self.n_head_local, self.head_dim
        C_local = nh * hd

        qkv = self.c_attn(x)                                      # [B, T, 3*C_local]
        q, k, v = qkv.split(C_local, dim=2)

        q = q.view(B, T, nh, hd).transpose(1, 2).contiguous()     # [B, nh, T, hd]
        k = k.view(B, T, nh, hd).transpose(1, 2).contiguous()
        v = v.view(B, T, nh, hd).transpose(1, 2).contiguous()

        # Scaled dot-product attention (manual, matching C++)
        scale = 1.0 / math.sqrt(hd)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale        # [B, nh, T, T]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, v)                                 # [B, nh, T, hd]
        out = out.transpose(1, 2).contiguous().view(B, T, C_local) # [B, T, C_local]
        return self.c_proj(out)

# =============================================================================
# MLP (column-parallel fc1, row-parallel fc2)
# =============================================================================

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_embd, n_layers = cfg['n_embd'], cfg['n_layers']
        res_scale = 1.0 / math.sqrt(2.0 * n_layers)
        self.fc1 = ColumnParallelLinear(n_embd, 4 * n_embd, bias=True, init_std=0.02)
        self.fc2 = RowParallelLinear(4 * n_embd, n_embd, bias=True, init_std=0.02 * res_scale)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

# =============================================================================
# GPT Model
# =============================================================================

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_embd, n_layers = cfg['n_embd'], cfg['n_layers']

        # Replicated embeddings
        self.wte = nn.Embedding(cfg['vocab_size'], n_embd)
        self.wpe = nn.Embedding(cfg['context_length'], n_embd)
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)

        # Transformer layers
        self.ln1  = nn.ModuleList([nn.LayerNorm(n_embd) for _ in range(n_layers)])
        self.ln2  = nn.ModuleList([nn.LayerNorm(n_embd) for _ in range(n_layers)])
        self.attn = nn.ModuleList([Attention(cfg) for _ in range(n_layers)])
        self.mlp  = nn.ModuleList([MLP(cfg) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)

        # Weight tying — lm_head reuses wte.weight (no own params)
        # F.linear(x, wte.weight) computes x @ wte.weight.T = [B,T,C] @ [C,V] = [B,T,V]

        # Cached position indices
        self.register_buffer('pos_ids', torch.arange(cfg['context_length']).unsqueeze(0))

        # Timers (matching C++ CudaTimer pattern)
        self._timers = {k: CudaTimer() for k in
                        ['tok_emb', 'pos_emb', 'attn', 'mlp', 'ln_f', 'lm_head']}
        self._times  = {k: 0.0 for k in self._timers}

    def reset_timing(self):
        for k in self._times:
            self._times[k] = 0.0

    def print_timing(self):
        t = self._times
        print(f"  [LAYER] tok_emb: {t['tok_emb']*1000:.1f}ms"
              f" | pos_emb: {t['pos_emb']*1000:.1f}ms"
              f" | attn: {t['attn']*1000:.1f}ms"
              f" | mlp: {t['mlp']*1000:.1f}ms"
              f" | ln_f: {t['ln_f']*1000:.1f}ms"
              f" | lm_head: {t['lm_head']*1000:.1f}ms")

    def forward(self, idx):
        B, T = idx.shape
        tm, tt = self._timers, self._times

        tm['tok_emb'].start()
        tok_emb = self.wte(idx)
        tt['tok_emb'] += tm['tok_emb'].elapsed_sec()

        tm['pos_emb'].start()
        pos_emb = self.wpe(self.pos_ids[:, :T])
        tt['pos_emb'] += tm['pos_emb'].elapsed_sec()

        x = tok_emb + pos_emb

        for i in range(self.cfg['n_layers']):
            # -- attn timer: ln1 + ln2 + attention (matches C++ timer_attn scope) --
            tm['attn'].start()
            h_attn = self.ln1[i](x)
            h_mlp  = self.ln2[i](x)
            attn_out = self.attn[i](h_attn)
            tt['attn'] += tm['attn'].elapsed_sec()

            # -- mlp timer: mlp + residual --
            tm['mlp'].start()
            mlp_out = self.mlp[i](h_mlp)
            x = x + attn_out + mlp_out          # PaLM parallel residual
            tt['mlp'] += tm['mlp'].elapsed_sec()

        tm['ln_f'].start()
        x = self.ln_f(x)
        tt['ln_f'] += tm['ln_f'].elapsed_sec()

        # lm_head: weight-tied with wte
        tm['lm_head'].start()
        logits = F.linear(x, self.wte.weight)   # [B,T,V]
        tt['lm_head'] += tm['lm_head'].elapsed_sec()

        return logits

# =============================================================================
# Data Loader (reads same binary uint16 shards as C++)
# =============================================================================

class DataLoaderLite:
    def __init__(self, B, T, split, data_root, device):
        self.B, self.T, self.device = B, T, device
        self.shards = sorted(glob.glob(os.path.join(data_root, f"edufineweb_{split}_*.bin")))
        assert self.shards, f"No {split} shards found in {data_root}"
        self.reset()

    def reset(self):
        self.shard_idx = 0
        self._load_shard(0)
        self.pos = 0

    def _load_shard(self, idx):
        self.tokens = np.memmap(self.shards[idx], dtype=np.uint16, mode='r')

    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.from_numpy(self.tokens[self.pos : self.pos + B*T + 1].astype(np.int64))
        x = buf[:-1].view(B, T).to(self.device)
        y = buf[1:].view(B, T).to(self.device)
        self.pos += B * T
        if self.pos + B * T + 1 > len(self.tokens):
            self.shard_idx = (self.shard_idx + 1) % len(self.shards)
            self._load_shard(self.shard_idx)
            self.pos = 0
        return x, y

# =============================================================================
# LR Schedule (cosine with warmup, matches C++)
# =============================================================================

def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

# =============================================================================
# Distributed Gradient Clipping (handles TP sharding correctly)
# =============================================================================

# Which parameter names are sharded (unique per GPU)?
_SHARDED_KEYS = {'c_attn.weight', 'c_attn.bias', 'c_proj.weight',
                 'fc1.weight', 'fc1.bias', 'fc2.weight'}

def _is_sharded(name):
    return any(k in name for k in _SHARDED_KEYS)

def clip_grad_norm_tp(model, max_norm):
    """Compute global grad norm across TP ranks, then clip."""
    rank = dist.get_rank()
    total_sq = torch.tensor(0.0, device='cuda')
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if _is_sharded(name):
            total_sq += p.grad.float().pow(2).sum()
        elif rank == 0:
            # Replicated grads are identical — count once
            total_sq += p.grad.float().pow(2).sum()
    dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
    total_norm = total_sq.sqrt()
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef)
    return total_norm.item()

# =============================================================================
# Sync replicated params at init (broadcast from rank 0)
# =============================================================================

def sync_replicated_params(model):
    for name, p in model.named_parameters():
        if not _is_sharded(name):
            dist.broadcast(p.data, src=0)

# =============================================================================
# Main
# =============================================================================

def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    ws   = dist.get_world_size()
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    torch.cuda.set_device(device)

    if rank == 0:
        print("=== GPT-2 Training Script (PyTorch Tensor Parallel) ===")
        print(f"Configuration:")
        for k, v in CONFIG.items():
            print(f"  {k}: {v}")
        print(f"  head_dim: {CONFIG['n_embd'] // CONFIG['n_head']}")
        print(f"  B={B}, T={T}")
        print(f"  global_batch: {GLOBAL_BATCH}")
        print(f"  grad_accum_steps: {GRAD_ACCUM}")
        print(f"  Weight Tying: ENABLED (wte <-> lm_head)")
        print(f"\nInitializing Tensor Parallel run with {ws} GPUs...")

    model = GPT(CONFIG).to(device)
    sync_replicated_params(model)

    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Number of parameters per GPU: {num_params}")

    # Optimizer (AdamW, no weight decay on bias/norm/ln)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if any(k in name for k in ['bias', 'ln', 'norm']):
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW([
        {'params': decay,    'weight_decay': 0.01},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8)

    # Data loaders
    train_loader = DataLoaderLite(B, T, "train", DATA_ROOT, device)
    val_loader   = DataLoaderLite(B, T, "val",   DATA_ROOT, device)

    # Timers
    timer_step  = CudaTimer()
    timer_data  = CudaTimer()
    timer_fwd   = CudaTimer()
    timer_loss  = CudaTimer()
    timer_bwd   = CudaTimer()
    timer_clip  = CudaTimer()
    timer_optim = CudaTimer()

    if rank == 0:
        print("\nStarting training...")

    # CSV log
    log_file = None
    if rank == 0:
        log_file = open("../attn/attn_run_log_pytorch_tp.csv", "w")
        log_file.write("step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec,"
                       "timer_data,timer_fwd,timer_loss,timer_bwd,timer_clip,timer_optim,"
                       "timer_tok_emb,timer_pos_emb,timer_attn,timer_mlp,timer_ln_f,timer_lm_head\n")

    val_loss_log = -1.0

    for step in range(MAX_STEPS):
        timer_step.start()

        # ---- Validation every 100 steps ----
        if step % 100 == 0 or step == MAX_STEPS - 1:
            model.eval()
            val_loader.reset()
            val_acc = 0.0
            val_steps = 20
            with torch.no_grad():
                for _ in range(val_steps):
                    x, y = val_loader.next_batch()
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, CONFIG['vocab_size']), y.view(-1))
                    val_acc += loss.item() / val_steps
            if rank == 0:
                print(f"validation loss: {val_acc:.4f}")
            val_loss_log = val_acc
            model.train()

        # ---- Training step ----
        model.reset_timing()
        t_data = t_fwd = t_loss_t = t_bwd = t_clip = t_optim = 0.0

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro in range(GRAD_ACCUM):
            timer_data.start()
            x, y = train_loader.next_batch()
            t_data += timer_data.elapsed_sec()

            timer_fwd.start()
            logits = model(x)
            t_fwd += timer_fwd.elapsed_sec()

            timer_loss.start()
            loss = F.cross_entropy(logits.view(-1, CONFIG['vocab_size']), y.view(-1))
            loss_accum += loss.detach().item() / GRAD_ACCUM
            t_loss_t += timer_loss.elapsed_sec()

            timer_bwd.start()
            (loss / GRAD_ACCUM).backward()
            t_bwd += timer_bwd.elapsed_sec()

        # NaN check
        if math.isnan(loss_accum) or math.isinf(loss_accum):
            if rank == 0:
                print(f"ERROR: NaN/Inf at step {step}")
            break

        # Gradient clipping
        timer_clip.start()
        grad_norm = clip_grad_norm_tp(model, 1.0)
        t_clip = timer_clip.elapsed_sec()

        # LR update
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Optimizer step
        timer_optim.start()
        optimizer.step()
        t_optim = timer_optim.elapsed_sec()

        dt = timer_step.elapsed_sec()
        tok_sec = B * T * GRAD_ACCUM / dt

        if rank == 0:
            print(f"step {step:5d} | loss: {loss_accum:.6f}"
                  f" | lr {lr:.4e} | norm: {grad_norm:.4f}"
                  f" | dt: {dt*1000:.2f}ms | tok/sec: {tok_sec:.2f}")
            print(f"  [TIMING] data: {t_data*1000:.1f}ms"
                  f" | fwd: {t_fwd*1000:.1f}ms"
                  f" | loss: {t_loss_t*1000:.1f}ms"
                  f" | bwd: {t_bwd*1000:.1f}ms"
                  f" | clip: {t_clip*1000:.1f}ms"
                  f" | optim: {t_optim*1000:.1f}ms")
            model.print_timing()

            mt = model._times
            log_file.write(f"{step},{loss_accum:.6f},{val_loss_log:.6f},{lr:.8f},"
                           f"{grad_norm:.6f},{dt*1000:.2f},{tok_sec:.2f},"
                           f"{t_data*1000:.2f},{t_fwd*1000:.2f},{t_loss_t*1000:.2f},"
                           f"{t_bwd*1000:.2f},{t_clip*1000:.2f},{t_optim*1000:.2f},"
                           f"{mt['tok_emb']*1000:.2f},{mt['pos_emb']*1000:.2f},"
                           f"{mt['attn']*1000:.2f},{mt['mlp']*1000:.2f},"
                           f"{mt['ln_f']*1000:.2f},{mt['lm_head']*1000:.2f}\n")
            log_file.flush()

        val_loss_log = -1.0

    if rank == 0:
        if log_file:
            log_file.close()
        print("\n=== Training Complete ===")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
