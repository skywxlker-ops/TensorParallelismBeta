import os
import time
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.device_mesh import init_device_mesh
import torch.distributed as dist
from torch.distributed.tensor import DTensor

# =============================================================================
# Configuration
# =============================================================================

class GPTConfig:
    def __init__(self):
        self.batch_size = 8
        self.context_length = 1024
        self.vocab_size = 50304
        self.n_embd = 384
        self.n_layers = 3
        self.dropout = 0.0 # No dropout in C++ impl usually

# =============================================================================
# Data Loader
# =============================================================================

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        
        # Get list of shards
        shards = sorted(glob.glob(os.path.join(data_root, f"*{split}*.npy")))
        if len(shards) == 0:
            raise ValueError(f"No shards found for split {split} in {data_root}")
            
        self.shards = shards
        self.current_shard_idx = 0
        
        self.reset()
        
    def reset(self):
        self.current_shard_idx = 0
        self.load_shard(self.shards[self.current_shard_idx])
        self.current_position = self.B * self.T * self.process_rank
        
    def load_shard(self, filename):
        # Use mmap_mode='r' for efficient reading from disk without loading everything to RAM
        self.tokens = np.load(filename, mmap_mode='r')
        # Ensure divisible by 2 (uint16) - numpy handles this via dtype usually, 
        # but let's assume the saved npy is correct.
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        
        x = torch.from_numpy(buf[:-1].astype(np.int64)).view(B, T) # inputs
        y = torch.from_numpy(buf[1:].astype(np.int64)).view(B, T)  # targets
        
        # Advance position
        self.current_position += B * T * self.num_processes
        
        # Check if we need to switch to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.load_shard(self.shards[self.current_shard_idx])
            self.current_position = B * T * self.process_rank
            
        return x.cuda(), y.cuda()

# =============================================================================
# Model Components
# =============================================================================

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd)
        self.fc_up = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # GPT-2 uses tanh approximation
        self.fc_down = nn.Linear(4 * config.n_embd, config.n_embd)
        
        # Init weights (Approximating C++ init)
        # C++: fc_up std=0.02
        # C++: fc_down std=0.02/sqrt(2*n_layers)
        nn.init.normal_(self.fc_up.weight, std=0.02)
        nn.init.zeros_(self.fc_up.bias)
        nn.init.normal_(self.fc_down.weight, std=0.02 / math.sqrt(2 * config.n_layers))
        nn.init.zeros_(self.fc_down.bias)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc_up(h)
        h = self.gelu(h)
        h = self.fc_down(h)
        return x + h

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.context_length, config.n_embd)
        
        self.mlps = nn.ModuleList([MLP(config) for _ in range(config.n_layers)])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        # Init params
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)
        
        # Buffers for timing - standard python float accumulation
        self.t_tok_emb = 0.0
        self.t_pos_emb = 0.0
        self.t_mlp = 0.0
        self.t_ln_f = 0.0
        self.t_lm_head = 0.0
        
        # Events for timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos_idx = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # [1, T]
        
        # --- Token Embedding ---
        self.start_event.record()
        tok_emb = self.wte(idx)
        self.end_event.record()
        torch.cuda.synchronize()
        self.t_tok_emb += self.start_event.elapsed_time(self.end_event) / 1000.0

        # --- Position Embedding ---
        self.start_event.record()
        pos_emb = self.wpe(pos_idx)
        self.end_event.record()
        torch.cuda.synchronize()
        self.t_pos_emb += self.start_event.elapsed_time(self.end_event) / 1000.0
        
        x = tok_emb + pos_emb
        
        # --- MLP Blocks ---
        self.start_event.record()
        for block in self.mlps:
            x = block(x)
        self.end_event.record()
        torch.cuda.synchronize()
        self.t_mlp += self.start_event.elapsed_time(self.end_event) / 1000.0
        
        # --- Final LayerNorm ---
        self.start_event.record()
        x = self.ln_f(x)
        self.end_event.record()
        torch.cuda.synchronize()
        self.t_ln_f += self.start_event.elapsed_time(self.end_event) / 1000.0
        
        # --- LM Head ---
        self.start_event.record()
        logits = self.lm_head(x)
        self.end_event.record()
        torch.cuda.synchronize()
        self.t_lm_head += self.start_event.elapsed_time(self.end_event) / 1000.0
        
        return logits
    
    def reset_timing(self):
        self.t_tok_emb = 0.0
        self.t_pos_emb = 0.0
        self.t_mlp = 0.0
        self.t_ln_f = 0.0
        self.t_lm_head = 0.0

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Initialize Distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    
    # 2-way TP
    device_mesh = init_device_mesh("cuda", (world_size,))
    
    if rank == 0:
        print("=== GPT-2 Tensor Parallel Benchmarking Script (PyTorch) ===")
    
    # Config
    config = GPTConfig()
    
    # Data Loader
    data_root = "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Data"
    train_loader = DataLoaderLite(config.batch_size, config.context_length, rank, world_size, "train", data_root)
    val_loader = DataLoaderLite(config.batch_size, config.context_length, rank, world_size, "val", data_root)
    
    # Model
    model = GPT(config).cuda()
    
    # Count params before TP
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Original Parameter Count: {num_params}")
    
    # --- Apply Tensor Parallelism ---
    # Strategy:
    # Embeddings: Replicated (default, no action needed for basic TP if we accept simple replication. 
    #             For true TP savings we'd Row/Col embed, but let's stick to MLP TP as requested)
    # MLP fc_up: Colwise
    # MLP fc_down: Rowwise
    
    tp_plan = {}
    for i in range(config.n_layers):
        tp_plan[f"mlps.{i}.fc_up"] = ColwiseParallel()
        tp_plan[f"mlps.{i}.fc_down"] = RowwiseParallel()
    
    model = parallelize_module(model, device_mesh, tp_plan)
    
    if rank == 0:
        print("Model structure after TP:")
        print(model)

    
    # Count params after TP (per rank)
    num_params_tp = sum(p.to_local().numel() if isinstance(p, DTensor) else p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Per-Rank Parameter Count: {num_params_tp}")
        
    # Optimizer
    # C++ impl uses: max_lr = 1e-4, min_lr = 1e-5.
    # Optimizer
    # Split into two optimizers to avoid mixed Tensor/DTensor issues with fused/foreach kernels
    dtensor_params = [p for p in model.parameters() if isinstance(p, DTensor)]
    tensor_params = [p for p in model.parameters() if not isinstance(p, DTensor)]

    opt_dt = torch.optim.AdamW(dtensor_params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
    opt_t = torch.optim.AdamW(tensor_params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True)
    
    # Training Setup
    max_steps = 20 # Short run for benchmark
    
    B, T = config.batch_size, config.context_length
    global_batch = 65536
    grad_accum_steps = global_batch // (B * T)

    # Adjust grad_accum_steps
    grad_accum_steps = global_batch // (B * T)
    
    # Timers
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    model.train()
    
    # LR Schedule
    def get_lr(it):
        warmup_steps = 10 
        # C++: const int warmup_steps = max_steps / 10;
        # Since I'm using arbitrary max_steps here, let's align.
        max_lr = 1e-4
        min_lr = 1e-5
        if it < warmup_steps:
             return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # Logging
    log_file = None
    if rank == 0:
        os.makedirs("TP_MLP_Torch_Logs", exist_ok=True)
        log_path = f"TP_MLP_Torch_Logs/log_{int(time.time())}.csv"
        log_file = open(log_path, "w")
        log_file.write("step,loss,lr,norm,dt_ms,tok_per_sec,time_data,time_fwd,time_loss,time_bwd,time_clip,time_optim,t_tok_emb,t_pos_emb,t_mlp,t_ln_f,t_lm_head\n")
        print(f"Logging to {log_path}")

    # Training Loop
    t_start = time.time()
    
    for step in range(max_steps):
        t0 = time.time()
        
        opt_dt.zero_grad()
        opt_t.zero_grad()
        loss_accum = 0.0
        
        # Component Accumulators
        c_data = 0.0
        c_fwd = 0.0
        c_loss = 0.0
        c_bwd = 0.0
        
        for micro_step in range(grad_accum_steps):
            # Data
            start_event.record()
            # To ensure input consistency for TP, Rank 0 loads, then broadcast.
            # Simplified: All ranks load from SAME shard/pos (rank=0 logic for all)
            # Hack DataLoader:
            # Re-init loader locally with rank=0, world_size=1 logic?
            # Or just broadcast x, y.
            if rank == 0:
                x, y = train_loader.next_batch()
            else:
                x = torch.empty((B, T), dtype=torch.long, device="cuda")
                y = torch.empty((B, T), dtype=torch.long, device="cuda")
            
            dist.broadcast(x, src=0)
            dist.broadcast(y, src=0)
            
            end_event.record()
            torch.cuda.synchronize()
            c_data += start_event.elapsed_time(end_event) / 1000.0
            
            # Forward
            start_event.record()
            logits = model(x)
            end_event.record()
            torch.cuda.synchronize()
            c_fwd += start_event.elapsed_time(end_event) / 1000.0
            
            # Loss
            start_event.record()
            # Flatten for CE
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            loss = loss / grad_accum_steps
            end_event.record()
            torch.cuda.synchronize()
            c_loss += start_event.elapsed_time(end_event) / 1000.0
            
            loss_accum += loss.item()
            
            # Backward
            start_event.record()
            loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            c_bwd += start_event.elapsed_time(end_event) / 1000.0
            
        # Clip
        start_event.record()

        # Custom clip_grad_norm_ to handle mixed DTensor and Tensor
        dtensor_params = [p for p in model.parameters() if isinstance(p, DTensor)]
        tensor_params = [p for p in model.parameters() if not isinstance(p, DTensor)]
        
        total_norm_sq = 0.0
        
        if dtensor_params:
            for p in dtensor_params:
                if p.grad is not None:
                    param_norm = p.grad.norm(2)
                    total_norm_sq += param_norm.item() ** 2
        
        if tensor_params:
            for p in tensor_params:
                if p.grad is not None:
                    param_norm = p.grad.norm(2)
                    total_norm_sq += param_norm.item() ** 2
                    
        total_norm = math.sqrt(total_norm_sq)
        clip_coef = 1.0 / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(clip_coef)
                    
        norm = total_norm
        end_event.record()
        torch.cuda.synchronize()
        c_clip = start_event.elapsed_time(end_event) / 1000.0
        
        # Optim
        lr = get_lr(step)
        for param_group in opt_dt.param_groups:
            param_group['lr'] = lr
        for param_group in opt_t.param_groups:
            param_group['lr'] = lr
            
        start_event.record()
        opt_dt.step()
        opt_t.step()
        end_event.record()
        torch.cuda.synchronize()
        c_optim = start_event.elapsed_time(end_event) / 1000.0
        
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (B * T * grad_accum_steps) / dt
        
        if rank == 0:
            print(f"step {step:5d} | loss: {loss_accum:.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            print(f"  [TIMING] data: {c_data*1000:.1f}ms | fwd: {c_fwd*1000:.1f}ms | loss: {c_loss*1000:.1f}ms | bwd: {c_bwd*1000:.1f}ms | clip: {c_clip*1000:.1f}ms | optim: {c_optim*1000:.1f}ms")
            # Model Layer Timings
            print(f"  [LAYER] tok_emb: {model.t_tok_emb*1000:.1f}ms | pos_emb: {model.t_pos_emb*1000:.1f}ms | mlps: {model.t_mlp*1000:.1f}ms | ln_f: {model.t_ln_f*1000:.1f}ms | lm_head: {model.t_lm_head*1000:.1f}ms")

            log_file.write(f"{step},{loss_accum},{lr},{norm},{dt*1000},{tokens_per_sec},{c_data*1000},{c_fwd*1000},{c_loss*1000},{c_bwd*1000},{c_clip*1000},{c_optim*1000},"
                           f"{model.t_tok_emb*1000},{model.t_pos_emb*1000},{model.t_mlp*1000},{model.t_ln_f*1000},{model.t_lm_head*1000}\n")
            log_file.flush()
        
        # Reset Layer Timings
        model.reset_timing()
        
    if rank == 0:
        log_file.close()
        print("Done.")

    dist.destroy_process_group()