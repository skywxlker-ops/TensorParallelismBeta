import os
import time
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, Partial, distribute_tensor
from torch.distributed.tensor.parallel import loss_parallel

# =============================================================================
# Configuration
# =============================================================================

class GPTConfig:
    def __init__(self):
        self.context_length = 1024
        self.vocab_size = 50304
        self.n_embd = 384
        self.n_layers = 3
        self.weight_tying = True

# =============================================================================
# Data Loader
# =============================================================================

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root):
        self.B = B
        self.T = T
        self.process_rank = process_rank 
        self.num_processes = num_processes
        
        self.shards = sorted(glob.glob(os.path.join(data_root, f"*{split}*.bin")))
        if not self.shards:
            self.shards = sorted(glob.glob(os.path.join(data_root, f"*{split}*.npy")))
        if not self.shards:
            raise ValueError(f"No shard found in {data_root} for split {split}")
            
        self.reset()
        
    def reset(self):
        self.current_shard_idx = 0
        self.load_shard(self.shards[self.current_shard_idx])
        self.current_position = self.B * self.T * self.process_rank
        
    def load_shard(self, filename):
        if filename.endswith('.npy'):
            self.tokens = np.load(filename, mmap_mode='r').view(-1).astype(np.uint16)
        else:
            self.tokens = np.fromfile(filename, dtype=np.uint16)
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        
        x = torch.from_numpy(buf[:-1].astype(np.int64)).view(B, T)
        y = torch.from_numpy(buf[1:].astype(np.int64)).view(B, T)
        
        self.current_position += B * T * self.num_processes
        
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.load_shard(self.shards[self.current_shard_idx])
            self.current_position = B * T * self.process_rank
            
        return x.cuda(), y.cuda()

# official loss_parallel API will be used instead

# =============================================================================
# Model Components
# =============================================================================

class MLP(nn.Module):
    def __init__(self, config, device_mesh):
        super().__init__()
        self.device_mesh = device_mesh
        
        # LayerNorm with DTensor params
        master_ln = nn.LayerNorm(config.n_embd)
        self.ln_weight = nn.Parameter(distribute_tensor(master_ln.weight.data, device_mesh, [Replicate()]))
        self.ln_bias = nn.Parameter(distribute_tensor(master_ln.bias.data, device_mesh, [Replicate()]))
        
        # Column-Parallel
        master_fc_up = nn.Linear(config.n_embd, 4 * config.n_embd)
        nn.init.normal_(master_fc_up.weight, std=0.02)
        nn.init.zeros_(master_fc_up.bias)
        self.fc_up_weight = nn.Parameter(distribute_tensor(master_fc_up.weight.data, device_mesh, [Shard(0)]))
        self.fc_up_bias = nn.Parameter(distribute_tensor(master_fc_up.bias.data, device_mesh, [Shard(0)]))
        
        self.gelu = nn.GELU(approximate='tanh')
        
        # Row-Parallel
        master_fc_down = nn.Linear(4 * config.n_embd, config.n_embd)
        nn.init.normal_(master_fc_down.weight, std=0.02 / math.sqrt(2 * config.n_layers))
        nn.init.zeros_(master_fc_down.bias)
        self.fc_down_weight = nn.Parameter(distribute_tensor(master_fc_down.weight.data, device_mesh, [Shard(1)]))
        self.fc_down_bias = nn.Parameter(distribute_tensor(master_fc_down.bias.data, device_mesh, [Replicate()]))

    def forward(self, x):
        # x is Replicated DTensor
        x_ln = F.layer_norm(x, (x.size(-1),), self.ln_weight, self.ln_bias)
        
        h = F.linear(x_ln, self.fc_up_weight, self.fc_up_bias) # h is Shard(-1)
        h = self.gelu(h)
        h_partial = F.linear(h, self.fc_down_weight, bias=None) 
        h_replicated = h_partial.redistribute(self.device_mesh, [Replicate()])
        return x + (h_replicated + self.fc_down_bias)

class GPT(nn.Module):
    def __init__(self, config, device_mesh):
        super().__init__()
        self.config = config
        self.device_mesh = device_mesh
        self.world_size = device_mesh.size()
        self.rank = device_mesh.get_rank()
        
        self.local_vocab = config.vocab_size // self.world_size
        self.vocab_start = self.rank * self.local_vocab
        self.vocab_end = self.vocab_start + self.local_vocab
        
        master_wte = torch.empty(config.vocab_size, config.n_embd)
        nn.init.normal_(master_wte, std=0.02)
        self.wte_weight = nn.Parameter(distribute_tensor(master_wte, device_mesh, [Shard(0)]))
        
        master_wpe = torch.empty(config.context_length, config.n_embd)
        nn.init.normal_(master_wpe, std=0.02)
        self.wpe_weight = nn.Parameter(distribute_tensor(master_wpe, device_mesh, [Replicate()]))
        
        self.mlps = nn.ModuleList([MLP(config, device_mesh) for _ in range(config.n_layers)])
        
        master_ln_f = nn.LayerNorm(config.n_embd)
        self.ln_f_weight = nn.Parameter(distribute_tensor(master_ln_f.weight.data, device_mesh, [Replicate()]))
        self.ln_f_bias = nn.Parameter(distribute_tensor(master_ln_f.bias.data, device_mesh, [Replicate()]))

    def forward(self, idx):
        # idx is Replicated DTensor [B, T]
        local_idx = idx.to_local()
        
        mask = (local_idx >= self.vocab_start) & (local_idx < self.vocab_end)
        clamped_idx = (local_idx - self.vocab_start).clamp(0, self.local_vocab - 1)
        
        tok_emb_local = F.embedding(clamped_idx, self.wte_weight.to_local())
        tok_emb_local.masked_fill_(~mask.unsqueeze(-1), 0.0)
        
        tok_emb = DTensor.from_local(tok_emb_local, self.device_mesh, [Partial()]).redistribute(self.device_mesh, [Replicate()])
        
        pos = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_dt = DTensor.from_local(pos, self.device_mesh, [Replicate()])
        pos_emb = F.embedding(pos_dt, self.wpe_weight)
        
        x = tok_emb + pos_emb
        for block in self.mlps:
            x = block(x)
            
        x_ln = F.layer_norm(x, (x.size(-1),), self.ln_f_weight, self.ln_f_bias)
        logits = F.linear(x_ln, self.wte_weight, bias=None)
        return logits

# =============================================================================
# Main
# =============================================================================

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    
    device_mesh = init_device_mesh("cuda", (world_size,))
    config = GPTConfig()
    
    B, T = 8, 1024
    global_batch = 65536
    grad_accum_steps = global_batch // (B * T)
    
    model = GPT(config, device_mesh).cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    params_per_gpu = sum(p.to_local().numel() for p in model.parameters())
    if rank == 0:
        print(f"Total parameters: {total_params:,}")
        print(f"Parameters per GPU: {params_per_gpu:,}")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    
    data_root = "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Data"
    train_loader = DataLoaderLite(B, T, 0, 1, "train", data_root)
    val_loader = DataLoaderLite(B, T, 0, 1, "val", data_root)

    # LR Scheduler logic (matching C++ benchmark)
    def get_lr(step):
        max_lr = 1e-4
        min_lr = 1e-5
        warmup_steps = 174
        max_steps = 1738
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        if step > max_steps:
            return min_lr
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # Logging setup
    log_file = None
    if rank == 0:
        log_file = open("PyTorch_run_log3.csv", "w")
        log_file.write("step,loss,lr,norm,dt_ms,tok_per_sec\n")

    for step in range(1738):
        t0 = time.time()
        
        if step % 100 == 0:
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for _ in range(10):
                    x, y = val_loader.next_batch()
                    x_dt = DTensor.from_local(x, device_mesh, [Replicate()])
                    logits = model(x_dt)
                    if logits.dim() == 3:
                        logits = logits.view(-1, logits.size(-1))
                    if y.dim() == 2:
                        targets = y.view(-1)
                    else:
                        targets = y
                        
                    with loss_parallel():
                        loss = F.cross_entropy(logits, targets)
                        val_loss_accum += loss.item() / 10
            if rank == 0:
                print(f"step {step} | val loss: {val_loss_accum:.4f}")
            model.train()

        optimizer.zero_grad()
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x_dt = DTensor.from_local(x, device_mesh, [Replicate()])
            logits = model(x_dt)
            if logits.dim() == 3:
                logits = logits.view(-1, logits.size(-1))
            if y.dim() == 2:
                targets = y.view(-1)
            else:
                targets = y
                
            with loss_parallel():
                loss = F.cross_entropy(logits, targets)
                loss = loss / grad_accum_steps
                loss.backward()
                loss_accum += loss.item()

        # Replicated param sync 
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    # All parameters are now DTensors. Sync Replicated ones.
                    if any(isinstance(pl, Replicate) for pl in p.grad.placements):
                        dist.all_reduce(p.grad.to_local(), op=dist.ReduceOp.SUM)
                        p.grad.to_local().div_(world_size)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Learning rate update
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.step()
        
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000
        tokens_per_sec = (B * T * grad_accum_steps) / (dt / 1000.0)
        
        if rank == 0:
            print(f"step {step:5d} | loss: {loss_accum:.6f} | lr {lr:.4e} | norm: {norm.item():.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            if log_file:
                log_file.write(f"{step},{loss_accum:.6f},{lr:.4e},{norm.item():.4f},{dt:.2f},{tokens_per_sec:.2f}\n")
                log_file.flush()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

