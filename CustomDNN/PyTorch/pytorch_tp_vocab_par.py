import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.tensor import Replicate, Shard, distribute_tensor, DTensor, Partial

# ---- 1. Initialize Distributed Environment ----
_min_gpu_count = 2
if torch.cuda.device_count() < _min_gpu_count:
    print(f"Unable to locate sufficient {_min_gpu_count} gpus. Exiting.")
    sys.exit()

if "OMPI_COMM_WORLD_SIZE" in os.environ:
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
else:
    # Default fallback for torchrun
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        
dist.init_process_group("nccl")
# Fix: Ensure all ranks initialize weights identically before sharding
torch.manual_seed(42)

_world_size = int(os.environ.get("WORLD_SIZE", 1))
accelerator = torch.accelerator.current_accelerator()
device_type = accelerator.type if accelerator else "cuda" 

# Set local device to rank
torch.cuda.set_device(int(os.environ["RANK"]) % torch.cuda.device_count())

device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()

if _rank == 0:
    print(f"Starting TP example. World size: {_world_size}")

# ---- 2. Dataset Definition ----
# The .npy files are flat int16 token arrays (matching C++ DataLoader format).
# Each file has shape (N,) where N = total tokens in the shard.
# We slice windows of T+1 tokens: input=tokens[i:i+T], target=tokens[i+1:i+T+1]
class FlatTokenDataset(Dataset):
    def __init__(self, shard_paths, T):
        self.T = T
        self.data = []
        for p in shard_paths:
            arr = np.load(p, mmap_mode='r')
            # Flatten (in case stored as e.g. (N,) int16 already)
            self.data.append(arr.reshape(-1).astype(np.int32))
        # Build cumulative chunk counts per shard
        self.chunk_counts = [(len(d) - 1) // T for d in self.data]
        self.total = sum(self.chunk_counts)

    def __len__(self):
        return self.total
    
    def __getitem__(self, idx):
        # Find which shard this index belongs to
        for i, count in enumerate(self.chunk_counts):
            if idx < count:
                start = idx * self.T
                chunk = torch.from_numpy(self.data[i][start:start + self.T + 1].copy().astype(np.int64))
                return chunk[:-1], chunk[1:]
            idx -= count
        raise IndexError("Index out of range")

# --- Configuration ---
class GPTConfig:
    batch_size = 8
    context_length = 1024
    vocab_size = 50304
    n_embd = 384
    n_layers = 3
    weight_tying = True

config = GPTConfig()

BATCH_SIZE = config.batch_size
T = config.context_length
global_batch = 65536
grad_accum_steps = global_batch // (BATCH_SIZE * T)

max_lr = 1e-4
min_lr = max_lr * 0.1
warmup_steps = 174
max_steps = 1738

shard_files = [
    "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Edufineweb_numpy/edufineweb_train_000001.npy",
    "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Edufineweb_numpy/edufineweb_train_000002.npy",
    "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Edufineweb_numpy/edufineweb_train_000003.npy",
    "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Edufineweb_numpy/edufineweb_train_000004.npy"
] 
val_shard_files = [
    "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Edufineweb_numpy/edufineweb_val_000000.npy"
]

dataset = FlatTokenDataset(shard_files, T=config.context_length)
# Fix: All ranks in a TP group must process the SAME data. 
# DistributedSampler(num_replicas=_world_size) splits the data, which is WRONG for TP.
sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
train_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,
    num_workers=0,
    pin_memory=False
)

val_dataset = FlatTokenDataset(val_shard_files, T=config.context_length)
# Same for validation
val_sampler = DistributedSampler(val_dataset, num_replicas=1, rank=0, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0)

# ---- 3. Model Definition ----

# Vocab-Parallel Embedding: Each rank holds (vocab_size / world_size) rows.
class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, device_mesh):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.device_mesh = device_mesh
        self.rank = device_mesh.get_rank()
        self.world_size = device_mesh.size()
        self.local_vocab = vocab_size // self.world_size
        self.vocab_start = self.rank * self.local_vocab
        self.vocab_end = self.vocab_start + self.local_vocab
        
        # Initialize full weight on master and distribute to shards
        master_weight = torch.empty(vocab_size, embed_dim)
        nn.init.normal_(master_weight, std=0.02)
        self.weight = nn.Parameter(
            distribute_tensor(master_weight, device_mesh, [Shard(0)])
        )

    def forward(self, idx):
        # Extract local tensor if idx is a DTensor
        local_idx_tensor = idx.to_local() if hasattr(idx, "to_local") else idx
        
        # Mask: zero out indices not in this rank's range
        mask = (local_idx_tensor >= self.vocab_start) & (local_idx_tensor < self.vocab_end)
        # Clamp to valid range for local lookup (out-of-range will be zeroed)
        local_idx = (local_idx_tensor - self.vocab_start).clamp(0, self.local_vocab - 1)
        
        # Local embedding lookup (using the local shard of the DTensor)
        local_weight = self.weight.to_local()
        out = F.embedding(local_idx, local_weight)
        
        # Zero out rows for tokens not owned by this rank
        out = out * mask.unsqueeze(-1).float()
        # Return as Partial DTensor. 
        # Fix: Redistribute to Replicate BEFORE adding to pos_emb (which is replicated).
        # This prevents pos_emb from being added multiple times after the reduction.
        return DTensor.from_local(out, self.device_mesh, [Partial()]).redistribute(self.device_mesh, [Replicate()])

# Vocab-Parallel Cross Entropy
# Logic matches CustomDNN: 
# 1. Global Max reduction for stability.
# 2. Global Sum-Exp reduction.
# 3. Global Target Logit extraction (local lookup + SUM all-reduce).
class VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, targets, vocab_start, vocab_end):
        # vocab_parallel_logits: [B*T, local_V]
        # targets: [B*T]
        
        # 1. Global Max
        local_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
        global_max = local_max
        
        # 2. Global Sum-Exp
        logits_minus_max = vocab_parallel_logits - global_max
        exp_logits = torch.exp(logits_minus_max)
        local_sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(local_sum_exp, op=dist.ReduceOp.SUM)
        global_sum_exp = local_sum_exp
        
        # 3. Target Logit extraction
        # Each rank checks if the targets fall in its vocab shard
        target_mask = (targets >= vocab_start) & (targets < vocab_end)
        target0 = (targets - vocab_start).clamp(min=0, max=vocab_parallel_logits.size(-1) - 1)
        
        # Extract local target logits (zeroing out if not in this shard)
        local_target_logits = torch.gather(vocab_parallel_logits, dim=-1, index=target0.unsqueeze(-1))
        local_target_logits.masked_fill_(~target_mask.unsqueeze(-1), 0.0)
        dist.all_reduce(local_target_logits, op=dist.ReduceOp.SUM)
        global_target_logit = local_target_logits
        
        # Loss formula matches C++
        # loss = log(sum_exp) - (target_logit - max_logit)
        loss = torch.log(global_sum_exp) - (global_target_logit - global_max)
        
        # Cache for backward
        probs = exp_logits / global_sum_exp
        ctx.save_for_backward(probs, target_mask, target0)
        
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output):
        probs, target_mask, target0 = ctx.saved_tensors
        # Probs are already Softmax(logits)
        grad_logits = probs.clone()
        
        # grad_logits = P - (1 if matches else 0)
        # We need to subtract 1 ONLY for the shard that owns the token
        grad_logits.scatter_add_(dim=-1, index=target0.unsqueeze(-1), 
                                 src=-1.0 * target_mask.unsqueeze(-1).float())
        
        # Scale by grad_output (usually 1.0) and divisor (B*T)
        # Since forward returns loss.mean(), we MUST divide by total elements.
        grad_logits.mul_(grad_output / probs.size(0))
        
        return grad_logits, None, None, None

def vocab_parallel_cross_entropy(logits, targets, vocab_start, vocab_end):
    # Logits from F.linear(x, sharded_weight) will be local-tensor of partial logits if not handled by DTensor
    # In our GPT forward, logits = F.linear(x, self.wte.weight)
    # Since x is Replicated and self.wte.weight is Sharded(0), F.linear returns Sharded(1) DTensor locally.
    
    local_logits = logits.to_local() if hasattr(logits, "to_local") else logits
    if local_logits.dim() == 3:
        local_logits = local_logits.view(-1, local_logits.size(-1))
    if targets.dim() == 2:
        targets = targets.view(-1)
    return VocabParallelCrossEntropy.apply(local_logits, targets, vocab_start, vocab_end)

class MLP(nn.Module):
    def __init__(self, config, device_mesh):
        super().__init__()
        self.config = config
        self.device_mesh = device_mesh
        
        # fc_up: Column-Parallel (Shard(0) for Weight [Out, In], Shard(0) for Bias [Out])
        master_fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc1_weight = nn.Parameter(distribute_tensor(master_fc1.weight.data, device_mesh, [Shard(0)]))
        self.fc1_bias = nn.Parameter(distribute_tensor(master_fc1.bias.data, device_mesh, [Shard(0)]))
        
        self.gelu = nn.GELU(approximate='none')
        
        # fc_down: Row-Parallel (Shard(1) for Weight [Out, In], Replicate() for Bias [Out])
        master_fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.fc2_weight = nn.Parameter(distribute_tensor(master_fc2.weight.data, device_mesh, [Shard(1)]))
        self.fc2_bias = nn.Parameter(distribute_tensor(master_fc2.bias.data, device_mesh, [Replicate()]))

    def forward(self, x):
        # x is Replicated DTensor.
        # 1. fc_up: result is Sharded(last_dim) DTensor
        h = F.linear(x, self.fc1_weight, self.fc1_bias)
        # 2. gelu: result is Sharded(last_dim) DTensor
        h = self.gelu(h)
        # 3. fc_down: result is Partial DTensor. 
        # Fix: Redistribute to Replicate (All-Reduce) BEFORE adding bias.
        # Otherwise bias is added on each rank and multiplied by world_size in the end.
        out_partial = F.linear(h, self.fc2_weight, bias=None)
        return out_partial.redistribute(self.device_mesh, [Replicate()]) + self.fc2_bias

class Block(nn.Module):
    def __init__(self, config, device_mesh):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd)
        # Manually distribute ln weights as replicated DTensors
        self.ln.weight = nn.Parameter(distribute_tensor(self.ln.weight.data, device_mesh, [Replicate()]))
        self.ln.bias = nn.Parameter(distribute_tensor(self.ln.bias.data, device_mesh, [Replicate()]))
        self.mlp = MLP(config, device_mesh)

    def forward(self, x):
        # CustomDNN exactly does: x + mlp(ln(x))
        return x + self.mlp(self.ln(x))

class GPT(nn.Module):
    def __init__(self, config, device_mesh):
        super().__init__()
        self.config = config
        self.device_mesh = device_mesh
        # Vocab-parallel token embedding
        self.wte = VocabParallelEmbedding(config.vocab_size, config.n_embd, device_mesh)
        # Position embedding is replicated
        master_wpe = nn.Embedding(config.context_length, config.n_embd)
        # Ensure wpe also uses DTensor for consistency in clip_grad_norm_
        self.wpe_weight = nn.Parameter(
            distribute_tensor(master_wpe.weight.data, device_mesh, [Replicate()])
        )
        self.mlps = nn.Sequential(*[Block(config, device_mesh) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Manually distribute ln_f weights
        self.ln_f.weight = nn.Parameter(distribute_tensor(self.ln_f.weight.data, device_mesh, [Replicate()]))
        self.ln_f.bias = nn.Parameter(distribute_tensor(self.ln_f.bias.data, device_mesh, [Replicate()]))

    def forward(self, idx):
        device = idx.device
        B, T = idx.size()
        
        # Wrap input and position indices as DTensors (Replicated)
        # to match the DTensor-based weights.
        idx_dt = DTensor.from_local(idx, self.device_mesh, [Replicate()])
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        pos_dt = DTensor.from_local(pos, self.device_mesh, [Replicate()])
        
        # Vocab-parallel embedding: each rank does local lookup + all_reduce
        tok_emb = self.wte(idx_dt)       # [B, T, n_embd] after all_reduce (replicated output)
        pos_emb = F.embedding(pos_dt, self.wpe_weight) # [1, T, n_embd]
        
        x = tok_emb + pos_emb
        x = self.mlps(x)
        x = self.ln_f(x)
        
        if self.config.weight_tying:
            # Result of linear with replicated x and sharded(0) weight is sharded(1) logits
            # Return as Partial for cross_entropy
            logits = F.linear(x, self.wte.weight, bias=None)
            return logits, self.wte.vocab_start, self.wte.vocab_end


# Move model to device and parallelize
tp_model = GPT(config, device_mesh).to(device_type)

# Tensor Parallel plan: only MLP layers are column/row parallel.
# wte is already vocab-parallel (manual implementation above).
# wpe remains replicated.
tp_plan = {}
for i in range(config.n_layers):
    # fc_up: column-parallel (shard hidden features dimension)
    tp_plan[f"mlps.{i}.mlp.fc_up"] = ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(-1))
    # fc_down: row-parallel (input sharded on hidden-features dim, output is all-reduced)
    tp_plan[f"mlps.{i}.mlp.fc_down"] = RowwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate())

# All parameters are now manually distributed as DTensors in __init__
# so parallelize_module is no longer needed.

# Custom AdamW behavior (matching gpt2_tp_test.cpp: bias & norm excluded from weight delay)
param_dict = {pn: p for pn, p in tp_model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'bias' not in n and 'ln' not in n and 'norm' not in n]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or 'bias' in n or 'ln' in n or 'norm' in n]

optim_groups = [
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': nodecay_params, 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# ---- 4. Training Loop ----
if _rank == 0:
    print("Tensor Parallel GPT Training Started")

# Logging (Rank 0)
log_file = None
if _rank == 0:
    # Ensure logs folder from config exists
    log_file = open("PyTorch_run_log.csv", "w")
    log_file.write("step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n")

train_iter = iter(train_loader)
val_iter = iter(val_loader)

for step in range(max_steps):
    # Validation every 100 steps
    val_loss_accum_log = -1.0
    if step % 100 == 0 or step == max_steps - 1:
        tp_model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                try:
                    v_inp, v_tgt = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    v_inp, v_tgt = next(val_iter)
                v_inp, v_tgt = v_inp.to(device_type), v_tgt.to(device_type)
                
                logits, v_start, v_end = tp_model(v_inp)
                # Use custom vocab-parallel cross entropy
                loss = vocab_parallel_cross_entropy(logits, v_tgt, v_start, v_end)
                val_loss_accum += loss.item() / val_loss_steps
        
        if _rank == 0:
            print(f"validation loss: {val_loss_accum:.4f}")
        val_loss_accum_log = val_loss_accum
        tp_model.train()
    
    # Start timer AFTER validation to correctly report training throughput
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    
    for micro_step in range(grad_accum_steps):
        try:
            inp, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inp, tgt = next(train_iter)
        
        inp, tgt = inp.to(device_type), tgt.to(device_type)
        logits, v_start, v_end = tp_model(inp)
        
        # Use custom vocab-parallel cross entropy
        loss = vocab_parallel_cross_entropy(logits, tgt, v_start, v_end)
        
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        
    # ---- Gradient Synchronization for Tensor Parallelism ----
    # Replicated parameters (LN, PosEmbed, Bias) get different gradients on different ranks 
    # because they are technically replicated. We must all-reduce them.
    # Sharded parameters (Embedding, MLP Weights) sum correctly if using Partial DTensors,
    # but here we handle them manually for safety if needed, or rely on DTensor dispatch.
    with torch.no_grad():
        for p in tp_model.parameters():
            if p.grad is not None:
                if isinstance(p.grad, DTensor):
                    # If grad is Partial, redistribute to Replicate to trigger all-reduce
                    if any(isinstance(pl, Partial) for pl in p.grad.placements):
                        p.grad.redistribute(device_mesh, [Replicate()])
                    
                    # If grad is Replicate, it might have diverged (due to floating point or bugs).
                    # We all-reduce the local part to be sure.
                    if any(isinstance(pl, Replicate) for pl in p.grad.placements):
                        dist.all_reduce(p.grad.to_local(), op=dist.ReduceOp.SUM)
                        p.grad.to_local().div_(_world_size)
                # Note: Sharded gradients don't need manual sync as each rank owns its shard.
                else:
                    # Regular tensor (shouldn't happen with our current script, but for safety)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(_world_size)

    torch.nn.utils.clip_grad_norm_(tp_model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.step()
    
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = BATCH_SIZE * T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    # Calculate norm manually for logging (since clip_grad_norm return is global)
    with torch.no_grad():
        total_norm_sq = torch.tensor(0.0, device=device_type)
        for p in tp_model.parameters():
            if p.grad is not None:
                # If d-tensor, we sum the local norm_sq then all_reduce(SUM)
                # If replicate, we take the local norm_sq (which is the same on all ranks).
                # To simplify, we sum all local norms and all_reduce. 
                # Replicated params will be overcounted by world_size, so we'll divide those or just
                # handle them specially.
                local_grad = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                total_norm_sq += local_grad.detach().pow(2).sum()
        
        dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)
        # Handle overcounting of replicated parameters (if any aren't correctly DTensor-sharded)
        # Actually in DTenosrs, sharded params sum correctly. Replicated params sum N times.
        # But here we want the GLOBAL norm.
        total_norm = total_norm_sq.sqrt()

    if _rank == 0:
        loss_val = loss_accum.item()
        norm_val = total_norm.item()
        print(f"step {step:4d} | loss: {loss_val:.6f} | lr {lr:.4e} | norm: {norm_val:.4f} | dt: {dt*1000.:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        if log_file is not None:
            log_file.write(f"{step},{loss_val:.6f},{val_loss_accum_log:.6f},{lr:.6f},{norm_val:.4f},{dt*1000.0:.2f},{tokens_per_sec:.2f}\n")
            log_file.flush()

if _rank == 0 and log_file is not None:
    log_file.close()
    print("=== Training Complete ===")

if dist.is_initialized():
    dist.destroy_process_group()
