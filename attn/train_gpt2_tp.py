import os
import math
import time
import csv
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# PyTorch Tensor Parallelism imports
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    loss_parallel,
)
from torch.distributed.tensor import DTensor, Replicate, Shard
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Model Definition (identical architecture to the DDP baseline)
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_k = nn.Linear(config.n_embd, config.n_embd)
        self.c_v = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # x.size() is [B, T, C] where C is the full unpartitioned dimension
        B, T, C = x.size()
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        
        # Local head dimension computation
        # q, k, v might be DTensors partitioned via Colwise, so we don't reshape manually across global dims
        # Actually DTensor view handles shape gracefully if the partition dimension maps correctly.
        k = k.view(B, T, -1, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, -1, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, -1, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # y is [B, local_n_head, T, head_dim] -> transpose -> [B, T, local_n_head, head_dim] -> flat
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 3
    n_head: int = 2
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing disabled (same as baseline)
        # self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # DTensor with Shard(-1) placement (use_local_output=False)
        loss = None
        if targets is not None:
            with loss_parallel():
                # logits is already a Shard(-1) DTensor from ColwiseParallel lm_head
                targets_dt = DTensor.from_local(targets.contiguous().view(-1), logits.device_mesh, [Replicate()])
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_dt)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = False  # Fused AdamW can't mix DTensor + Tensor params in TP
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
# Data Loading (identical to baseline)
# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = "/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/DTensor/Data_Loader/Edufineweb_numpy"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# TP Setup & Parallelization
# -----------------------------------------------------------------------------
# Launch: torchrun --standalone --nproc_per_node=2 train_gpt2_tp.py

assert torch.cuda.is_available(), "TP requires CUDA"
dist.init_process_group(backend='nccl')

tp_rank = int(os.environ['RANK'])
tp_local_rank = int(os.environ['LOCAL_RANK'])
tp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{tp_local_rank}'
torch.cuda.set_device(device)
master_process = tp_rank == 0

device_type = "cuda"

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# TP processes the SAME data on all ranks (no data splitting)
total_batch_size = 65536
B = 8
T = 1024
# In TP, all ranks see the same data, so grad_accum_steps is NOT divided by world_size
grad_accum_steps = total_batch_size // (B * T)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# All ranks load SAME data (process_rank=0, num_processes=1)
train_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="val")

torch.set_float32_matmul_precision('high')

# Create model with same config as baseline
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# --- Apply Tensor Parallelism ---
tp_mesh = init_device_mesh("cuda", (tp_world_size,))

if master_process:
    print(f"\n=== Applying Tensor Parallelism (TP={tp_world_size}) ===")
    print(f"  Sharding: Embeddings, Attention, and MLP")

# Parallelize Embeddings and output projection
# ColwiseParallel on nn.Embedding: shards weight along vocab dim (Shard(0))
# output_layouts=Replicate() means internal all-reduce gives full embeddings
parallelize_module(
    model,
    tp_mesh,
    {
        "transformer.wte": ColwiseParallel(output_layouts=Replicate()),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }
)

# Parallelize each transformer block — Attention and MLP
for i, block in enumerate(model.transformer.h):
    # Attention
    parallelize_module(
        block.attn,
        tp_mesh,
        {
            "c_q": ColwiseParallel(),
            "c_k": ColwiseParallel(),
            "c_v": ColwiseParallel(),
            "c_proj": RowwiseParallel(),
        },
    )
    # MLP: c_fc is Column-Parallel, c_proj is Row-Parallel
    parallelize_module(
        block.mlp,
        tp_mesh,
        {
            "c_fc": ColwiseParallel(),
            "c_proj": RowwiseParallel(),
        },
    )

def tp_clip_grad_norm_(parameters, max_norm, tp_mesh, norm_type=2.0):
    """Correct clip_grad_norm_ for TP with mixed DTensor/Tensor params.
    
    Sharded DTensor grads: local squared norms are summed via all-reduce.
    Replicated DTensor/plain Tensor grads: squared norms used as-is (identical on all ranks).
    Verified: produces norms matching the single-GPU baseline at step 0.
    """
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    max_norm = float(max_norm)
    tp_world_size = tp_mesh.size()
    device = parameters[0].grad.device
    
    # Accumulate squared norms
    sharded_sq = torch.tensor(0.0, device=device)
    replicated_sq = torch.tensor(0.0, device=device)
    
    for p in parameters:
        g = p.grad
        if isinstance(g, DTensor):
            is_sharded = any(pl.is_shard() for pl in g.placements)
            local_norm_sq = torch.norm(g.to_local(), norm_type) ** norm_type
            if is_sharded:
                sharded_sq += local_norm_sq
            else:
                replicated_sq += local_norm_sq
        else:
            replicated_sq += torch.norm(g, norm_type) ** norm_type
    
    # All-reduce sharded component: sum of local squared norms = global squared norm
    dist.all_reduce(sharded_sq, group=tp_mesh.get_group())
    
    total_norm = (sharded_sq + replicated_sq) ** (1.0 / norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    for p in parameters:
        if isinstance(p.grad, DTensor):
            p.grad._local_tensor.mul_(clip_coef_clamped)
        else:
            p.grad.mul_(clip_coef_clamped)
    
    return total_norm



if master_process:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters per GPU: {total_params:,}")
    print(f"=== TP Applied Successfully (Unified DTensor) ===\n")

# Training hyperparameters (identical to baseline)
max_lr = 4e-4
min_lr = max_lr * 0.1
warmup_steps = 338
max_steps = 3384

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# Logging
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_attn_tp.txt")
with open(log_file, "w") as f:
    pass
csv_file = os.path.join(log_dir, f"log_attn_tp7.csv")
if master_process:
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "type", "loss", "lr", "norm", "dt_ms", "tok_per_sec"])

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Validation
    if step % 500 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, "val", f"{val_loss_accum.item():.6f}", "", "", "", ""])
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    # Text generation
    if ((step > 0 and step % 1000 == 0) or last_step):
        model.eval()
        num_return_sequences = 4
        max_length = 60
        tokens = enc.encode("america is")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + tp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        if master_process:
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {tp_rank} sample {i}: {decoded}")

    # Training step
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    # === CRITICAL FIX: Sync replicated parameter gradients ===
    # In pure TP without DDP, BF16 all-reduces in the backward pass of 
    # sharded layers (like RowwiseParallel) cause tiny numeric differences 
    # across ranks. These accumulate in the Adam states over 1000+ steps, 
    # causing replicated parameters (embeddings, layernorms) to drift apart. 
    # We must explicitly average replicated gradients BEFORE clipping/optimizer.
    for p in model.parameters():
        if p.grad is None:
            continue
        
        is_replicated = False
        if isinstance(p, DTensor):
            if all(pl.is_replicate() for pl in p.placements):
                is_replicated = True
        else:
            is_replicated = True # Plain tensors are naturally replicated
            
        if is_replicated:
            # Get the underlying standard tensor for all-reduce
            grad_tensor = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
            dist.all_reduce(grad_tensor, op=dist.ReduceOp.AVG, group=tp_mesh.get_group())
            
    norm = tp_clip_grad_norm_(model.parameters(), 1.0, tp_mesh)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = B * T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    
    # === DRIFT CHECK EVERY 5 STEPS ===
    if step % 5 == 0:
        # wte.weight is a DTensor — extract local shard for drift check
        wte_param = model.transformer.wte.weight
        wte_local = wte_param.to_local().detach().clone() if hasattr(wte_param, 'to_local') else wte_param.detach().clone()
        wte_other = wte_local.clone()
        dist.all_reduce(wte_other, op=dist.ReduceOp.SUM, group=tp_mesh.get_group())
        diff = (wte_other - 2 * wte_local).abs().max().item()
        if master_process:
            print(f"step {step:5d} | wte_drift: {diff:.3e} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f}")

    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, "train", f"{loss_accum.item():.6f}", f"{lr:.6e}", f"{norm:.4f}", f"{dt*1000:.2f}", f"{tokens_per_sec:.2f}"])

dist.destroy_process_group()
