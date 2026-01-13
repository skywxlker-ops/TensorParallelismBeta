"""
GPT-2 Training with PyTorch Native Tensor Parallelism + Real Dataset

This script implements GPT-2 training using:
  - PyTorch's native Tensor Parallelism (TP)
  - Real dataset: edufineweb_train_000001.npy
  - HellaSwag evaluation

Run with:
    torchrun --nnodes 1 --nproc-per-node 2 gpt_tensor_parallel_train.py
"""

import os
import math
import time
import inspect
from dataclasses import dataclass
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._tensor import init_device_mesh, DTensor

from hellaswag import render_example, iterate_examples

# -----------------------------------------------------------------------------
# Gradient clipping for DTensor compatibility

def clip_grad_norm_dtensor(parameters, max_norm, norm_type=2.0):
    """Clip gradients for models with mixed DTensor and regular Tensor parameters."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(parameters)
    
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)
    
    total_norm_sq = torch.tensor(0.0, device=grads[0].device if hasattr(grads[0], 'device') else 'cpu')
    
    for grad in grads:
        if isinstance(grad, DTensor):
            local_grad = grad.to_local()
            grad_norm_sq = local_grad.pow(norm_type).sum()
        else:
            grad_norm_sq = grad.pow(norm_type).sum()
        total_norm_sq = total_norm_sq + grad_norm_sq.to(total_norm_sq.device)
    
    if dist.is_initialized():
        dist.all_reduce(total_norm_sq)
    
    total_norm = total_norm_sq.pow(1.0 / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    
    for grad in grads:
        if isinstance(grad, DTensor):
            grad.to_local().mul_(clip_coef.to(grad.to_local().device))
        else:
            grad.mul_(clip_coef.to(grad.device))
    
    return total_norm

# -----------------------------------------------------------------------------
# Model Components

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        qkv_size = qkv.size(-1)
        local_embd = qkv_size // 3
        local_n_head = local_embd * self.n_head // self.n_embd
        head_dim = local_embd // local_n_head
        
        q, k, v = qkv.split(local_embd, dim=2)
        k = k.view(B, T, local_n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, local_n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, local_n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, local_embd)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
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
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
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
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, master_process):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        if master_process:
            print(f"num decayed params: {len(decay_params)}, num nodecay: {len(nodecay_params)}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=False, foreach=False)
        return optimizer


# -----------------------------------------------------------------------------
# DataLoader

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderTP:
    """DataLoader for Tensor Parallelism - all ranks see same data."""
    def __init__(self, B, T, data_file, rank, world_size):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        
        print(f"[Rank {rank}] Loading tokens from {data_file}...")
        self.tokens = load_tokens(data_file)
        print(f"[Rank {rank}] Loaded {len(self.tokens):,} tokens")
        
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        
        return x, y

    def reset(self):
        self.current_position = 0


# -----------------------------------------------------------------------------
# TP Setup

def get_tp_parallelize_plan(config):
    plan = {}
    for i in range(config.n_layer):
        plan[f"transformer.h.{i}.attn.c_attn"] = ColwiseParallel()
        plan[f"transformer.h.{i}.attn.c_proj"] = RowwiseParallel()
        plan[f"transformer.h.{i}.mlp.c_fc"] = ColwiseParallel()
        plan[f"transformer.h.{i}.mlp.c_proj"] = RowwiseParallel()
    return plan


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def get_most_likely_row(tokens, mask, logits):
    """Helper for HellaSwag evaluation."""
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# -----------------------------------------------------------------------------
# Main

def main():
    # Initialize
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    if world_size > 1:
        device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,))
        rank = device_mesh.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device_mesh = None
        rank = 0
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    master_process = (rank == 0)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    if master_process:
        print("=" * 60)
        print("GPT-2 Training with Tensor Parallelism + Real Data")
        print("=" * 60)
        print(f"World size: {world_size}, Device: {device}")
    
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    
    # Hyperparameters
    B = 8            # batch size
    T = 512          # sequence length
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 1000
    max_steps = 10000
    eval_interval = 1000
    gen_interval = 2000   # Generate samples every 500 steps
    
    # Generation prompts (used throughout training)
    gen_prompts = [
        "Hello, I am a language model,",
        "The quick brown fox",
        "In the year 2025,",
        "Hi my name is",
        "Tell me a joke",
    ]
    
    torch.set_float32_matmul_precision('high')
    
    # Model config (GPT2-medium scale, fits on 2x12GB with TP)
    config = GPTConfig(
        block_size=T,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
    )
    
    if master_process:
        print(f"\nModel: {config.n_layer}L, {config.n_head}H, {config.n_embd}E")
        print(f"Batch: {B}, Seq: {T}")
    
    # Create model
    model = GPT(config).to(device)
    
    # Apply TP
    if device_mesh is not None:
        tp_plan = get_tp_parallelize_plan(config)
        model = parallelize_module(model, device_mesh, tp_plan)
        if master_process:
            print(f"Tensor Parallelism applied to {len(tp_plan)} modules")
    
    # Data
    data_file = os.path.join(os.path.dirname(__file__), "edufineweb_train_000001.npy")
    train_loader = DataLoaderTP(B, T, data_file, rank, world_size)
    
    # Optimizer
    optimizer = model.configure_optimizers(0.1, max_lr, master_process)
    
    # Tokenizer for generation
    enc = tiktoken.get_encoding("gpt2")
    
    # Training
    if master_process:
        print(f"\nStarting training for {max_steps} steps...")
    
    for step in range(max_steps):
        t0 = time.time()
        
        # HellaSwag evaluation
        if step % eval_interval == 0 or step == max_steps - 1:
            model.eval()
            num_correct, num_total = 0, 0
            for i, example in enumerate(iterate_examples("val")):
                if i >= 100:  # Quick eval on first 100 examples
                    break
                if i % world_size != rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens, mask = tokens.to(device), mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct += int(pred_norm == label)
            
            if dist.is_initialized():
                num_correct_t = torch.tensor(num_correct, device=device)
                num_total_t = torch.tensor(num_total, device=device)
                dist.all_reduce(num_correct_t)
                dist.all_reduce(num_total_t)
                num_correct, num_total = num_correct_t.item(), num_total_t.item()
            
            if master_process and num_total > 0:
                print(f"HellaSwag: {num_correct}/{num_total} = {num_correct/num_total:.4f}")
        
        # Sample generation at intervals
        if step > 0 and step % gen_interval == 0:
            model.eval()
            if master_process:
                print(f"\n--- Sample Generation at step {step} ---")
            
            for prompt_text in gen_prompts[:5]:  # Generate 2 samples during training
                tokens = enc.encode(prompt_text)
                tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    for _ in range(40):
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, _ = model(tokens[:, -config.block_size:])
                        logits = logits[:, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                        ix = torch.multinomial(topk_probs, 1)
                        xcol = torch.gather(topk_indices, -1, ix)
                        tokens = torch.cat((tokens, xcol), dim=1)
                
                if master_process:
                    print(f">>> {enc.decode(tokens[0].tolist())}")
            
            if master_process:
                print("---\n")
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Get batch (same data on all TP ranks)
        torch.manual_seed(step)  # Ensure same batch
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        loss.backward()
        norm = clip_grad_norm_dtensor(model.parameters(), 1.0)
        
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (B * T) / dt
        
        if master_process and step % 10 == 0:
            print(f"step {step:5d} | loss: {loss.item():.4f} | lr: {lr:.2e} | "
                  f"norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    # =========================================================================
    # Final Validation
    # =========================================================================
    if master_process:
        print("\n" + "=" * 60)
        print("FINAL VALIDATION")
        print("=" * 60)
    
    model.eval()
    
    # 1. Validation Loss (use later portion of data)
    val_losses = []
    train_loader.current_position = 50_000_000  # Start from middle of data for validation
    
    with torch.no_grad():
        for i in range(50):  # 50 validation batches
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            val_losses.append(loss.item())
    
    val_loss = sum(val_losses) / len(val_losses)
    
    if dist.is_initialized():
        val_loss_t = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_t)
        val_loss = val_loss_t.item() / world_size
    
    if master_process:
        print(f"\nValidation Loss: {val_loss:.4f}")
    
    # 2. Full HellaSwag Evaluation (all examples)
    num_correct, num_total = 0, 0
    for i, example in enumerate(iterate_examples("val")):
        if i % world_size != rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct += int(pred_norm == label)
        
        if master_process and i > 0 and i % 1000 == 0:
            print(f"  HellaSwag progress: {i}/10042...")
    
    if dist.is_initialized():
        num_correct_t = torch.tensor(num_correct, device=device)
        num_total_t = torch.tensor(num_total, device=device)
        dist.all_reduce(num_correct_t)
        dist.all_reduce(num_total_t)
        num_correct, num_total = num_correct_t.item(), num_total_t.item()
    
    if master_process and num_total > 0:
        print(f"\nFinal HellaSwag: {num_correct}/{num_total} = {num_correct/num_total:.4f}")
        print(f"  (GPT-2 124M baseline: ~0.294)")
    
    # 3. Sample Text Generation
    if master_process:
        print("\n" + "-" * 60)
        print("Final Sample Generations:")
        print("-" * 60)
    
    for prompt_text in gen_prompts:
        # Encode prompt
        tokens = enc.encode(prompt_text)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # All ranks must participate in generation for TP
        with torch.no_grad():
            for _ in range(50):  # Generate 50 tokens
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(tokens[:, -config.block_size:])
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                tokens = torch.cat((tokens, xcol), dim=1)
        
        # Only rank 0 prints
        if master_process:
            generated_text = enc.decode(tokens[0].tolist())
            print(f"\n>>> {prompt_text}")
            print(f"    {generated_text}")
    
    # Print final summary
    if master_process:
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"  Final train loss: {loss.item():.4f}")
        print(f"  Validation loss:  {val_loss:.4f}")
        if num_total > 0:
            print(f"  HellaSwag acc:    {num_correct/num_total:.4f}")
        print("=" * 60)
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


#torchrun --nnodes 1 --nproc-per-node 2 gpt_tensor_parallel_train.py

