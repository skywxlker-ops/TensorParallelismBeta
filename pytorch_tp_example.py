#!/usr/bin/env python3
"""
@file pytorch_tp_example.py
@brief GPT-2 training script in PyTorch with Tensor Parallelism

This script implements GPT-2 training using PyTorch with custom TP primitives.
Architecture: Token Embedding -> Position Embedding -> MLP x n_layers -> Linear -> Cross Entropy

Usage:
    torchrun --nproc_per_node=2 pytorch_tp_example.py
"""

import os
import math
import csv
import time
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ReduceOp


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab size (padded to 64)
    n_embd: int = 384        # Reduced from 768 for smaller model (~30M params)
    n_layers: int = 6


# =============================================================================
# Distributed Setup
# =============================================================================

def setup_distributed():
    """Initialize distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


# =============================================================================
# TP Primitives: Column Parallel & Row Parallel Linear
# =============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.
    
    Weight shape: [out_features, in_features]
    Splits output features across GPUs.
    Each GPU holds: [out_features // world_size, in_features]
    
    Forward: Y_local = X @ W_local.T  (no communication needed)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        assert out_features % self.world_size == 0, \
            f"out_features ({out_features}) must be divisible by world_size ({self.world_size})"
        
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // self.world_size
        
        # Each GPU holds a slice of the output dimension
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features, device="cuda")
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_rank, device="cuda")
            )
        else:
            self.register_parameter("bias", None)
        
        self._init_weights()
    
    def _init_weights(self):
        # Xavier/He initialization
        std = math.sqrt(2.0 / self.in_features)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq, in_features]
        Returns:
            Local output of shape [batch, seq, out_features_per_rank]
        """
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.
    
    Weight shape: [out_features, in_features]
    Splits input features across GPUs.
    Each GPU holds: [out_features, in_features // world_size]
    
    Forward: Y = AllReduce(X_local @ W_local.T)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        assert in_features % self.world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({self.world_size})"
        
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_rank = in_features // self.world_size
        
        # Each GPU holds a slice of the input dimension
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank, device="cuda")
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device="cuda"))
        else:
            self.register_parameter("bias", None)
        
        self._init_weights()
    
    def _init_weights(self):
        std = math.sqrt(2.0 / self.in_features)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Local input tensor of shape [batch, seq, in_features_per_rank]
        Returns:
            Output tensor of shape [batch, seq, out_features] after all-reduce
        """
        output = F.linear(x, self.weight)
        dist.all_reduce(output, op=ReduceOp.SUM)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


# =============================================================================
# Vocab Parallel Embedding (Shard on vocab dimension)
# =============================================================================

class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer with vocabulary parallelism.
    Each GPU holds [vocab_size // world_size, embed_dim].
    Forward performs local lookup + all-reduce.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        assert vocab_size % self.world_size == 0
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.vocab_per_rank = vocab_size // self.world_size
        self.vocab_start = self.rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank
        
        self.weight = nn.Parameter(
            torch.empty(self.vocab_per_rank, embed_dim, device="cuda")
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: [B, T] token indices
        Returns:
            embeddings: [B, T, embed_dim]
        """
        # Create mask for tokens in this rank's vocabulary range
        mask = (indices >= self.vocab_start) & (indices < self.vocab_end)
        
        # Local indices (shifted to 0-based for this shard)
        local_indices = indices - self.vocab_start
        local_indices = local_indices.clamp(0, self.vocab_per_rank - 1)
        
        # Lookup
        embeddings = F.embedding(local_indices, self.weight)
        
        # Zero out embeddings for tokens not in this rank's range
        embeddings = embeddings * mask.unsqueeze(-1).float()
        
        # All-reduce to combine embeddings from all ranks
        dist.all_reduce(embeddings, op=ReduceOp.SUM)
        
        return embeddings


# =============================================================================
# MLP Block with Tensor Parallelism
# =============================================================================

class TensorParallelMLP(nn.Module):
    """
    MLP Block with TP.
    Architecture: LayerNorm -> ColumnParallel -> GELU -> RowParallel
    
    This pattern minimizes communication:
        - ColumnParallel: No comm needed (input is replicated)
        - RowParallel: One all-reduce at the end
    """
    
    def __init__(self, n_embd: int):
        super().__init__()
        hidden_size = 4 * n_embd
        
        self.ln = nn.LayerNorm(n_embd, device="cuda")
        self.fc_up = ColumnParallelLinear(n_embd, hidden_size, bias=True)
        self.fc_down = RowParallelLinear(hidden_size, n_embd, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x [B, T, C] -> [B, T, C]"""
        # Pre-Norm
        h = self.ln(x)
        
        # Column parallel (no comm)
        h = self.fc_up(h)
        
        # GELU activation
        h = F.gelu(h)
        
        # Row parallel (all-reduce)
        h = self.fc_down(h)
        
        return h


# =============================================================================
# GPT Model with Tensor Parallelism
# =============================================================================

class GPT(nn.Module):
    """
    GPT-2 model with Tensor Parallelism.
    
    Architecture:
        Token Embedding (vocab parallel) + Position Embedding (replicated)
        -> MLP x n_layers (with residual)
        -> Final LayerNorm
        -> Weight-tied output projection
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embedding: Vocab Parallel
        self.wte = VocabParallelEmbedding(config.vocab_size, config.n_embd)
        
        # Position embedding: Replicated (small enough)
        self.wpe = nn.Embedding(config.context_length, config.n_embd, device="cuda")
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)
        
        # MLP blocks
        self.mlps = nn.ModuleList([
            TensorParallelMLP(config.n_embd) for _ in range(config.n_layers)
        ])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd, device="cuda")
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: [B, T] token indices
        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = idx.shape
        
        # Position IDs [B, T]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0).expand(B, -1)
        
        # Get embeddings [B, T, C]
        tok_emb = self.wte(idx)      # [B, T, C]
        pos_emb = self.wpe(pos)      # [B, T, C]
        
        # Add embeddings
        x = tok_emb + pos_emb
        
        # Apply MLP blocks with residual connections
        for mlp in self.mlps:
            x = x + mlp(x)
        
        # Final normalization
        x = self.ln_f(x)
        
        # Weight tying: logits = x @ wte.weight.T 
        # wte.weight is [V/P, C], so we need all-gather then matmul
        weight_list = [torch.zeros_like(self.wte.weight) for _ in range(dist.get_world_size())]
        dist.all_gather(weight_list, self.wte.weight)
        full_weight = torch.cat(weight_list, dim=0)  # [V, C]
        
        logits = x @ full_weight.T  # [B, T, V]
        
        return logits
    
    def count_params(self) -> int:
        """Count total parameters (accounting for sharding)."""
        world_size = dist.get_world_size()
        total = 0
        for name, p in self.named_parameters():
            if "wte" in name:
                # Vocab parallel: each rank holds V/P
                total += p.numel() * world_size
            elif "fc_up" in name or "fc_down" in name:
                # TP layers: each rank holds fraction
                total += p.numel() * world_size
            else:
                # Replicated
                total += p.numel()
        return total


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# Gradient Clipping
# =============================================================================

def clip_grad_norm_distributed(model: nn.Module, max_norm: float) -> float:
    """Clip gradients with proper distributed handling."""
    total_norm_sq = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.pow(2).sum().item()
    
    # All-reduce to get global norm
    total_norm_tensor = torch.tensor([total_norm_sq], device="cuda")
    dist.all_reduce(total_norm_tensor, op=ReduceOp.SUM)
    global_norm = total_norm_tensor.item() ** 0.5
    
    if global_norm > max_norm:
        clip_coef = max_norm / (global_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    
    return global_norm


# =============================================================================
# Cross Entropy Loss with Vocab Parallelism
# =============================================================================

def cross_entropy_loss_vocab_parallel(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross entropy loss for vocab-parallel logits.
    Since logits are already gathered, use standard cross entropy.
    
    Args:
        logits: [B, T, V] 
        targets: [B, T]
    Returns:
        loss: scalar
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
    return loss


# =============================================================================
# Synthetic Data Loader (for testing)
# =============================================================================

class SyntheticDataLoader:
    """Synthetic data loader for testing."""
    
    def __init__(self, B: int, T: int, vocab_size: int, device: torch.device):
        self.B = B
        self.T = T
        self.vocab_size = vocab_size
        self.device = device
    
    def next_batch(self):
        """Generate random batch."""
        x = torch.randint(0, self.vocab_size, (self.B, self.T), device=self.device)
        y = torch.randint(0, self.vocab_size, (self.B, self.T), device=self.device)
        return x, y


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    rank, world_size = setup_distributed()
    
    try:
        if rank == 0:
            print("=== GPT-2 Training Script (PyTorch Tensor Parallelism) ===")
        
        # Configuration
        config = GPTConfig()
        config.context_length = 1024
        config.vocab_size = 50304
        # n_embd and n_layers use dataclass defaults (384, 6)
        
        # Training hyperparameters
        B = 4           # Batch size per rank
        T = 1024        # Sequence length
        grad_accum_steps = 16  # 4*1024*16 = 65536 tokens per step
        
        max_lr = 3e-4   # Scaled for 65k batch (sqrt scaling from 6e-4)
        min_lr = max_lr * 0.1
        warmup_steps = 204    # ~10% of max_steps
        max_steps = 2044     # Adjusted for larger batch
        
        if rank == 0:
            print("Configuration:")
            print(f"  vocab_size: {config.vocab_size}")
            print(f"  context_length: {config.context_length}")
            print(f"  n_embd: {config.n_embd}")
            print(f"  n_layers: {config.n_layers}")
            print(f"  World Size: {world_size}")
            print(f"  B={B}, T={T} per rank")
            print(f"  grad_accum_steps: {grad_accum_steps}")
        
        # Create model
        model = GPT(config)
        
        if rank == 0:
            num_params = model.count_params()
            print(f"Number of parameters: {num_params}")
        
        # Create optimizer (AdamW with weight decay)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1
        )
        
        # Create synthetic data loader
        train_loader = SyntheticDataLoader(B, T, config.vocab_size, torch.device("cuda"))
        
        if rank == 0:
            print("\nStarting training...")
        
        # Create CSV log file
        log_file = None
        csv_writer = None
        if rank == 0:
            log_file = open("training_log.csv", "w", newline="")
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(["step", "loss", "val_loss", "lr", "norm", "dt_ms", "tok_per_sec"])
        
        for step in range(max_steps):
            t0 = time.perf_counter()
            
            # Update learning rate
            lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            # Zero gradients
            optimizer.zero_grad()
            
            loss_accum = 0.0
            
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                
                # Forward
                logits = model(x)
                loss = cross_entropy_loss_vocab_parallel(logits, y)
                
                # Scale loss for accumulation
                loss = loss / grad_accum_steps
                
                # Backward
                loss.backward()
                
                # Accumulate loss for logging
                if micro_step == grad_accum_steps - 1:
                    loss_accum = loss.item() * grad_accum_steps
            
            # Clip gradients
            norm = clip_grad_norm_distributed(model, max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            
            if rank == 0:
                tokens_processed = B * T * grad_accum_steps
                tokens_per_sec = tokens_processed / dt
                
                print(f"step {step:5d} | loss: {loss_accum:.6f} | lr {lr:.4e} | "
                      f"dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
                
                csv_writer.writerow([step, f"{loss_accum:.6f}", "-1", f"{lr:.6f}", 
                                    f"{norm:.6f}", f"{dt * 1000:.2f}", f"{tokens_per_sec:.2f}"])
                log_file.flush()
        
        if rank == 0 and log_file:
            log_file.close()
            print("\n=== Training Complete ===")
        
    except Exception as e:
        print(f"RANK {rank} ERROR: {e}")
        raise
    
    cleanup()


if __name__ == "__main__":
    main()
