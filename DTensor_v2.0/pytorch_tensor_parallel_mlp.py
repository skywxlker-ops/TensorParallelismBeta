#!/usr/bin/env python3
"""
PyTorch DTensor Reference Implementation for Tensor Parallel MLP.

This script uses manual DTensor operations to match the C++ implementation,
avoiding the AsyncCollectiveTensor issues with parallelize_module.

Usage:
    torchrun --nproc_per_node=2 pytorch_tensor_parallel_mlp.py

Flow (matching tensor_parallel_mlp.cpp):
    1. X: Replicated across GPUs
    2. Layer 1: H = X @ W1_Shard + B1 (output sharded)
    3. Layer 2: Y_partial = H @ W2_Shard + B2 (partial sums)
    4. All-reduce Y to get final result
    5. Backward pass computes gradients
"""
    
import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh

torch.set_printoptions(precision=4, sci_mode=False)


def display_tensor(name, tensor, rank=None, only_rank=None):
    """Display tensor with format similar to C++ implementation."""
    if only_rank is not None and rank != only_rank:
        return
    
    print(f"\n{name}:")
    if isinstance(tensor, DTensor):
        print(f"  DTensor: placements={tensor.placements}, shape={tensor.shape}")
        local = tensor.to_local()
        print(f"  Local (rank {rank}): shape={tuple(local.shape)}, device='{local.device}'")
        print(local)
    else:
        print(f"  Tensor: shape={tuple(tensor.shape)}, device='{tensor.device}'")
        print(tensor)


def tensor_parallel_mlp_manual(rank, world_size, device_mesh, seed=None):
    """
    Tensor Parallel MLP using manual DTensor operations.
    
    This mirrors the C++ implementation exactly:
    - X replicated
    - W1 column-sharded (dim 2), W2 row-sharded (dim 1)
    - Manual all-reduce for sync
    - Autograd through local tensor operations
    
    Dimensions (matching C++):
        B = 2   (batch size)
        T = 4   (token/sequence length)
        C = 2   (input features)
        F = 4   (hidden features, sharded to F/2 = 2 per GPU)
    """
    B, T, C, F_dim = 2, 4, 2, 4
    F_local = F_dim // world_size
    
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    if seed is not None:
        torch.manual_seed(seed + rank)  # Different seed per rank for weights
    
    print(f"\n{'='*45}")
    print(f"\tRANK {rank} - TENSOR PARALLEL MLP (Manual)")
    print(f"{'='*45}")
    print(f"B={B}, T={T}, C={C}, F={F_dim}, local_F={F_local}")
    
    # ============ TIMING SETUP ============
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    sync_start = torch.cuda.Event(enable_timing=True)
    sync_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    
    # ============ FORWARD PASS (TIMED) - includes initialization like C++ ============
    start_event.record()
    
    # ============ INPUT X (Replicated) ============
    torch.manual_seed(seed)  # Same seed for X across all ranks
    X = torch.rand(B, T, C, device=device, requires_grad=False)
    dist.broadcast(X, src=0)
    
    if rank == 0:
        print("\n x - dtensor :")
        display_tensor("X (Input, Replicated)", X, rank)
    
    # ============ LAYER 1 WEIGHTS (Column Parallel) ============
    # W1: [B, C, F] sharded on dim 2 -> local [B, C, F_local]
    torch.manual_seed(seed + rank * 100)
    W1_local = torch.rand(B, C, F_local, device=device, requires_grad=True)
    B1_local = torch.rand(B, T, F_local, device=device, requires_grad=True)
    
    if rank == 0:
        print("\n w1 shard (local):")
    display_tensor(f"W1_local (GPU {rank})", W1_local, rank)
    
    if rank == 0:
        print("\n b1 (local):")
    display_tensor(f"B1_local (GPU {rank})", B1_local, rank)
    
    # ============ LAYER 2 WEIGHTS (Row Parallel) ============
    # W2: [B, F, C] sharded on dim 1 -> local [B, F_local, C]
    torch.manual_seed(seed + rank * 200)
    W2_local = torch.rand(B, F_local, C, device=device, requires_grad=True)
    B2_local = torch.rand(B, T, C, device=device, requires_grad=True)
    
    if rank == 0:
        print("\n w2 shard (local):")
    display_tensor(f"W2_local (GPU {rank})", W2_local, rank)
    
    if rank == 0:
        print("\n b2 (local):")
    display_tensor(f"B2_local (GPU {rank})", B2_local, rank)
    
    # Layer 1: H = X @ W1_local + B1 (column parallel - output sharded)
    H = torch.bmm(X, W1_local) + B1_local
    
    print("\n h - dtensor (local shard):")
    display_tensor(f"H (GPU {rank})", H, rank)
    
    # Layer 2: Y_partial = H @ W2_local + B2 (row parallel - partial sum)
    Y_partial = torch.bmm(H, W2_local) + B2_local
    
    end_event.record()
    torch.cuda.synchronize()
    forward_time = start_event.elapsed_time(end_event)
    
    if rank == 0:
        print("\n Y before sync (partial sum):")
    display_tensor(f"Y_partial (GPU {rank})", Y_partial, rank, only_rank=0)
    
    # ============ SYNC (All-Reduce) ============
    sync_start.record()
    
    # Clone to preserve gradient chain, then all-reduce in-place
    Y = Y_partial.clone()
    dist.all_reduce(Y, op=dist.ReduceOp.SUM)
    
    sync_end.record()
    torch.cuda.synchronize()
    sync_time = sync_start.elapsed_time(sync_end)
    
    if rank == 0:
        print("\n Y after sync:")
    display_tensor(f"Y (GPU {rank})", Y, rank, only_rank=0)
    
    # ============ BACKWARD PASS (TIMED) ============
    if rank == 0:
        print("\n=== Computing Gradients (Backward Pass) ===")
    
    backward_start.record()
    
    # Loss = mean(Y)
    loss = Y.mean()
    loss.backward()
    
    backward_end.record()
    torch.cuda.synchronize()
    backward_time = backward_start.elapsed_time(backward_end)
    
    # ============ DISPLAY GRADIENTS ============
    if rank == 0:
        print("\n=== Gradients Computed ===\n")
        
        if W1_local.grad is not None:
            print("W1_Shard gradient:")
            print(f"Tensor(shape={tuple(W1_local.grad.shape)}, dtype={W1_local.grad.dtype}, device='{W1_local.grad.device}')")
            print(W1_local.grad)
        else:
            print("W1_local has no gradient")
        
        if W2_local.grad is not None:
            print("\nW2_Shard gradient:")
            print(f"Tensor(shape={tuple(W2_local.grad.shape)}, dtype={W2_local.grad.dtype}, device='{W2_local.grad.device}')")
            print(W2_local.grad)
        else:
            print("W2_local has no gradient")
    
    return {
        'forward_time_ms': forward_time,
        'sync_time_ms': sync_time,
        'backward_time_ms': backward_time,
        'X': X,
        'Y': Y,
        'H': H,
    }


def main():
    parser = argparse.ArgumentParser(description='PyTorch DTensor Tensor Parallel MLP')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--iterations', type=int, default=10, help='Number of benchmark iterations')
    args = parser.parse_args()
    
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"Initialized distributed: world_size={world_size}")
    
    # Create device mesh for tensor parallelism
    device_mesh = init_device_mesh("cuda", (world_size,))
    
    # Run tensor parallel MLP
    results = tensor_parallel_mlp_manual(rank, world_size, device_mesh, args.seed)
    
    # ============ BENCHMARKS ============
    if rank == 0:
        print("\n\n ========= BENCHMARKS =========\n")
        print(f" FORWARD DURATION  : {results['forward_time_ms']:.4f} ms")
        print(f" SYNC DURATION     : {results['sync_time_ms']:.4f} ms")
        print(f" BACKWARD DURATION : {results['backward_time_ms']:.4f} ms")
        print(f" TOTAL DURATION    : {results['forward_time_ms'] + results['sync_time_ms'] + results['backward_time_ms']:.4f} ms")
        print()
        print(f" B : 2")
        print(f" T : 4")
        print(f" C : 2")
        print(f" F : 4")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
