"""
Benchmark DTensor sharding speed for (batch, time, channels) tensors with varying sizes.

Usage:
    torchrun --nproc_per_node=2 benchmark_dtensor_sharding_vary.py
"""
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
import time

def benchmark_sharding(b, t, c, mesh, shard_dim=0, num_iterations=100):
    """Measure sharding speed for a (b, t, c) tensor."""
    
    rank = dist.get_rank()
    
    # Create tensor on GPU
    tensor = torch.randn(b, t, c, device=f"cuda:{rank}")
    
    # Warm-up
    for _ in range(10):
        dt = distribute_tensor(tensor, mesh, [Shard(shard_dim)])
        torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for i in range(num_iterations):
        dt = distribute_tensor(tensor, mesh, [Shard(shard_dim)])
        
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / num_iterations * 1000
    tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
    
    if rank == 0:
        print(f"Shape: ({b}, {t}, {c}) | Size: {tensor_size_mb:.2f} MB | Time: {avg_time_ms:.3f} ms | Throughput: {tensor_size_mb / (avg_time_ms / 1000):.2f} MB/s")

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    mesh = DeviceMesh("cuda", list(range(world_size)))
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DTensor Sharding Benchmark (Varying Sizes)")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"Fixed Batch (b): 8")
        print(f"{'='*60}")

    # Varying parameters
    b = 8
    # Varying c (channels) and t (time/sequence length)
    # Using a mix of sizes to simulate different model scales
    t_values = [128, 512, 1024, 2048]
    c_values = [768, 1024, 2048, 4096]
    
    # Iterate through all combinations
    for t in t_values:
        for c in c_values:
            # We use shard_dim=2 (channels) as typically done in the original benchmark example
            benchmark_sharding(b, t, c, mesh, shard_dim=2, num_iterations=1000)
            
    if rank == 0:
        print(f"{'='*60}\n")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
