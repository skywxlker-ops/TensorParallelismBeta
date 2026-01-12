"""
Benchmark DTensor sharding speed for (batch, time, channels) tensors with varying sizes.
Outputs results to CSV file for comparison.

Usage:
    torchrun --nproc_per_node=2 pytorch_benchmarking_vary.py
"""
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
import time
import csv
import os

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
    throughput = tensor_size_mb / (avg_time_ms / 1000)
    
    return {
        'b': b, 't': t, 'c': c,
        'size_mb': tensor_size_mb,
        'time_ms': avg_time_ms,
        'throughput_mbs': throughput
    }

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    mesh = DeviceMesh("cuda", list(range(world_size)))
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"PyTorch DTensor Sharding Benchmark (Varying Sizes)")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"Fixed Batch (b): 8")
        print(f"{'='*60}")

    # Varying parameters
    b = 8
    t_values = [128, 512, 1024, 2048]
    c_values = [768, 1024, 2048, 4096]
    
    results = []
    
    # Iterate through all combinations
    for t in t_values:
        for c in c_values:
            result = benchmark_sharding(b, t, c, mesh, shard_dim=2, num_iterations=1000)
            results.append(result)
            
            if rank == 0:
                print(f"Shape: ({result['b']}, {result['t']}, {result['c']}) | "
                      f"Size: {result['size_mb']:.2f} MB | "
                      f"Time: {result['time_ms']:.3f} ms | "
                      f"Throughput: {result['throughput_mbs']:.2f} MB/s")
    
    # Write CSV on rank 0 only
    if rank == 0:
        csv_path = os.path.join(os.path.dirname(__file__), 'pytorch_benchmark_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['b', 't', 'c', 'size_mb', 'time_ms', 'throughput_mbs'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")
        print(f"{'='*60}\n")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
