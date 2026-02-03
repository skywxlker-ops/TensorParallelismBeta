"""
Benchmark DTensor sharding speed for (batch, time, channels) tensors.

Usage:
    torchrun --nproc_per_node=2 benchmark_dtensor_sharding.py
"""
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
import time

def benchmark_sharding(b, t, c, shard_dim=0, num_iterations=100):
    """Measure sharding speed for a (b, t, c) tensor."""
    
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    mesh = DeviceMesh("cuda", list(range(world_size)))
    
    # Create tensor on GPU
    tensor = torch.randn(b, t, c, device=f"cuda:{rank}")

    for _ in range(10):
        dt = distribute_tensor(tensor, mesh, [Shard(shard_dim)])
        torch.cuda.synchronize()
    

    torch.cuda.synchronize()
    start = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(num_iterations):

        dt = distribute_tensor(tensor, mesh, [Shard(shard_dim)])

        torch.cuda.nvtx.range_pop()
        
    
    end = time.perf_counter()


    
    avg_time_ms = (end - start) / num_iterations * 1000
    tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
    
    if rank == 0:
        print(f"\n{'='*50}")
        print(f"DTensor Sharding Benchmark")
        print(f"{'='*50}")
        print(f"Shape:       ({b}, {t}, {c})")
        print(f"Size:        {tensor_size_mb:.2f} MB")
        print(f"Shard dim:   {shard_dim} ({'batch' if shard_dim==0 else 'time' if shard_dim==1 else 'channel'})")
        print(f"World size:  {world_size}")
        print(f"Iterations:  {num_iterations}")
        print(f"{'='*50}")
        print(f"Avg time:    {avg_time_ms:.3f} ms")
        print(f"Throughput:  {tensor_size_mb / (avg_time_ms / 1000):.2f} MB/s")
        print(f"{'='*50}\n")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    # Example: batch=32, time=512, channels=768 (like GPT-2 small)
    benchmark_sharding(b=8, t=998, c=1996, shard_dim=2)
