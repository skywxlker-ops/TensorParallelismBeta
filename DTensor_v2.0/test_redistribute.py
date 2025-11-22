
import os
import torch
import numpy as np
from mpi4py import MPI
import dtensor as dt

def test_redistribute():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size != 2:
        if rank == 0:
            print("This test requires exactly 2 MPI processes.")
        return

    # 1. Setup
    dt.init()
    nccl_id = dt.get_unique_id() if rank == 0 else None
    nccl_id = comm.bcast(nccl_id, root=0)

    pg = dt.ProcessGroup(rank, world_size, rank, nccl_id)
    mesh = dt.Mesh(world_size)

    # Global tensor shape
    global_shape = [4, 8]
    
    # =================================================================
    # Test 1: Sharded to Replicated
    # =================================================================
    if rank == 0:
        print(f"--- Running Test 1: Sharded to Replicated ---", flush=True)

    # Layout 1: Sharded on columns (dim 1)
    layout_sharded = dt.Layout(mesh, global_shape, dt.ShardingType.SHARDED, 1)
    
    # Create a tensor with data [0, 1, 2, ..., 31]
    global_data = np.arange(np.prod(global_shape), dtype=np.float32).reshape(global_shape)
    
    # Get local shard for each rank
    # Rank 0 gets columns 0-3, Rank 1 gets columns 4-7
    local_data_shard = np.split(global_data, world_size, axis=1)[rank]

    # Create and set data for the initial DTensor
    tensor_sharded = dt.DTensor(mesh, pg)
    tensor_sharded.set_data(local_data_shard.flatten().tolist(), layout_sharded)

    # New Layout: Replicated
    layout_replicated = dt.Layout(mesh, global_shape, dt.ShardingType.REPLICATED)

    # Redistribute!
    tensor_replicated = tensor_sharded.redistribute(layout_replicated)

    # Verification
    assert tensor_replicated.get_layout().is_replicated()
    replicated_data = np.array(tensor_replicated.get_data()).reshape(global_shape)

    # Each rank should now have the full global data
    if not np.array_equal(replicated_data, global_data):
        print(f"[Rank {rank}] FAILED Sharded -> Replicated: Data mismatch.")
        print("Expected:\n", global_data)
        print("Got:\n", replicated_data)
    else:
        if rank == 0:
            print(f"[Rank {rank}] PASSED Sharded -> Replicated.")

    comm.barrier() 
    
    # =================================================================
    # Test 2: Replicated to Sharded
    # =================================================================
    if rank == 0:
        print(f"--- Running Test 2: Replicated to Sharded ---", flush=True)

    # Redistribute the replicated tensor back to the original sharded layout
    tensor_resharded = tensor_replicated.redistribute(layout_sharded)

    # Verification
    assert tensor_resharded.get_layout().is_sharded()
    resharded_data = np.array(tensor_resharded.get_data()).reshape(layout_sharded.get_local_shape(rank))

    if not np.array_equal(resharded_data, local_data_shard):
        print(f"[Rank {rank}] FAILED Replicated -> Sharded: Data mismatch.")
        print("Expected:\n", local_data_shard)
        print("Got:\n", resharded_data)
    else:
        if rank == 0:
            print(f"[Rank {rank}] PASSED Replicated -> Sharded.")

    comm.barrier()

    # =================================================================
    # Test 3: Sharded to Sharded (different dimension)
    # =================================================================
    if rank == 0:
        print(f"--- Running Test 3: Sharded to Sharded (different dim) ---", flush=True)

    # New Layout: Sharded on rows (dim 0)
    layout_sharded_rows = dt.Layout(mesh, global_shape, dt.ShardingType.SHARDED, 0)

    # Redistribute from column-sharded to row-sharded
    tensor_row_sharded = tensor_sharded.redistribute(layout_sharded_rows)
    
    # Verification
    assert tensor_row_sharded.get_layout().is_sharded()
    
    # Get the expected new local shard
    expected_row_shard = np.split(global_data, world_size, axis=0)[rank]
    row_sharded_data = np.array(tensor_row_sharded.get_data()).reshape(layout_sharded_rows.get_local_shape(rank))

    if not np.array_equal(row_sharded_data, expected_row_shard):
        print(f"[Rank {rank}] FAILED Sharded -> Sharded (col to row): Data mismatch.")
        print("Expected:\n", expected_row_shard)
        print("Got:\n", row_sharded_data)
    else:
        if rank == 0:
            print(f"[Rank {rank}] PASSED Sharded -> Sharded (col to row).")
            
    comm.barrier()
    if rank == 0:
        print(f"--- All tests completed ---", flush=True)


if __name__ == "__main__":
    test_redistribute()
