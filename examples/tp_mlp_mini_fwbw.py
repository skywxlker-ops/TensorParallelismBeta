#!/usr/bin/env python3
import argparse
import numpy as np
import dtensor.dtensor_cpp as dt
import atexit

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Insert this helper near top (after dt initialized)
def print_in_order(rank, world, name, arr):
    """Print arr for each rank in-order (rank 0, 1, ..., world-1)."""
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    for r in range(world):
        dt.mpi_barrier()
        if rank == r:
            print(f"\n[rank {rank}] ---- {name} ----\nshape = {arr.shape}\n{arr}\n[rank {rank}] ---- end {name} ----\n", flush=True)
    dt.mpi_barrier()


def gelu_prime(x):
    # derivative of approximate GELU implementation used above:
    # gelu(x) = 0.5 * x * (1 + tanh(s))
    # s = sqrt(2/pi) * (x + 0.044715 x^3)
    # derivative computed via chain rule (approx)
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    s = sqrt_2_over_pi * (x + 0.044715 * x**3)
    tanh_s = np.tanh(s)
    # d/dx tanh(s) = (1 - tanh(s)^2) * ds/dx
    ds_dx = sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)
    # Using product rule on 0.5 * x * (1 + tanh(s))
    term1 = 0.5 * (1.0 + tanh_s)
    term2 = 0.5 * x * (1.0 - tanh_s**2) * ds_dx
    return term1 + term2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=float, default=2)
    ap.add_argument("--hidden", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    # --- Setup MPI + NCCL ---
    dt.mpi_init()
    rank = dt.mpi_rank()
    world = dt.mpi_world_size()
    print(f"[rank {rank}] MPI initialized on device {dt.cuda_get_device()}")

    if args.ffn % world != 0:
        raise SystemExit(f"--ffn ({args.ffn}) must be divisible by world size ({world})")
    # if args.hidden % world != 0:
    #     print(f"[rank {rank}] warning: hidden ({args.hidden}) not divisible by world ({world})")

    # Bootstrap NCCL id across ranks
    nccl_id = dt.nccl_unique_id_bytes_bcast_root0()
    pg = dt.ProcessGroup(rank, world, rank, nccl_id)
    print(f"[rank {rank}] ProcessGroup created")

    B = int(args.batch)
    H = args.hidden
    F = args.ffn
    tp = world

    # --- Make replicated input X ---
    rng_in = np.random.default_rng(args.seed)
    X = rng_in.standard_normal((B, H), dtype=np.float32)
    print_in_order(rank, world, "X", X)

    # -----------------------
    # ColumnParallelLinear
    # -----------------------
    F_per = F // tp
    rng_w1 = np.random.default_rng(1234 + rank)
    W1_i = (rng_w1.standard_normal((H, F_per), dtype=np.float32) * (1.0 / np.sqrt(H))).astype(np.float32)
    b1_i = np.zeros((F_per,), dtype=np.float32)

    print_in_order(rank, world, "W1_i", W1_i)
    print_in_order(rank, world, "b1_i", b1_i)

    Y_i = X @ W1_i
    Y_i += b1_i
    print_in_order(rank, world, "Y_i", Y_i)

    # GELU shard-wise
    H_i = gelu(Y_i).astype(np.float32)
    print_in_order(rank, world, "H_i (post-GELU)", H_i)

    # -----------------------
    # RowParallelLinear
    # -----------------------
    rng_w2 = np.random.default_rng(4321 + rank)
    W2_i = (rng_w2.standard_normal((F_per, H), dtype=np.float32) * (1.0 / np.sqrt(F))).astype(np.float32)
    b2 = np.zeros((H,), dtype=np.float32)

    print_in_order(rank, world, "W2_i", W2_i)
    Z_i = H_i @ W2_i
    print_in_order(rank, world, "Z_i (partial)", Z_i)

    # --- All-reduce partials on device ---
    Z_i_flat = Z_i.reshape(-1).astype(np.float32, copy=False)
    t = dt.DTensor(world, Z_i_flat.size, rank)
    t.copy_from_numpy(Z_i_flat)

    work = pg.all_reduce_f32(t.device_ptr(), t.size())
    work.wait()

    Z = t.to_numpy().reshape(B, H)
    Z += b2
    print_in_order(rank, world, "Z (after all-reduce)", Z)

    # --- Reference computation ---
    W1_full = np.empty((H, F), dtype=np.float32)
    W2_full = np.empty((F, H), dtype=np.float32)
    for r in range(tp):
        rng1r = np.random.default_rng(1234 + r)
        W1_r = (rng1r.standard_normal((H, F_per), dtype=np.float32) * (1.0/np.sqrt(H))).astype(np.float32)
        W1_full[:, r*F_per:(r+1)*F_per] = W1_r

        rng2r = np.random.default_rng(4321 + r)
        W2_r = (rng2r.standard_normal((F_per, H), dtype=np.float32) * (1.0/np.sqrt(F))).astype(np.float32)
        W2_full[r*F_per:(r+1)*F_per, :] = W2_r

    Y_ref = X @ W1_full
    H_ref = gelu(Y_ref)
    Z_ref = H_ref @ W2_full + b2

    max_abs_err = float(np.max(np.abs(Z - Z_ref)))
    l2 = float(np.linalg.norm(Z - Z_ref) / (1 + np.linalg.norm(Z_ref)))

    print(f"[rank {rank}] max abs error={max_abs_err:.6f}, relative L2={l2:.6e}, checksum={float(Z.sum()):.6f}")

    if rank == 0:
        print(f"\nFinal TP config: world={world}, batch={B}, hidden={H}, ffn={F}")
        print("Output shape:", Z.shape)

    # BACKWARD (tensor-parallel aware)
    
    dZ = np.ones_like(Z, dtype=np.float32)
    print_in_order(rank, world, "dZ ", dZ)
    
    dW2_i = H_i.T @ dZ      
    print_in_order(rank, world, "dW2_i", dW2_i)
   
    db2_local = np.sum(dZ, axis=0)  
    print_in_order(rank, world, "db2_local", db2_local)
    
    dH_i = dZ @ W2_i.T       
    print_in_order(rank, world, "dH_i", dH_i)
    
    
    dY_i = dH_i * gelu_prime(Y_i) 
    print_in_order(rank, world, "dY_i (post-gelu-prime)", dY_i)
    
    
    db1_i = np.sum(dY_i, axis=0)  

    print_in_order(rank, world, "db1_i", db1_i)
    
    
    dW1_i = X.T @ dY_i  
    print_in_order(rank, world, "dW1_i", dW1_i)
    
    dX_i = dY_i @ W1_i.T
    
    print_in_order(rank, world, "dX_i (local)", dX_i)
    
  
    dX_i_flat = dX_i.reshape(-1).astype(np.float32, copy=False)
    t_dx = dt.DTensor(world, dX_i_flat.size, rank)
    t_dx.copy_from_numpy(dX_i_flat)
    
    work_dx = pg.all_reduce_f32(t_dx.device_ptr(), t_dx.size())
    work_dx.wait()
    
    dX = t_dx.to_numpy().reshape(B, H)
    print_in_order(rank, world, "dX (after all-reduce)", dX)
    
    # At this point:
    # - dW1_i is the gradient for this rank's shard of W1
    # - dW2_i is the gradient for this rank's shard of W2
    # - db1_i is gradient for shard b1 (local)
    # - db2_local is gradient for replicated b2 (same on all ranks; no reduce needed)
    # - dX is the full gradient w.r.t the replicated input X (after all-reduce)
    #
    # If you want to verify against a reference (non-sharded) implementation,
    # you can (on rank 0) reconstruct full W1_full/W2_full (you already do that)
    # and compute the reference gradients (X, W1_full, W2_full) and compare.
    
    # Print per-rank shapes/checksums for quick sanity:
    print(f"[rank {rank}] shapes: dW1_i {dW1_i.shape}, dW2_i {dW2_i.shape}, dX {dX.shape}")
    print_in_order(rank, world, "checksum dW1_i", np.array([float(dW1_i.sum())]))
    print_in_order(rank, world, "checksum dW2_i", np.array([float(dW2_i.sum())]))
    print_in_order(rank, world, "checksum db1_i", np.array([float(db1_i.sum())]))
    print_in_order(rank, world, "checksum db2_local", np.array([float(db2_local.sum())]))
    print_in_order(rank, world, "checksum dX", np.array([float(dX.sum())]))
    
def safe_finalize():
    dt.mpi_barrier()

atexit.register(safe_finalize)

if __name__ == "__main__":
    main()


