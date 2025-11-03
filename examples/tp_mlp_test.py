#!/usr/bin/env python3
import argparse
import numpy as np
import dtensor.dtensor_cpp as dt
import atexit

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def print_debug(rank, name, arr, max_elems=4):
    """Helper to print small slices of arrays safely per rank."""
    flat = arr.flatten()
    snippet = flat[:max_elems]
    print(f"[rank {rank}] {name} shape={arr.shape}, sample={snippet}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=float, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--ffn", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    # --- Setup MPI + NCCL ---
    dt.mpi_init()
    rank = dt.mpi_rank()
    world = dt.mpi_world_size()
    print(f"[rank {rank}] MPI initialized on device {dt.cuda_get_device()}")

    if args.ffn % world != 0:
        raise SystemExit(f"--ffn ({args.ffn}) must be divisible by world size ({world})")
    if args.hidden % world != 0:
        print(f"[rank {rank}] warning: hidden ({args.hidden}) not divisible by world ({world})")

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
    print_debug(rank, "X", X)

    # -----------------------
    # ColumnParallelLinear
    # -----------------------
    F_per = F // tp
    rng_w1 = np.random.default_rng(1234 + rank)
    W1_i = (rng_w1.standard_normal((H, F_per), dtype=np.float32) * (1.0 / np.sqrt(H))).astype(np.float32)
    b1_i = np.zeros((F_per,), dtype=np.float32)

    print_debug(rank, "W1_i", W1_i)
    print_debug(rank, "b1_i", b1_i)

    Y_i = X @ W1_i
    Y_i += b1_i
    print_debug(rank, "Y_i", Y_i)

    # GELU shard-wise
    H_i = gelu(Y_i).astype(np.float32)
    print_debug(rank, "H_i (post-GELU)", H_i)

    # -----------------------
    # RowParallelLinear
    # -----------------------
    rng_w2 = np.random.default_rng(4321 + rank)
    W2_i = (rng_w2.standard_normal((F_per, H), dtype=np.float32) * (1.0 / np.sqrt(F))).astype(np.float32)
    b2 = np.zeros((H,), dtype=np.float32)

    print_debug(rank, "W2_i", W2_i)
    Z_i = H_i @ W2_i
    print_debug(rank, "Z_i (partial)", Z_i)

    # --- All-reduce partials on device ---
    Z_i_flat = Z_i.reshape(-1).astype(np.float32, copy=False)
    t = dt.DTensor(world, Z_i_flat.size, rank)
    t.copy_from_numpy(Z_i_flat)

    work = pg.all_reduce_f32(t.device_ptr(), t.size())
    work.wait()

    Z = t.to_numpy().reshape(B, H)
    Z += b2
    print_debug(rank, "Z (after all-reduce)", Z)

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

def safe_finalize():
    dt.mpi_barrier()

atexit.register(safe_finalize)

if __name__ == "__main__":
    main()
