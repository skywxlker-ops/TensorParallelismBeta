#!/usr/bin/env python3
import argparse
import numpy as np
import dtensor.dtensor_cpp as dt

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=float, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--ffn", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    # --- Setup MPI + NCCL ---
    dt.mpi_init()
    print(f"[rank {dt.mpi_rank()}] device =", dt.cuda_get_device())
    rank = dt.mpi_rank()
    world = dt.mpi_world_size()

    if args.ffn % world != 0:
        raise SystemExit(f"--ffn ({args.ffn}) must be divisible by world size ({world})")
    if args.hidden % world != 0:
        # not strictly required for this demo, but nice symmetry
        print(f"[rank {rank}] warning: hidden ({args.hidden}) not divisible by world ({world})")

    # bootstrap NCCL id across ranks
    nccl_id = dt.nccl_unique_id_bytes_bcast_root0()
    pg = dt.ProcessGroup(rank, world, rank, nccl_id)

    print(f"[rank {rank}] device after PG =", dt.cuda_get_device())

    B = int(args.batch)
    H = args.hidden
    F = args.ffn
    tp = world

    # --- Make replicated input X on all ranks (same seed => same tensor) ---
    rng_in = np.random.default_rng(args.seed)
    X = rng_in.standard_normal((B, H), dtype=np.float32)

    # -----------------------
    # ColumnParallelLinear
    #   W1_i: [H, F/tp]
    #   Y_i = X @ W1_i + b1_i    (no gather)
    # -----------------------
    F_per = F // tp
    rng_w1 = np.random.default_rng(1234 + rank)
    W1_i = (rng_w1.standard_normal((H, F_per), dtype=np.float32) * (1.0 / np.sqrt(H))).astype(np.float32)
    b1_i = np.zeros((F_per,), dtype=np.float32)

    Y_i = X @ W1_i
    Y_i += b1_i

    # GELU shard-wise
    H_i = gelu(Y_i).astype(np.float32)  # shape: [B, F/tp]

    # -----------------------
    # RowParallelLinear
    #   W2_i: [F/tp, H]
    #   Z_i = H_i @ W2_i         (partial)
    #   Z   = sum_i Z_i          (all-reduce)
    # bias b2 is replicated; add on host after allreduce
    # -----------------------
    rng_w2 = np.random.default_rng(4321 + rank)
    W2_i = (rng_w2.standard_normal((F_per, H), dtype=np.float32) * (1.0 / np.sqrt(F))).astype(np.float32)
    b2 = np.zeros((H,), dtype=np.float32)

    Z_i = H_i @ W2_i            # [B, H] partial

    # --- All-reduce partials on device ---
    # flatten to 1D for device copy, then reshape back for checking
    Z_i_flat = Z_i.reshape(-1).astype(np.float32, copy=False)

    # Use DTensor to provide a device buffer
    t = dt.DTensor(world, Z_i_flat.size, rank)
    t.copy_from_numpy(Z_i_flat)

    # in-place all-reduce SUM
    work = pg.all_reduce_f32(t.device_ptr(), t.size())
    work.wait()

    Z = t.to_numpy().reshape(B, H)
    Z += b2  # replicated bias on host

    # --- Reference computation on host (no TP) for validation ---
    # Reconstruct full W1, W2 with the same per-rank seeds so every process can check locally.
    W1_full = np.empty((H, F), dtype=np.float32)
    W2_full = np.empty((F, H), dtype=np.float32)
    for r in range(tp):
        rng1r = np.random.default_rng(1234 + r)
        W1_r = (rng1r.standard_normal((H, F_per), dtype=np.float32) * (1.0/np.sqrt(H))).astype(np.float32)
        W1_full[:, r*F_per:(r+1)*F_per] = W1_r

        rng2r = np.random.default_rng(4321 + r)
        W2_r = (rng2r.standard_normal((F_per, H), dtype=np.float32) * (1.0/np.sqrt(F))).astype(np.float32)
        W2_full[r*F_per:(r+1)*F_per, :] = W2_r

    Y_ref = X @ W1_full + np.zeros((F,), dtype=np.float32)
    H_ref = gelu(Y_ref)
    Z_ref = H_ref @ W2_full + b2

    # --- Check & print ---
    max_abs_err = float(np.max(np.abs(Z - Z_ref)))
    l2 = float(np.linalg.norm(Z - Z_ref) / (1 + np.linalg.norm(Z_ref)))

    if rank == 0:
        print(f"TP config: world={world} (tp-size), batch={B}, hidden={H}, ffn={F}")
        print("Output shape:", Z.shape)
        print("Output checksum:", float(Z.sum()))
        print("Max abs error vs reference:", max_abs_err)
        print("Relative L2 error:", l2)

if __name__ == "__main__":
    main()

import atexit

def safe_finalize():
    try:
        # ensure PG and buffers are destroyed before finalize
        # (they go out of scope here; gc collects soon after)
        pass
    finally:
        try:
            dt.mpi_barrier()
        except Exception:
            pass
        try:
            dt.mpi_finalize()
        except Exception:
            pass

atexit.register(safe_finalize)