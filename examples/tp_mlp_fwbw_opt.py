#!/usr/bin/env python3
import argparse
import numpy as np
import dtensor.dtensor_cpp as dt
import atexit


# DO NOT register yet; we’ll do it after successful mpi_init()

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def print_debug(rank, name, arr, max_elems=4):
    flat = arr.flatten()
    snippet = flat[:max_elems]
    print(f"[rank {rank}] {name} shape={arr.shape}, sample={snippet}")

def gelu_prime(x):
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    s = sqrt_2_over_pi * (x + 0.044715 * x**3)
    tanh_s = np.tanh(s)
    ds_dx = sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)
    term1 = 0.5 * (1.0 + tanh_s)
    term2 = 0.5 * x * (1.0 - tanh_s**2) * ds_dx
    return term1 + term2

def main():
    global _MPI_INITIALIZED

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=float, default=8)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--ffn", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--check_ref", type=int, default=1, help="build CPU reference (turn off for big H/F)")

    # --- NEW: accept the extra flags from the launcher (you can ignore them for now) ---
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--eps", type=float, default=1e-8)

    args, _unk = ap.parse_known_args()

    # --- Setup MPI + NCCL ---
    dt.mpi_init()
    _MPI_INITIALIZED = True
    atexit.register(_safe_finalize)   # register only after init succeeds

    rank = dt.mpi_rank()
    world = dt.mpi_world_size()
    print(f"[rank {rank}] MPI initialized on device {dt.cuda_get_device()}")

    if args.ffn % world != 0:
        raise SystemExit(f"--ffn ({args.ffn}) must be divisible by world size ({world})")
    if args.hidden % world != 0:
        print(f"[rank {rank}] warning: hidden ({args.hidden}) not divisible by world ({world})")

    nccl_id = dt.nccl_unique_id_bytes_bcast_root0()
    pg = dt.ProcessGroup(rank, world, rank, nccl_id)
    print(f"[rank {rank}] ProcessGroup created")

    B = int(args.batch)
    H = args.hidden
    F = args.ffn
    tp = world
    F_per = F // tp

    # ---------- host init ----------
    rng_in = np.random.default_rng(args.seed)
    X = rng_in.standard_normal((B, H), dtype=np.float32)
    print_debug(rank, "X", X)

    rng_w1 = np.random.default_rng(1234 + rank)
    W1_i = (rng_w1.standard_normal((H, F_per), dtype=np.float32) * (1.0 / np.sqrt(H))).astype(np.float32)
    b1_i = np.zeros((F_per,), dtype=np.float32)

    rng_w2 = np.random.default_rng(4321 + rank)
    W2_i = (rng_w2.standard_normal((F_per, H), dtype=np.float32) * (1.0 / np.sqrt(F))).astype(np.float32)
    b2 = np.zeros((H,), dtype=np.float32)

    print_debug(rank, "W1_i", W1_i)
    print_debug(rank, "W2_i", W2_i)

    # ---------- device buffers ----------
    X_d  = dt.DTensor(world, B * H, rank);       X_d.copy_from_numpy(X.ravel())
    W1_d = dt.DTensor(world, H * F_per, rank);   W1_d.copy_from_numpy(W1_i.ravel())
    W2_d = dt.DTensor(world, F_per * H, rank);   W2_d.copy_from_numpy(W2_i.ravel())

    Y_d  = dt.DTensor(world, B * F_per, rank)
    H_d  = dt.DTensor(world, B * F_per, rank)
    Z_d  = dt.DTensor(world, B * H, rank)

    # ---------- forward on GPU ----------
    pg.gemm_f32(X_d.device_ptr(), W1_d.device_ptr(), Y_d.device_ptr(), B, F_per, H)

    Y_host = Y_d.to_numpy().reshape(B, F_per)
    Y_host += b1_i
    H_host = gelu(Y_host).astype(np.float32)
    H_d.copy_from_numpy(H_host.ravel())

    pg.gemm_f32(H_d.device_ptr(), W2_d.device_ptr(), Z_d.device_ptr(), B, H, F_per)

    pg.all_reduce_f32(Z_d.device_ptr(), Z_d.size()).wait()

    Z = Z_d.to_numpy().reshape(B, H)
    Z += b2
    print_debug(rank, "Z (after all-reduce)", Z)

    # ---------- (optional) reference on CPU ----------
    if args.check_ref:
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

        max_abs_err = float(np.max(np.abs(Z - Z_ref)))
        l2 = float(np.linalg.norm(Z - Z_ref) / (1 + np.linalg.norm(Z_ref)))
        print(f"[rank {rank}] max abs error={max_abs_err:.6f}, relative L2={l2:.6e}, checksum={float(Z.sum()):.6f}")

    if rank == 0:
        print(f"\nFinal TP config: world={world}, batch={B}, hidden={H}, ffn={F}")
        print("Output shape:", Z.shape)

    # ---------- backward (still on CPU for now) ----------
    dZ = np.ones_like(Z, dtype=np.float32)
    print_debug(rank, "dZ ", dZ)

    dW2_i = H_host.T @ dZ
    db2_local = np.sum(dZ, axis=0)
    dH_i = dZ @ W2_i.T
    dY_i = dH_i * gelu_prime(Y_host)
    db1_i = np.sum(dY_i, axis=0)
    dW1_i = X.T @ dY_i
    dX_i = dY_i @ W1_i.T

    dX_i_flat = dX_i.ravel().astype(np.float32, copy=False)
    dX_d = dt.DTensor(world, dX_i_flat.size, rank)
    dX_d.copy_from_numpy(dX_i_flat)
    pg.all_reduce_f32(dX_d.device_ptr(), dX_d.size()).wait()
    dX = dX_d.to_numpy().reshape(B, H)

    print_debug(rank, "dX (after all-reduce)", dX)
    print(f"[rank {rank}] shapes: dW1_i {dW1_i.shape}, dW2_i {dW2_i.shape}, dX {dX.shape}")
    print_debug(rank, "checksum dW1_i", np.array([float(dW1_i.sum())]))
    print_debug(rank, "checksum dW2_i", np.array([float(dW2_i.sum())]))
    print_debug(rank, "checksum db1_i", np.array([float(db1_i.sum())]))
    print_debug(rank, "checksum db2_local", np.array([float(db2_local.sum())]))
    print_debug(rank, "checksum dX", np.array([float(dX.sum())]))

if __name__ == "__main__":
    main()
