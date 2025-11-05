# tp_mlp_fwbw_matmul.py
#!/usr/bin/env python3
import argparse
import numpy as np
import dtensor.dtensor_cpp as dt
import gc
import traceback

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_prime(x):
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    s = sqrt_2_over_pi * (x + 0.044715 * x**3)
    tanh_s = np.tanh(s)
    ds_dx = sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)
    term1 = 0.5 * (1.0 + tanh_s)
    term2 = 0.5 * x * (1.0 - tanh_s**2) * ds_dx
    return term1 + term2

def print_debug(rank, name, arr, max_elems=4):
    flat = arr.flatten()
    snippet = flat[:max_elems]
    print(f"[rank {rank}] {name} shape={arr.shape}, sample={snippet}", flush=True)

def print_stats(rank, name, arr):
    arr = arr.astype(np.float32, copy=False)
    msg = (f"[rank {rank}] {name} stats: "
           f"min={arr.min():.5f} max={arr.max():.5f} "
           f"mean={arr.mean():.5f} std={arr.std():.5f}")
    print(msg, flush=True)

def _copy_from_np(t, arr):
    if hasattr(t, "copy_from_numpy"):
        t.copy_from_numpy(arr.astype(np.float32).ravel())
    else:
        t.copy_from_host(arr.astype(np.float32).ravel())

def _to_numpy(t):
    if hasattr(t, "to_numpy"):
        return t.to_numpy().astype(np.float32)
    else:
        out = np.empty((t.size(),), dtype=np.float32)
        t.copy_to_host(out)
        return out

def _allocator_print(tag: str):
    print(f"[allocator] {tag}", flush=True)
    if hasattr(dt, "allocator_stats"):
        print(dt.allocator_stats(), flush=True)
    elif hasattr(dt, "allocator_print_stats"):
        dt.allocator_print_stats()
    else:
        print("  (no allocator stats binding exposed)", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=16)  # NEW: sequence length T
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--ffn", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--check_ref", type=int, default=0, help="Build CPU reference for sanity-check.")
    ap.add_argument("--debug_fwd", type=int, default=1, help="Print forward intermediates.")
    ap.add_argument("--debug_bwd", type=int, default=1, help="Print backward intermediates.")
    ap.add_argument("--device_debug_copy", type=int, default=0,
                    help="If 1, copy device tensors to host for printing (heavier).")
    ap.add_argument("--debug_max_elems", type=int, default=4)
    args = ap.parse_args()

    rank = 0
    world = 1
    prior_exc = None
    try:
        # --- Setup MPI + NCCL ---
        dt.mpi_init()
        rank = dt.mpi_rank()
        world = dt.mpi_world_size()
        if args.ffn % world != 0:
            raise SystemExit(f"--ffn ({args.ffn}) must be divisible by world size ({world})")

        nccl_id = dt.nccl_unique_id_bytes_bcast_root0()
        pg = dt.ProcessGroup(rank, world, rank, nccl_id)

        # Ensure the strided-batched GEMM API exists
        if not hasattr(pg, "gemm_strided_batched_f32"):
            raise SystemExit("Current build lacks ProcessGroup.gemm_strided_batched_f32; "
                             "compile with the strided-batched GEMM binding for 3D path.")

        B = int(args.batch)
        T = int(args.tokens)
        H = int(args.hidden)
        F = int(args.ffn)
        tp = world
        F_per = F // tp
        BT = B * T  # for flattened weight-grad GEMMs

        # ----- host inits -----
        rng_in = np.random.default_rng(args.seed)
        X = rng_in.standard_normal((B, T, H), dtype=np.float32)

        rng_w1 = np.random.default_rng(1234 + rank)
        W1_i = (rng_w1.standard_normal((H, F_per), dtype=np.float32) * (1.0 / np.sqrt(H))).astype(np.float32)
        b1_i = np.zeros((F_per,), dtype=np.float32)

        rng_w2 = np.random.default_rng(4321 + rank)
        W2_i = (rng_w2.standard_normal((F_per, H), dtype=np.float32) * (1.0 / np.sqrt(F))).astype(np.float32)
        b2 = np.zeros((H,), dtype=np.float32)

        if args.debug_fwd:
            print_debug(rank, "X", X, args.debug_max_elems)
            print_stats(rank, "X", X)
            print_debug(rank, "W1 shard", W1_i, args.debug_max_elems)
            print_debug(rank, "W2 shard", W2_i, args.debug_max_elems)

        # ----- device buffers -----
        X_d   = dt.DTensor(world, B * T * H, rank);     _copy_from_np(X_d, X)
        W1_d  = dt.DTensor(world, H * F_per, rank);     _copy_from_np(W1_d, W1_i)
        W2_d  = dt.DTensor(world, F_per * H, rank);     _copy_from_np(W2_d, W2_i)
        b1_d  = dt.DTensor(world, F_per, rank);         _copy_from_np(b1_d, b1_i)  # (not used on device)
        b2_d  = dt.DTensor(world, H, rank);             _copy_from_np(b2_d, b2)    # (not used on device)

        # Transposes for backward/weight-grads
        W1_T_i = W1_i.T.copy()            # (F_per, H)
        W2_T_i = W2_i.T.copy()            # (H, F_per)
        W1_T_d = dt.DTensor(world, F_per * H, rank);  _copy_from_np(W1_T_d, W1_T_i)
        W2_T_d = dt.DTensor(world, H * F_per, rank);  _copy_from_np(W2_T_d, W2_T_i)

        # Intermediates
        Y_d  = dt.DTensor(world, B * T * F_per, rank)   # pre-activation (B,T,F_per)
        H1_d = dt.DTensor(world, B * T * F_per, rank)   # post-GELU     (B,T,F_per)
        Z_d  = dt.DTensor(world, B * T * H, rank)       # partial logits (B,T,H)

        # ---------- forward (3D with strided-batched GEMM) ----------
        # Y = X @ W1_i ; per-batch multiply of (T,H) @ (H,F_per) -> (T,F_per)

        # Strides are in elements (row-major): strideA=T*H, strideB=0 (shared weights), strideC=T*F_per
        pg.gemm_strided_batched_f32(
            X_d.device_ptr(), W1_d.device_ptr(), Y_d.device_ptr(),
            T, F_per, H,
            T*H, 0, T*F_per, B
        )

        # Host GELU + b1
        Y_host = _to_numpy(Y_d).reshape(B, T, F_per).astype(np.float32)
        if args.debug_fwd:
            print_debug(rank, "Y = X@W1 (pre-bias)", Y_host, args.debug_max_elems)
            print_stats(rank, "Y_pre", Y_host)
        preact = (Y_host + b1_i).astype(np.float32)
        H1_host = gelu(preact).astype(np.float32)
        if args.debug_fwd:
            print_debug(rank, "Y + b1", preact, args.debug_max_elems)
            print_debug(rank, "H1 = GELU(Y + b1)", H1_host, args.debug_max_elems)
            print_stats(rank, "H1", H1_host)
        _copy_from_np(H1_d, H1_host)

        # Z_partial = H1 @ W2_i ; per-batch (T,F_per) @ (F_per,H) -> (T,H)
        pg.gemm_strided_batched_f32(
            H1_d.device_ptr(), W2_d.device_ptr(), Z_d.device_ptr(),
            T, H, F_per,
            T*F_per, 0, T*H, B
        )

        # tensor-parallel all-reduce on device to sum partials
        pg.all_reduce_f32(Z_d.device_ptr(), Z_d.size()).wait()

        Z_host = _to_numpy(Z_d).reshape(B, T, H).astype(np.float32)
        Z_host += b2  # add bias on host (or on device if you prefer)
        if args.debug_fwd:
            print_debug(rank, "b2", b2, args.debug_max_elems)
            print_debug(rank, "Z_final = allreduce(Z_partial) + b2", Z_host, args.debug_max_elems)
            print_stats(rank, "Z_final", Z_host)

        if rank == 0:
            print(f"[rank {rank}] Forward done.", flush=True)

        # allocator stats after forward
        if hasattr(pg, "print_allocator_stats"):
            pg.print_allocator_stats("after forward")
        else:
            _allocator_print("after forward")

        # ---------- backward ----------
        # Seed gradient dZ with ones over (B,T,H) or from your loss
        dZ_host = np.ones((B, T, H), dtype=np.float32)
        if args.debug_bwd:
            print_debug(rank, "dZ (seed grad)", dZ_host, args.debug_max_elems)
        dZ_d = dt.DTensor(world, B * T * H, rank); _copy_from_np(dZ_d, dZ_host)

        # 1) dH = dZ @ W2^T ; per-batch (T,H) @ (H,F_per) -> (T,F_per)
        dH_d = dt.DTensor(world, B * T * F_per, rank)
        pg.gemm_strided_batched_f32(
            dZ_d.device_ptr(), W2_T_d.device_ptr(), dH_d.device_ptr(),
            T, F_per, H,
            T*H, 0, T*F_per, B
        )

        # 2) dY = dH * GELU'(Y + b1)
        dH_host = _to_numpy(dH_d).reshape(B, T, F_per)
        dY_host = (dH_host * gelu_prime(preact)).astype(np.float32)
        if args.debug_bwd:
            print_debug(rank, "GELU'(Y+b1)", gelu_prime(preact), args.debug_max_elems)
            print_debug(rank, "dY = dH * GELU'(Y+b1)", dY_host, args.debug_max_elems)
            print_stats(rank, "dY", dY_host)
        dY_d = dt.DTensor(world, B * T * F_per, rank); _copy_from_np(dY_d, dY_host)

        # 3) dX_partial = dY @ W1^T ; per-batch (T,F_per) @ (F_per,H) -> (T,H)
        dX_d = dt.DTensor(world, B * T * H, rank)
        pg.gemm_strided_batched_f32(
            dY_d.device_ptr(), W1_T_d.device_ptr(), dX_d.device_ptr(),
            T, H, F_per,
            T*F_per, 0, T*H, B
        )

        # Sum partial dX across TP shards -> full dX
        pg.all_reduce_f32(dX_d.device_ptr(), dX_d.size()).wait()  # FIXED: reduce dX, not Z

        # ----- parameter grads (use big 2-D GEMMs on flattened BT) -----
        # dW2_i = H1^T @ dZ ; (F_per, BT) @ (BT, H) -> (F_per, H)
        H1_T_flat = H1_host.reshape(BT, F_per).T.copy()  # (F_per, BT)
        dZ_flat   = dZ_host.reshape(BT, H).copy()        # (BT, H)
        H1_T_d = dt.DTensor(world, F_per * BT, rank); _copy_from_np(H1_T_d, H1_T_flat)
        dZ_flat_d = dt.DTensor(world, BT * H, rank); _copy_from_np(dZ_flat_d, dZ_flat)
        dW2_d = dt.DTensor(world, F_per * H, rank)
        pg.gemm_f32(H1_T_d.device_ptr(), dZ_flat_d.device_ptr(), dW2_d.device_ptr(), F_per, H, BT)

        # dW1_i = X^T @ dY ; (H, BT) @ (BT, F_per) -> (H, F_per)
        X_T_flat = X.reshape(BT, H).T.copy()             # (H, BT)
        dY_flat  = dY_host.reshape(BT, F_per).copy()     # (BT, F_per)
        X_T_d = dt.DTensor(world, H * BT, rank); _copy_from_np(X_T_d, X_T_flat)
        dY_flat_d = dt.DTensor(world, BT * F_per, rank); _copy_from_np(dY_flat_d, dY_flat)
        dW1_d = dt.DTensor(world, H * F_per, rank)
        pg.gemm_f32(X_T_d.device_ptr(), dY_flat_d.device_ptr(), dW1_d.device_ptr(), H, F_per, BT)

        # Bias grads
        db1_i = dY_host.sum(axis=(0,1)).astype(np.float32)   # (F_per,)
        db2_local = dZ_host.sum(axis=(0,1)).astype(np.float32)  # (H,)
        if args.debug_bwd:
            print_debug(rank, "db1 shard", db1_i, args.debug_max_elems)
            print_stats(rank, "db1 shard", db1_i)
            print_debug(rank, "db2 local", db2_local, args.debug_max_elems)

        # (Optional) if b2 is replicated across TP, all-reduce db2_local here

        # bring grads to host for logging / final prints
        dW1_i = _to_numpy(dW1_d).reshape(H, F_per)
        dW2_i = _to_numpy(dW2_d).reshape(F_per, H)
        dX    = _to_numpy(dX_d).reshape(B, T, H)

        if args.debug_bwd:
            print_debug(rank, "dX_final (after all-reduce)", dX, args.debug_max_elems)
            print_stats(rank, "dX_final", dX)
        print_debug(rank, "dW1_i", dW1_i, args.debug_max_elems)
        print_debug(rank, "dW2_i", dW2_i, args.debug_max_elems)
        print_debug(rank, "dX", dX, args.debug_max_elems)

        # allocator stats after backward
        if hasattr(pg, "print_allocator_stats"):
            pg.print_allocator_stats("after backward")
        else:
            _allocator_print("after backward")

    except SystemExit:
        raise
    except BaseException as e:
        prior_exc = e
        print(f"[rank {rank}] EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        try:
            dt.mpi_barrier()
            dt.mpi_finalize()
        except BaseException as fe:
            print(f"[rank {rank}] Finalize error: {fe}", flush=True)
            if prior_exc is None:
                raise

if __name__ == "__main__":
    main()
