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
    """Print caching allocator stats with a tag, using whichever binding exists."""
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

        B = int(args.batch)
        H = args.hidden
        F = args.ffn
        tp = world
        F_per = F // tp

        
        rng_in = np.random.default_rng(args.seed)
        X = rng_in.standard_normal((B, H), dtype=np.float32)

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

        
        X_d   = dt.DTensor(world, B * H, rank);       _copy_from_np(X_d, X)
        W1_d  = dt.DTensor(world, H * F_per, rank);   _copy_from_np(W1_d, W1_i)
        W2_d  = dt.DTensor(world, F_per * H, rank);   _copy_from_np(W2_d, W2_i)
        b1_d  = dt.DTensor(world, F_per, rank);       _copy_from_np(b1_d, b1_i)
        b2_d  = dt.DTensor(world, H, rank);           _copy_from_np(b2_d, b2)

        
        W1_T_i = W1_i.T.copy()             
        W2_T_i = W2_i.T.copy()             
        X_T    = X.T.copy()                

        W1_T_d = dt.DTensor(world, F_per * H, rank);  _copy_from_np(W1_T_d, W1_T_i)
        W2_T_d = dt.DTensor(world, H * F_per, rank);  _copy_from_np(W2_T_d, W2_T_i)
        X_T_d  = dt.DTensor(world, H * B, rank);      _copy_from_np(X_T_d, X_T)

        
        del W1_T_i, W2_T_i, X_T, W1_i, W2_i
        gc.collect()

        
        Y_d = dt.DTensor(world, B * F_per, rank)
        H_d = dt.DTensor(world, B * F_per, rank)
        Z_d = dt.DTensor(world, B * H, rank)

        # ---------- forward (GPU matmuls) ----------
        # Y_i = X @ W1_i
        pg.gemm_f32(X_d.device_ptr(), W1_d.device_ptr(), Y_d.device_ptr(), B, F_per, H)
        Y_pre = _to_numpy(Y_d).reshape(B, F_per)
        if args.debug_fwd:
            print_debug(rank, "Y = X@W1 (pre-bias)", Y_pre, args.debug_max_elems)
            print_stats(rank, "Y_pre", Y_pre)

        # bias + GELU on host
        Y_host = (Y_pre + b1_i).astype(np.float32)
        H_host = gelu(Y_host).astype(np.float32)
        if args.debug_fwd:
            print_debug(rank, "Y + b1", Y_host, args.debug_max_elems)
            print_debug(rank, "H = GELU(Y + b1)", H_host, args.debug_max_elems)
            print_stats(rank, "H", H_host)
        _copy_from_np(H_d, H_host)

        # Z_i = H_i @ W2_i
        pg.gemm_f32(H_d.device_ptr(), W2_d.device_ptr(), Z_d.device_ptr(), B, H, F_per)
        Z_part = _to_numpy(Z_d).reshape(B, H)
        if args.debug_fwd:
            print_debug(rank, "Z_partial = H@W2 (before all-reduce)", Z_part, args.debug_max_elems)
            print_stats(rank, "Z_partial", Z_part)

        # all-reduce partials on device -> full Z
        pg.all_reduce_f32(Z_d.device_ptr(), Z_d.size()).wait()

        Z = _to_numpy(Z_d).reshape(B, H)
        Z += b2
        if args.debug_fwd:
            print_debug(rank, "b2", b2, args.debug_max_elems)
            print_debug(rank, "Z_final = allreduce(Z_partial) + b2", Z, args.debug_max_elems)
            print_stats(rank, "Z_final", Z)

        if rank == 0:
            print(f"[rank {rank}] Forward done.", flush=True)

        # NEW: allocator stats after forward
        pg.print_allocator_stats("after forward")


        # ---------- backward (all main GEMMs on GPU) ----------
        # Seed gradient
        dZ = np.ones((B, H), dtype=np.float32)
        if args.debug_bwd:
            print_debug(rank, "dZ (seed grad)", dZ, args.debug_max_elems)
        dZ_d = dt.DTensor(world, B * H, rank); _copy_from_np(dZ_d, dZ)

        # H^T
        H_T = H_host.T.copy()  # (F_per, B)
        if args.debug_bwd:
            print_debug(rank, "H^T (host)", H_T, args.debug_max_elems)
        H_T_d = dt.DTensor(world, F_per * B, rank); _copy_from_np(H_T_d, H_T)

        # dW2_i = H^T @ dZ
        dW2_d = dt.DTensor(world, F_per * H, rank)
        pg.gemm_f32(H_T_d.device_ptr(), dZ_d.device_ptr(), dW2_d.device_ptr(), F_per, H, B)
        if args.debug_bwd and args.device_debug_copy:
            dW2_host = _to_numpy(dW2_d).reshape(F_per, H)
            print_debug(rank, "dW2 = H^T@dZ (device->host)", dW2_host, args.debug_max_elems)
            print_stats(rank, "dW2", dW2_host)

        # dH = dZ @ W2^T
        dH_d = dt.DTensor(world, B * F_per, rank)
        pg.gemm_f32(dZ_d.device_ptr(), W2_T_d.device_ptr(), dH_d.device_ptr(), B, F_per, H)
        if args.debug_bwd and args.device_debug_copy:
            dH_host_dbg = _to_numpy(dH_d).reshape(B, F_per)
            print_debug(rank, "dH = dZ@W2^T (device->host)", dH_host_dbg, args.debug_max_elems)
            print_stats(rank, "dH", dH_host_dbg)

        # No longer need these
        del H_T_d, dZ_d
        gc.collect()

        # dY = dH * GELU'(Y)
        dH_host = _to_numpy(dH_d).reshape(B, F_per)
        del dH_d
        gc.collect()
        dY_host = (dH_host * gelu_prime(Y_host)).astype(np.float32)
        if args.debug_bwd:
            print_debug(rank, "GELU'(Y)", gelu_prime(Y_host), args.debug_max_elems)
            print_debug(rank, "dY = dH * GELU'(Y)", dY_host, args.debug_max_elems)
            print_stats(rank, "dY", dY_host)
        dY_d = dt.DTensor(world, B * F_per, rank); _copy_from_np(dY_d, dY_host)

        # dW1_i = X^T @ dY
        dW1_d = dt.DTensor(world, H * F_per, rank)
        pg.gemm_f32(X_T_d.device_ptr(), dY_d.device_ptr(), dW1_d.device_ptr(), H, F_per, B)
        if args.debug_bwd and args.device_debug_copy:
            dW1_host_dbg = _to_numpy(dW1_d).reshape(H, F_per)
            print_debug(rank, "dW1 = X^T@dY (device->host)", dW1_host_dbg, args.debug_max_elems)
            print_stats(rank, "dW1", dW1_host_dbg)

        # dX_i = dY @ W1^T
        dX_d = dt.DTensor(world, B * H, rank)
        pg.gemm_f32(dY_d.device_ptr(), W1_T_d.device_ptr(), dX_d.device_ptr(), B, H, F_per)

        # before all-reduce
        if args.debug_bwd and args.device_debug_copy:
            dX_part = _to_numpy(dX_d).reshape(B, H)
            print_debug(rank, "dX_partial = dY@W1^T (before all-reduce)", dX_part, args.debug_max_elems)
            print_stats(rank, "dX_partial", dX_part)

        del X_T_d, W1_T_d, dY_d
        gc.collect()

        # dX = sum over TP shards
        pg.all_reduce_f32(Z_d.device_ptr(), Z_d.size()).wait()


        # (Optional) biases on host
        db1_i = dY_host.sum(axis=0).astype(np.float32)
        db2_local = dZ.sum(axis=0).astype(np.float32)
        if args.debug_bwd:
            print_debug(rank, "db1 shard", db1_i, args.debug_max_elems)
            print_stats(rank, "db1 shard", db1_i)
            print_debug(rank, "db2 local", db2_local, args.debug_max_elems)

        # bring grads to host for logging / final prints
        dW1_i = _to_numpy(dW1_d).reshape(H, F_per)
        dW2_i = _to_numpy(dW2_d).reshape(F_per, H)
        dX    = _to_numpy(dX_d).reshape(B, H)

        # after all-reduce
        if args.debug_bwd:
            print_debug(rank, "dX_final (after all-reduce)", dX, args.debug_max_elems)
            print_stats(rank, "dX_final", dX)

        # Summary prints (as before)
        print_debug(rank, "dW1_i", dW1_i, args.debug_max_elems)
        print_debug(rank, "dW2_i", dW2_i, args.debug_max_elems)
        print_debug(rank, "dX", dX, args.debug_max_elems)

        # NEW: allocator stats after backward
        pg.print_allocator_stats("after backward")

    except SystemExit:
        raise
    except BaseException as e:
        prior_exc = e
        print(f"[rank {rank}] EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        # Finalize MPI but don't hide failures.
        try:
            dt.mpi_barrier()
            dt.mpi_finalize()
        except BaseException as fe:
            print(f"[rank {rank}] Finalize error: {fe}", flush=True)
            if prior_exc is None:
                raise

if __name__ == "__main__":
    main()
