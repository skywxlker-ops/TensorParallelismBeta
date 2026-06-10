"""Compare final dQ/dK/dV from PT vs C++ CP backward isolated test.

Loads /tmp/cp_bwd_test/{dQ,dK,dV}_{pt,cpp}_rank{r}.bin and reports
max-abs-diff, mean-abs-diff, and per-position correlation.

Run after both Pytorch/cp_bwd_isolated_test.py and
./cp_bwd_isolated_test_exec have been executed.
"""
import numpy as np

B, H, T_local, D = 1, 2, 64, 64
SHAPE = (B, H, T_local, D)
N = B * H * T_local * D
DUMP_DIR = "/tmp/cp_bwd_test"


def load(name, rank):
    arr = np.fromfile(f"{DUMP_DIR}/{name}_rank{rank}.bin", dtype=np.float32)
    if arr.size != N:
        raise RuntimeError(f"{name}_rank{rank}.bin: expected {N} floats, got {arr.size}")
    return arr.reshape(SHAPE)


def stat(pt, cpp, label):
    diff = pt - cpp
    abs_diff = np.abs(diff)
    pt_mag = np.abs(pt)
    max_ad = abs_diff.max()
    mean_ad = abs_diff.mean()
    pt_mean = pt_mag.mean()
    pt_max = pt_mag.max()
    rel_mean = mean_ad / max(pt_mean, 1e-12)
    rel_max = max_ad / max(pt_max, 1e-12)
    # Cosine similarity
    p = pt.flatten().astype(np.float64)
    c = cpp.flatten().astype(np.float64)
    cos = (p @ c) / (np.linalg.norm(p) * np.linalg.norm(c) + 1e-30)
    print(f"  {label:6} | max_abs_diff={max_ad:.3e}  mean_abs_diff={mean_ad:.3e}  "
          f"rel_max={rel_max:.3e}  rel_mean={rel_mean:.3e}  cos_sim={cos:.10f}")
    # Where is max diff?
    flat_diff = abs_diff.flatten()
    arg_max = int(flat_diff.argmax())
    b, h, t, d = np.unravel_index(arg_max, pt.shape)
    print(f"           argmax at [B={b}, H={h}, T={t}, D={d}]: PT={pt[b,h,t,d]:.6e}  "
          f"C++={cpp[b,h,t,d]:.6e}  diff={diff[b,h,t,d]:.3e}")
    # Per-T mean-abs-diff profile (where in the seq dim does the error live?)
    per_T = abs_diff.mean(axis=(0, 1, 3))  # mean over B, H, D → [T]
    chunk_sz = pt.shape[2] // 2
    head_mean = per_T[:chunk_sz].mean()
    tail_mean = per_T[chunk_sz:].mean()
    print(f"           per-T mean abs diff:  head [0:{chunk_sz}]={head_mean:.3e}  "
          f"tail [{chunk_sz}:]={tail_mean:.3e}  ratio={tail_mean/max(head_mean, 1e-30):.2f}x")


for rank in (0, 1):
    print(f"=== rank {rank} ===")
    for name in ("dQ", "dK", "dV"):
        pt = load(f"{name}_pt", rank)
        cpp = load(f"{name}_cpp", rank)
        stat(pt, cpp, name)
    print()
