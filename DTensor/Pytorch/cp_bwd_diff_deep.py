"""Diff deep (full-tensor) CP-backward intermediates: PT vs C++.

Compares pt_*.bin and cpp_*.bin in /tmp/cp_bwd_test/deep/ produced when both
tests are run with DUMP_CP_DEEP=1. Reports the first label+step+rank where
C++ diverges from PT — that pins the exact op where the LB bug enters.
"""
import os
import sys
import glob
import numpy as np

DEEP_DIR = "/tmp/cp_bwd_test/deep"

# Labels (in chronological order within a step)
LABELS = [
    "grad_q_before",     # state at top of step
    "grad_q_step",       # SDPA backward output
    "gq_1st_clone",      # partial path only
    "gq_2nd_clone",      # partial path only
    "gq_2nd_plus_step",  # partial path only
    "grad_q_after",      # after accumulation
]


def load(path):
    return np.fromfile(path, dtype=np.float32)


def report(pt_path, cpp_path, label):
    pt = load(pt_path)
    cpp = load(cpp_path)
    if pt.size != cpp.size:
        print(f"  {label}: SIZE MISMATCH pt={pt.size} cpp={cpp.size}")
        return False
    diff = np.abs(pt - cpp)
    max_ad = float(diff.max())
    mean_ad = float(diff.mean())
    pt_max = float(np.abs(pt).max())
    pt_mean = float(np.abs(pt).mean())
    rel_max = max_ad / max(pt_max, 1e-30)
    rel_mean = mean_ad / max(pt_mean, 1e-30)
    # threshold: anything beyond 1e-4 relative is suspicious (fp32 noise floor ~1e-5)
    flag = " ✗ DIVERGE" if rel_max > 1e-4 else " ✓"
    print(f"  {label:24s} | n={pt.size:6d} max_ad={max_ad:.3e} "
          f"rel_max={rel_max:.3e} rel_mean={rel_mean:.3e}{flag}")
    return rel_max <= 1e-4


def main():
    if not os.path.isdir(DEEP_DIR):
        print(f"ERROR: {DEEP_DIR} does not exist. Run both tests with DUMP_CP_DEEP=1.")
        sys.exit(1)

    # Find which (step, rank) pairs exist
    pairs = set()
    for f in glob.glob(f"{DEEP_DIR}/pt_*.bin"):
        name = os.path.basename(f).removeprefix("pt_").removesuffix(".bin")
        # name = <label>_step<i>_rank<r>
        parts = name.rsplit("_rank", 1)
        if len(parts) != 2:
            continue
        r = parts[1]
        label_step = parts[0]
        label_step_parts = label_step.rsplit("_step", 1)
        if len(label_step_parts) != 2:
            continue
        step = label_step_parts[1]
        pairs.add((int(step), int(r)))

    if not pairs:
        print(f"ERROR: no pt_*.bin files in {DEEP_DIR}. Did DUMP_CP_DEEP=1 fire?")
        sys.exit(1)

    for step, r in sorted(pairs):
        print(f"\n=== step {step}, rank {r} ===")
        for label in LABELS:
            pt_path = f"{DEEP_DIR}/pt_{label}_step{step}_rank{r}.bin"
            cpp_path = f"{DEEP_DIR}/cpp_{label}_step{step}_rank{r}.bin"
            if not os.path.exists(pt_path) or not os.path.exists(cpp_path):
                continue
            report(pt_path, cpp_path, label)

    # Per-forward-step intermediates: block_out, block_lse, merged_out, merged_lse
    fwd_pairs = set()
    for f in glob.glob(f"{DEEP_DIR}/pt_*_fwdstep*_rank*.bin"):
        name = os.path.basename(f).removeprefix("pt_").removesuffix(".bin")
        try:
            r = int(name.rsplit("_rank", 1)[1])
            i = int(name.rsplit("_rank", 1)[0].rsplit("_fwdstep", 1)[1])
            fwd_pairs.add((i, r))
        except (ValueError, IndexError):
            pass
    for i, r in sorted(fwd_pairs):
        print(f"\n=== forward step {i}, rank {r} ===")
        for label in ("block_out", "block_lse", "merged_out", "merged_lse"):
            pt_path = f"{DEEP_DIR}/pt_{label}_fwdstep{i}_rank{r}.bin"
            cpp_path = f"{DEEP_DIR}/cpp_{label}_fwdstep{i}_rank{r}.bin"
            if not os.path.exists(pt_path) or not os.path.exists(cpp_path):
                continue
            report(pt_path, cpp_path, label)

    # Forward final outputs (no step index): merged_out, merged_lse
    fwd_ranks = set()
    for f in glob.glob(f"{DEEP_DIR}/pt_merged_out_rank*.bin"):
        name = os.path.basename(f).removeprefix("pt_merged_out_rank").removesuffix(".bin")
        try:
            fwd_ranks.add(int(name))
        except ValueError:
            pass
    for r in sorted(fwd_ranks):
        print(f"\n=== forward final, rank {r} ===")
        for label in ("merged_out", "merged_lse"):
            pt_path = f"{DEEP_DIR}/pt_{label}_rank{r}.bin"
            cpp_path = f"{DEEP_DIR}/cpp_{label}_rank{r}.bin"
            if not os.path.exists(pt_path) or not os.path.exists(cpp_path):
                print(f"  {label}: MISSING (pt={os.path.exists(pt_path)} cpp={os.path.exists(cpp_path)})")
                continue
            report(pt_path, cpp_path, label)


if __name__ == "__main__":
    main()
