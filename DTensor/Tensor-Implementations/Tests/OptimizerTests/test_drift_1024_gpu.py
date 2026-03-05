#!/usr/bin/env python3
"""
test_drift_1024_gpu.py

GPU-only long-horizon drift test: 500 steps, 1024x1024 tensors.
Runs PyTorch CUDA reference for Adam and AdamW, then compares against C++ CSV.

Uses the same LCG PRNG as C++ for gradient generation.
Only 32 sampled elements (first 16 + last 16) are compared per step.

Usage:
  1. First run the C++ test:
       make run-snippet FILE=Tests/OptimizerTests/test_drift_1024_gpu.cpp
  2. Then run this script:
       cd Tests/OptimizerTests && python3 test_drift_1024_gpu.py
"""
import torch
import csv
import os
import sys

# ============================================================================
# Configuration
# ============================================================================
NUM_STEPS = 5000
ROWS = 1024
COLS = 1024
NUMEL = ROWS * COLS  # 1,048,576

# Sample indices: first 16 + last 16 (must match C++)
NUM_SAMPLES = 32
SAMPLE_INDICES = list(range(16)) + list(range(NUMEL - 16, NUMEL))


# ============================================================================
# Deterministic LCG PRNG — must match C++ exactly
# ============================================================================
def lcg_float(seed: int) -> float:
    """Returns a float in [-1.0, 1.0) given a seed. Matches C++ lcg_float()."""
    MOD = 2**64
    A = 6364136223846793005
    C = 1442695040888963407
    state = seed % MOD
    state = (A * state + C) % MOD
    state = (A * state + C) % MOD
    bits = (state >> 33) & 0x7FFFFFFF
    return (float(bits) / float(0x7FFFFFFF)) * 2.0 - 1.0


def make_gradient(step: int, device: torch.device) -> torch.Tensor:
    """Generate 1024x1024 gradient with deterministic values matching C++.
    
    Only computes the 32 sampled elements' positions fully via LCG, but we need
    ALL elements to match C++ since the optimizer processes the full tensor.
    We generate the full tensor using LCG for correctness.
    """
    # For 1M elements, pure Python LCG is slow (~2s/step).
    # Use vectorized approach: pre-compute all seeds, run LCG in numpy
    import numpy as np
    
    seeds = np.arange(NUMEL, dtype=np.uint64) + np.uint64(step * 2000000)
    
    A = np.uint64(6364136223846793005)
    C = np.uint64(1442695040888963407)
    
    state = seeds
    state = A * state + C  # numpy uint64 naturally wraps mod 2^64
    state = A * state + C
    
    bits = (state >> np.uint64(33)).astype(np.uint32) & np.uint32(0x7FFFFFFF)
    values = (bits.astype(np.float32) / np.float32(0x7FFFFFFF)) * 2.0 - 1.0
    
    grad = torch.from_numpy(values.reshape(ROWS, COLS)).to(device)
    return grad


# ============================================================================
# CSV utilities
# ============================================================================
def write_csv(path: str, steps_data: list):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['step'] + [f'p{idx}' for idx in SAMPLE_INDICES]
        writer.writerow(header)
        for step, values in steps_data:
            row = [step] + [f'{v:.8f}' for v in values]
            writer.writerow(row)


def read_csv(path: str):
    data = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            step = int(row[0])
            values = [float(x) for x in row[1:]]
            data[step] = values
    return data


def sample_tensor(t: torch.Tensor) -> list:
    flat = t.detach().cpu().flatten()
    return [flat[idx].item() for idx in SAMPLE_INDICES]


# ============================================================================
# PyTorch reference runners (GPU only)
# ============================================================================
def make_initial_weights(device: torch.device) -> torch.Tensor:
    W = torch.zeros(ROWS, COLS, dtype=torch.float32, requires_grad=True, device=device)
    with torch.no_grad():
        flat = W.view(-1)
        for i in range(NUMEL):
            flat[i] = 0.1 * (i + 1) / float(NUMEL)
    return W


def run_adam(device: torch.device, csv_path: str):
    print(f"\n=== PyTorch Adam (CUDA) — {NUM_STEPS} steps, [{ROWS}x{COLS}] ===")
    W = make_initial_weights(device)
    optimizer = torch.optim.Adam([W], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    steps_data = [(0, sample_tensor(W))]
    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        W.grad = make_gradient(step, device)
        optimizer.step()
        steps_data.append((step, sample_tensor(W)))
        if step % 100 == 0:
            print(f"  Step {step}/{NUM_STEPS}")

    write_csv(csv_path, steps_data)
    print(f"  Wrote {csv_path}")


def run_adamw(device: torch.device, csv_path: str):
    print(f"\n=== PyTorch AdamW (CUDA) — {NUM_STEPS} steps, [{ROWS}x{COLS}] ===")
    W = make_initial_weights(device)
    optimizer = torch.optim.AdamW([W], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    steps_data = [(0, sample_tensor(W))]
    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        W.grad = make_gradient(step, device)
        optimizer.step()
        steps_data.append((step, sample_tensor(W)))
        if step % 100 == 0:
            print(f"  Step {step}/{NUM_STEPS}")

    write_csv(csv_path, steps_data)
    print(f"  Wrote {csv_path}")


# ============================================================================
# Comparison
# ============================================================================
def compare_csvs(own_path: str, pytorch_path: str, name: str, tolerance: float = 1e-5):
    if not os.path.exists(own_path):
        print(f"  [SKIP] {name}: C++ CSV not found at {own_path}")
        print(f"         Run: make run-snippet FILE=Tests/OptimizerTests/test_drift_1024_gpu.cpp")
        return None

    own_data = read_csv(own_path)
    pytorch_data = read_csv(pytorch_path)

    max_dev = 0.0
    worst_step = 0
    worst_elem_idx = 0
    num_steps_checked = 0

    for step in sorted(pytorch_data.keys()):
        if step not in own_data:
            print(f"  [FAIL] {name}: Step {step} missing from C++ output")
            return False

        own_vals = own_data[step]
        pt_vals = pytorch_data[step]

        for i in range(len(pt_vals)):
            dev = abs(own_vals[i] - pt_vals[i])
            if dev > max_dev:
                max_dev = dev
                worst_step = step
                worst_elem_idx = i
        num_steps_checked += 1

    passed = max_dev < tolerance

    print(f"\n  --- {name} ---")
    print(f"  Tensor shape    : [{ROWS}x{COLS}] ({NUMEL:,} elements)")
    print(f"  Elements sampled: {NUM_SAMPLES}")
    print(f"  Steps compared  : {num_steps_checked}")
    print(f"  Max deviation   : {max_dev:.2e}")
    print(f"  Worst at        : step={worst_step}, element=p{SAMPLE_INDICES[worst_elem_idx]}")
    if passed:
        print(f"  Result          : [PASS] (tolerance={tolerance:.0e})")
    else:
        own_val = own_data[worst_step][worst_elem_idx]
        pt_val = pytorch_data[worst_step][worst_elem_idx]
        print(f"  OwnTensor value : {own_val:.8f}")
        print(f"  PyTorch value   : {pt_val:.8f}")
        print(f"  Result          : [FAIL] deviation {max_dev:.2e} > tolerance {tolerance:.0e}")

    return passed


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("  GPU Drift Test — 1024x1024 Tensors, 500 Steps")
    print("  PyTorch Reference + Comparison")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")

    # Run PyTorch references
    print("\n" + "-" * 40)
    print("Running PyTorch references...")
    print("-" * 40)

    run_adam(device, "drift_1024_adam_gpu_pytorch.csv")
    run_adamw(device, "drift_1024_adamw_gpu_pytorch.csv")

    # Compare
    print("\n" + "=" * 70)
    print("  Comparing OwnTensor vs PyTorch (GPU, 1024x1024)")
    print("=" * 70)

    results = []
    results.append(compare_csvs("drift_1024_adam_gpu.csv", "drift_1024_adam_gpu_pytorch.csv", "Adam (GPU)"))
    results.append(compare_csvs("drift_1024_adamw_gpu.csv", "drift_1024_adamw_gpu_pytorch.csv", "AdamW (GPU)"))

    # Summary
    print("\n" + "=" * 70)
    valid = [r for r in results if r is not None]
    passed = sum(1 for r in valid if r)
    total = len(valid)
    skipped = len(results) - total

    print(f"  Overall: {passed}/{total} PASSED", end="")
    if skipped > 0:
        print(f", {skipped} SKIPPED", end="")
    print()

    if passed == total and total > 0:
        print("All GPU 1024x1024 drift tests PASSED!")
    else:
        print("Some tests FAILED or were skipped.")
    print("=" * 70)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
