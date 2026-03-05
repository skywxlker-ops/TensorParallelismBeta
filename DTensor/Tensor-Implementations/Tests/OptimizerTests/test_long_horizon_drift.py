#!/usr/bin/env python3
"""
test_long_horizon_drift.py

Long-horizon drift test: 500 steps with non-uniform gradients.
Runs PyTorch reference for Adam, AdamW, and SGD, then automatically
compares against C++ CSV output.

Usage:
  1. First run the C++ test:
       make run-snippet FILE=Tests/OptimizerTests/test_long_horizon_drift.cpp
  2. Then run this script from the OptimizerTests directory:
       cd Tests/OptimizerTests && python3 test_long_horizon_drift.py
"""
import torch
import numpy as np
import csv
import os
import sys


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

    # Extract top 31 bits (matching C++ "state >> 33")
    bits = (state >> 33) & 0x7FFFFFFF
    return (float(bits) / float(0x7FFFFFFF)) * 2.0 - 1.0


def make_gradient(step: int, numel: int, device: torch.device) -> torch.Tensor:
    """Generate gradient tensor with deterministic non-uniform values."""
    values = []
    for i in range(numel):
        seed = step * 1000 + i
        values.append(lcg_float(seed))
    grad = torch.tensor(values, dtype=torch.float32, device=device).reshape(4, 4)
    return grad


# ============================================================================
# CSV utilities
# ============================================================================
def write_csv(path: str, steps_data: list, numel: int):
    """Write per-step parameter values to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['step'] + [f'p{i}' for i in range(numel)]
        writer.writerow(header)
        for step, values in steps_data:
            row = [step] + [f'{v:.8f}' for v in values]
            writer.writerow(row)


def read_csv(path: str):
    """Read CSV and return dict: step -> list of float values."""
    data = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            step = int(row[0])
            values = [float(x) for x in row[1:]]
            data[step] = values
    return data


# ============================================================================
# PyTorch reference runners
# ============================================================================
NUM_STEPS = 5000
NUMEL = 16  # 4x4


def make_initial_weights(device: torch.device) -> torch.Tensor:
    """Create initial weights [0.1, 0.2, ..., 1.6] matching C++."""
    W = torch.zeros(4, 4, dtype=torch.float32, requires_grad=True, device=device)
    with torch.no_grad():
        for i in range(NUMEL):
            W.view(-1)[i] = 0.1 * (i + 1)
    return W


def run_adam_pytorch(device_str: str, csv_path: str):
    """Run PyTorch Adam for NUM_STEPS steps."""
    device = torch.device(device_str)
    print(f"\n=== PyTorch Adam ({device_str.upper()}) — {NUM_STEPS} steps ===")

    W = make_initial_weights(device)
    optimizer = torch.optim.Adam([W], lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.01)

    steps_data = []
    # Step 0: initial state
    steps_data.append((0, W.detach().cpu().flatten().tolist()))

    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        W.grad = make_gradient(step, NUMEL, device)
        optimizer.step()
        steps_data.append((step, W.detach().cpu().flatten().tolist()))

    write_csv(csv_path, steps_data, NUMEL)
    print(f"  Wrote {csv_path}")


def run_adamw_pytorch(device_str: str, csv_path: str):
    """Run PyTorch AdamW for NUM_STEPS steps."""
    device = torch.device(device_str)
    print(f"\n=== PyTorch AdamW ({device_str.upper()}) — {NUM_STEPS} steps ===")

    W = make_initial_weights(device)
    optimizer = torch.optim.AdamW([W], lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                                   weight_decay=0.01)

    steps_data = []
    steps_data.append((0, W.detach().cpu().flatten().tolist()))

    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        W.grad = make_gradient(step, NUMEL, device)
        optimizer.step()
        steps_data.append((step, W.detach().cpu().flatten().tolist()))

    write_csv(csv_path, steps_data, NUMEL)
    print(f"  Wrote {csv_path}")


def run_sgd_pytorch(device_str: str, csv_path: str):
    """Run PyTorch SGD (momentum + weight_decay) for NUM_STEPS steps."""
    device = torch.device(device_str)
    print(f"\n=== PyTorch SGD+Momentum+WD ({device_str.upper()}) — {NUM_STEPS} steps ===")

    W = make_initial_weights(device)
    optimizer = torch.optim.SGD([W], lr=0.01, momentum=0.9, weight_decay=0.01)

    steps_data = []
    steps_data.append((0, W.detach().cpu().flatten().tolist()))

    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        W.grad = make_gradient(step, NUMEL, device)
        optimizer.step()
        steps_data.append((step, W.detach().cpu().flatten().tolist()))

    write_csv(csv_path, steps_data, NUMEL)
    print(f"  Wrote {csv_path}")


# ============================================================================
# Comparison
# ============================================================================
def compare_csvs(own_path: str, pytorch_path: str, name: str, tolerance: float = 1e-5):
    """Compare two CSV files element-wise and report max deviation."""
    if not os.path.exists(own_path):
        print(f"  [SKIP] {name}: C++ CSV not found at {own_path}")
        print(f"         Run the C++ test first!")
        return False

    own_data = read_csv(own_path)
    pytorch_data = read_csv(pytorch_path)

    max_dev = 0.0
    worst_step = 0
    worst_elem = 0
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
                worst_elem = i
        num_steps_checked += 1

    passed = max_dev < tolerance

    print(f"\n  --- {name} ---")
    print(f"  Steps compared : {num_steps_checked}")
    print(f"  Max deviation  : {max_dev:.2e}")
    print(f"  Worst at       : step={worst_step}, element=p{worst_elem}")
    if passed:
        print(f"  Result         : [PASS] (tolerance={tolerance:.0e})")
    else:
        # Show the actual values at the worst point
        own_val = own_data[worst_step][worst_elem]
        pt_val = pytorch_data[worst_step][worst_elem]
        print(f"  OwnTensor value: {own_val:.8f}")
        print(f"  PyTorch value  : {pt_val:.8f}")
        print(f"  Result         : [FAIL] deviation {max_dev:.2e} > tolerance {tolerance:.0e}")

    return passed


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("  Long-Horizon Drift Test — PyTorch Reference + Comparison")
    print("=" * 70)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, GPU tests will be skipped")

    # ----- Run PyTorch references -----
    print("\n" + "-" * 40)
    print("Running PyTorch references...")
    print("-" * 40)

    run_adam_pytorch("cpu",  "drift_adam_cpu_pytorch.csv")
    run_adamw_pytorch("cpu", "drift_adamw_cpu_pytorch.csv")
    run_sgd_pytorch("cpu",   "drift_sgd_cpu_pytorch.csv")

    if cuda_available:
        run_adam_pytorch("cuda",  "drift_adam_gpu_pytorch.csv")
        run_adamw_pytorch("cuda", "drift_adamw_gpu_pytorch.csv")

    # ----- Compare against C++ output -----
    print("\n" + "=" * 70)
    print("  Comparing OwnTensor vs PyTorch")
    print("=" * 70)

    results = []

    results.append(compare_csvs("drift_adam_cpu.csv",  "drift_adam_cpu_pytorch.csv",  "Adam (CPU)"))
    results.append(compare_csvs("drift_adamw_cpu.csv", "drift_adamw_cpu_pytorch.csv", "AdamW (CPU)"))
    results.append(compare_csvs("drift_sgd_cpu.csv",   "drift_sgd_cpu_pytorch.csv",   "SGD (CPU)"))

    if cuda_available:
        results.append(compare_csvs("drift_adam_gpu.csv",  "drift_adam_gpu_pytorch.csv",  "Adam (GPU)"))
        results.append(compare_csvs("drift_adamw_gpu.csv", "drift_adamw_gpu_pytorch.csv", "AdamW (GPU)"))

    # ----- Summary -----
    print("\n" + "=" * 70)
    valid_results = [r for r in results if r is not None]
    passed = sum(1 for r in valid_results if r)
    total = len(valid_results)
    skipped = len(results) - total

    print(f"  Overall: {passed}/{total} PASSED", end="")
    if skipped > 0:
        print(f", {skipped} SKIPPED", end="")
    print()

    if passed == total and total > 0:
        print("All optimizer drift tests PASSED!")
    else:
        print("Some tests FAILED or were skipped.")
    print("=" * 70)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
