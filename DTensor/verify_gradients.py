#!/usr/bin/env python3
"""
Verification script to compare gradients between normal MLP (mlp_seed) 
and tensor parallel MLP (tensor_parallel_mlp_seed).

Both MLPs use IDENTICAL fixed data so we can directly compare:
- The normal MLP is the ground truth (known correct)
- The tensor parallel MLP should produce equivalent results

For tensor parallelism:
- Sharded gradients (W1_Shard, W2_Shard, etc.) should match the corresponding
  slices of the full gradients from the normal MLP

Usage:
    python3 verify_gradients.py
"""

import subprocess
import sys
import re
import numpy as np
from pathlib import Path

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# ============ PARSING FUNCTIONS ============

def parse_tensor_array(lines, start_idx):
    """Parse a tensor array from lines starting at start_idx."""
    tensor_lines = []
    bracket_count = 0
    idx = start_idx
    
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
            
        if bracket_count == 0 and not line.startswith('['):
            idx += 1
            continue
            
        tensor_lines.append(line)
        bracket_count += line.count('[') - line.count(']')
        idx += 1
        
        if bracket_count == 0 and tensor_lines:
            break
    
    if not tensor_lines:
        return None, idx
    
    tensor_str = ' '.join(tensor_lines)
    tensor_str = re.sub(r'\],\s*\n?\s*\[', '], [', tensor_str)
    
    try:
        tensor_str = re.sub(r',(\s*\])', r'\1', tensor_str)
        arr = np.array(eval(tensor_str))
        return arr, idx
    except Exception as e:
        return None, idx


def parse_mlp_output(text, is_tensor_parallel=False):
    """Parse the output from either MLP implementation."""
    lines = text.split('\n')
    tensors = {}
    
    # Track occurrences for duplicate markers
    h_count = 0
    w1_shard_count = 0
    w2_shard_count = 0
    w3_shard_count = 0
    w4_shard_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if 'Tensor(shape=' in line:
            tensor_name = None
            
            for j in range(i-1, max(0, i-10), -1):
                prev_line = lines[j].strip().lower()
                
                # Gradients
                if is_tensor_parallel:
                    if 'w1_shard gradient' in prev_line:
                        tensor_name = 'W1_Shard_grad'
                        break
                    elif 'w2_shard gradient' in prev_line:
                        tensor_name = 'W2_Shard_grad'
                        break
                    elif 'w3_shard gradient' in prev_line:
                        tensor_name = 'W3_Shard_grad'
                        break
                    elif 'w4_shard gradient' in prev_line:
                        tensor_name = 'W4_Shard_grad'
                        break
                else:
                    if 'w1 gradient' in prev_line:
                        tensor_name = 'W1_grad'
                        break
                    elif 'w2 gradient' in prev_line:
                        tensor_name = 'W2_grad'
                        break
                    elif 'w3 gradient' in prev_line:
                        tensor_name = 'W3_grad'
                        break
                    elif 'w4 gradient' in prev_line:
                        tensor_name = 'W4_grad'
                        break
                
                # Forward pass tensors
                if 'x - dtensor' in prev_line:
                    tensor_name = 'X'
                    break
                    
                # Weights
                if is_tensor_parallel:
                    if 'w1 shard' in prev_line:
                        tensor_name = f'W1_Shard_gpu{w1_shard_count}'
                        w1_shard_count += 1
                        break
                    elif 'w2 shard' in prev_line:
                        tensor_name = f'W2_Shard_gpu{w2_shard_count}'
                        w2_shard_count += 1
                        break
                    elif 'w3 shard' in prev_line:
                        tensor_name = f'W3_Shard_gpu{w3_shard_count}'
                        w3_shard_count += 1
                        break
                    elif 'w4 shard' in prev_line:
                        tensor_name = f'W4_Shard_gpu{w4_shard_count}'
                        w4_shard_count += 1
                        break
                else:
                    if 'w1 - dtensor' in prev_line:
                        tensor_name = 'W1'
                        break
                    elif 'w2 - dtensor' in prev_line:
                        tensor_name = 'W2'
                        break
                    elif 'w3 - dtensor' in prev_line:
                        tensor_name = 'W3'
                        break
                    elif 'w4 dtensor' in prev_line:
                        tensor_name = 'W4'
                        break
                
                # Outputs
                if is_tensor_parallel:
                    if 'y2 after sync' in prev_line:
                        tensor_name = 'Y2_after'
                        break
                else:
                    if 'y2 dtensor' in prev_line:
                        tensor_name = 'Y2'
                        break
            
            if tensor_name and tensor_name not in tensors:
                arr, next_i = parse_tensor_array(lines, i + 1)
                if arr is not None:
                    tensors[tensor_name] = arr
                i = next_i
                continue
        
        i += 1
    
    return tensors


def run_command(cmd, cwd):
    """Run a command and capture its output."""
    print(f"  Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        return output, result.returncode
    except subprocess.TimeoutExpired:
        print("  Command timed out!")
        return "", 1
    except Exception as e:
        print(f"  Error running command: {e}")
        return "", 1


def compare_gradients(normal_tensors, tp_tensors):
    """
    Compare gradients between normal MLP and tensor parallel MLP.
    
    For tensor parallel:
    - W1_Shard (column parallel, dim 2): gradient should match W1_grad[:, :, :F/2]
    - W2_Shard (row parallel, dim 1): gradient should match W2_grad[:, :F/2, :]
    - W3_Shard (column parallel, dim 2): gradient should match W3_grad[:, :, :F/2]
    - W4_Shard (row parallel, dim 1): gradient should match W4_grad[:, :F/2, :]
    """
    results = {}
    
    print("\n" + "="*60)
    print("\tGRADIENT COMPARISON: Normal MLP vs Tensor Parallel MLP")
    print("="*60)
    
    comparisons = [
        ('W1', 'W1_Shard_grad', 'W1_grad', 'column', 2),  # Column parallel, shard dim 2
        ('W2', 'W2_Shard_grad', 'W2_grad', 'row', 1),     # Row parallel, shard dim 1
        ('W3', 'W3_Shard_grad', 'W3_grad', 'column', 2),  # Column parallel, shard dim 2
        ('W4', 'W4_Shard_grad', 'W4_grad', 'row', 1),     # Row parallel, shard dim 1
    ]
    
    for name, tp_key, normal_key, parallel_type, shard_dim in comparisons:
        print(f"\n--- {name} Gradient ({parallel_type}-parallel, shard dim {shard_dim}) ---")
        
        if tp_key not in tp_tensors:
            print(f"  WARNING: {tp_key} not found in tensor parallel output")
            results[name] = ('MISSING', None)
            continue
            
        if normal_key not in normal_tensors:
            print(f"  WARNING: {normal_key} not found in normal MLP output")
            results[name] = ('MISSING', None)
            continue
        
        tp_grad = tp_tensors[tp_key]
        normal_grad = normal_tensors[normal_key]
        
        print(f"  TP Shard gradient shape: {tp_grad.shape}")
        print(f"  Normal full gradient shape: {normal_grad.shape}")
        
        # Extract the corresponding slice from normal gradient
        # GPU 0 gets the first half of the sharded dimension
        shard_size = tp_grad.shape[shard_dim]
        
        if shard_dim == 2:
            expected_grad = normal_grad[:, :, :shard_size]
        elif shard_dim == 1:
            expected_grad = normal_grad[:, :shard_size, :]
        else:
            print(f"  ERROR: Unexpected shard dimension {shard_dim}")
            results[name] = ('ERROR', None)
            continue
        
        print(f"\n  Tensor Parallel Gradient (GPU 0 shard):")
        print(f"  {tp_grad}")
        
        print(f"\n  Expected (slice from normal MLP):")
        print(f"  {expected_grad}")
        
        # Compare
        max_diff = np.max(np.abs(tp_grad - expected_grad))
        match = max_diff < 0.001
        results[name] = ('PASS' if match else 'FAIL', max_diff)
        
        print(f"\n  Max difference: {max_diff:.6f} -> {'PASS' if match else 'FAIL'}")
    
    return results


def print_summary(results):
    """Print overall verification summary."""
    print("\n" + "="*60)
    print("\t\tVERIFICATION SUMMARY")
    print("="*60)
    
    all_pass = True
    
    for name, (status, diff) in results.items():
        if diff is not None:
            print(f"  {name} gradient: {status} (max diff: {diff:.6f})")
        else:
            print(f"  {name} gradient: {status}")
        if status != 'PASS':
            all_pass = False
    
    print()
    if all_pass and results:
        print("  ✓ ALL GRADIENT COMPARISONS PASSED!")
        print("  The tensor parallel implementation is CORRECT.")
    elif results:
        print("  ✗ Some gradient comparisons failed. Check details above.")
    else:
        print("  ⚠ No gradients were compared. Check parsing.")
    print()


def main():
    script_dir = Path(__file__).parent
    
    print("="*60)
    print("  Gradient Verification: mlp_seed vs tensor_parallel_mlp_seed")
    print("="*60)
    
    # Step 1: Run normal MLP (mlp_seed)
    print("\n[1] Running Normal MLP (mlp_seed) with fixed data...")
    normal_cmd = "mpirun --oversubscribe -np 2 ./examples/mlp_seed"
    normal_output, ret1 = run_command(normal_cmd, script_dir)
    
    if ret1 != 0:
        print(f"  Warning: Normal MLP returned non-zero exit code: {ret1}")
    
    # Step 2: Run tensor parallel MLP (tensor_parallel_mlp_seed)
    print("\n[2] Running Tensor Parallel MLP (tensor_parallel_mlp_seed) with fixed data...")
    tp_cmd = "mpirun --oversubscribe -np 2 ./examples/tensor_parallel_mlp_seed"
    tp_output, ret2 = run_command(tp_cmd, script_dir)
    
    if ret2 != 0:
        print(f"  Warning: Tensor Parallel MLP returned non-zero exit code: {ret2}")
    
    # Step 3: Parse outputs
    print("\n[3] Parsing outputs...")
    
    normal_tensors = parse_mlp_output(normal_output, is_tensor_parallel=False)
    print(f"  Parsed {len(normal_tensors)} tensors from normal MLP")
    grad_keys = [k for k in normal_tensors.keys() if 'grad' in k.lower()]
    print(f"  Gradients: {grad_keys}")
    
    tp_tensors = parse_mlp_output(tp_output, is_tensor_parallel=True)
    print(f"  Parsed {len(tp_tensors)} tensors from tensor parallel MLP")
    tp_grad_keys = [k for k in tp_tensors.keys() if 'grad' in k.lower()]
    print(f"  Gradients: {tp_grad_keys}")
    
    # Step 4: Compare gradients
    print("\n[4] Comparing gradients...")
    results = compare_gradients(normal_tensors, tp_tensors)
    
    # Step 5: Print summary
    print_summary(results)
    
    # Return exit code based on results
    all_pass = all(status == 'PASS' for status, _ in results.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
