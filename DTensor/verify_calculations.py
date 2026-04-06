#!/usr/bin/env python3
"""
Verification script for tensor parallel MLP forward and backward passes.

This script parses the terminal output from tensor_parallel_mlp and verifies:
1. Linear layer forward pass: H = X @ W1_Shard + B1
2. Linear layer forward pass: Y = H @ W2_Shard + B2  
3. Backward gradients for W1_Shard and W2_Shard

Usage:
    mpirun -np 2 ./examples/tensor_parallel_mlp 2>&1 | python3 verify_calculations.py
    OR
    python3 verify_calculations.py < output_log.txt
"""

import sys
import re
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ============ PARSING FUNCTIONS ============

def parse_tensor_array(lines, start_idx):
    """
    Parse a tensor array from lines starting at start_idx.
    Tensor arrays are formatted as nested brackets: [[...], [...], ...]
    Returns (numpy_array, next_line_idx).
    """
    # Collect all lines that are part of the tensor array
    tensor_lines = []
    bracket_count = 0
    idx = start_idx
    
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
            
        # Skip header lines that don't start with '['
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
    
    # Join and parse the tensor string
    tensor_str = ' '.join(tensor_lines)
    # Replace commas between brackets with proper separators
    tensor_str = re.sub(r'\],\s*\n?\s*\[', '], [', tensor_str)
    
    try:
        # Clean up and evaluate
        # Remove trailing commas before brackets
        tensor_str = re.sub(r',(\s*\])', r'\1', tensor_str)
        arr = np.array(eval(tensor_str))
        return arr, idx
    except Exception as e:
        print(f"Warning: Failed to parse tensor array: {e}")
        print(f"Tensor string: {tensor_str[:200]}...")
        return None, idx


def find_section(lines, marker):
    """Find the line index containing the marker string."""
    for i, line in enumerate(lines):
        if marker in line:
            return i
    return -1


def parse_tensor_header(line):
    """Parse tensor header like 'Tensor(shape=(2, 4, 2), dtype=Float32, device='cuda:0')'"""
    shape_match = re.search(r'shape=\(([\d,\s]+)\)', line)
    if shape_match:
        shape_str = shape_match.group(1)
        shape = tuple(int(x.strip()) for x in shape_str.split(',') if x.strip())
        return shape
    return None


def parse_output(text):
    """
    Parse the entire tensor_parallel_mlp output and extract all tensors.
    Returns a dictionary with tensor names as keys and numpy arrays as values.
    """
    lines = text.split('\n')
    tensors = {}
    
    # Patterns to identify different tensor sections
    sections = [
        ('X', 'x - dtensor'),
        ('W1', 'w1 - dtensor'),
        ('W1_Shard_gpu0', 'w1 shard'),  # First occurrence is GPU 0
        ('W1_Shard_gpu1', None),  # Will be second w1 shard display
        ('B1', 'b1 - dtensor'),
        ('H_gpu0', 'h - dtensor'),  # First H display
        ('H_gpu1', None),  # Second H display
        ('W2', 'w2 - dtensor'),
        ('W2_Shard_gpu0', 'w2 shard'),
        ('W2_Shard_gpu1', None),
        ('B2', 'b2 - dtensor'),
        ('Y_before_sync_gpu0', 'Y before sync'),
        ('Y_before_sync_gpu1', None),
        ('Y_after_sync', 'Y after sync'),
        ('W1_Shard_grad', 'W1_Shard gradient'),
        ('W2_Shard_grad', 'W2_Shard gradient'),
    ]
    
    i = 0
    w1_shard_count = 0
    h_count = 0
    w2_shard_count = 0
    y_before_count = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for tensor headers
        if 'Tensor(shape=' in line:
            shape = parse_tensor_header(line)
            
            # Determine which tensor this is based on previous markers
            tensor_name = None
            
            # Look back to find the most recent section marker
            for j in range(i-1, max(0, i-10), -1):
                prev_line = lines[j].lower().strip()
                
                if 'x - dtensor' in prev_line:
                    tensor_name = 'X'
                    break
                elif 'w1 - dtensor' in prev_line:
                    tensor_name = 'W1'
                    break
                elif 'w1 shard' in prev_line:
                    tensor_name = f'W1_Shard_gpu{w1_shard_count}'
                    w1_shard_count += 1
                    break
                elif 'b1 - dtensor' in prev_line:
                    tensor_name = 'B1'
                    break
                elif 'h - dtensor' in prev_line:
                    tensor_name = f'H_gpu{h_count}'
                    h_count += 1
                    break
                elif 'w2 - dtensor' in prev_line:
                    tensor_name = 'W2'
                    break
                elif 'w2 shard' in prev_line:
                    tensor_name = f'W2_Shard_gpu{w2_shard_count}'
                    w2_shard_count += 1
                    break
                elif 'b2 - dtensor' in prev_line:
                    tensor_name = 'B2'
                    break
                elif 'y before sync' in prev_line:
                    tensor_name = f'Y_before_sync_gpu{y_before_count}'
                    y_before_count += 1
                    break
                elif 'y after sync' in prev_line:
                    tensor_name = 'Y_after_sync'
                    break
                elif 'w1_shard gradient' in prev_line:
                    tensor_name = 'W1_Shard_grad'
                    break
                elif 'w2_shard gradient' in prev_line:
                    tensor_name = 'W2_Shard_grad'
                    break
            
            if tensor_name:
                arr, next_i = parse_tensor_array(lines, i + 1)
                if arr is not None:
                    tensors[tensor_name] = arr
                    print(f"Parsed {tensor_name}: shape {arr.shape}")
                i = next_i
                continue
        
        i += 1
    
    return tensors


def verify_forward_pass(tensors):
    """Verify forward pass calculations."""
    results = {}
    
    print("\n" + "="*45)
    print("\tFORWARD PASS VERIFICATION")
    print("="*45)
    
    # Check if we have required tensors for GPU 0
    required_gpu0 = ['X', 'W1_Shard_gpu0', 'B1', 'H_gpu0']
    if not all(k in tensors for k in required_gpu0):
        print(f"Warning: Missing tensors for GPU 0 verification. Have: {list(tensors.keys())}")
        return results
    
    X = tensors['X']
    W1_gpu0 = tensors['W1_Shard_gpu0']
    B1 = tensors['B1']
    H_gpu0 = tensors['H_gpu0']
    
    # Compute H = X @ W1 + B1
    # For 3D batched matmul: H[b] = X[b] @ W1[b] for each batch
    H_computed = np.zeros_like(H_gpu0)
    for b in range(X.shape[0]):
        H_computed[b] = X[b] @ W1_gpu0[b] + B1[b]
    
    print("\n--- Layer 1: H = X @ W1_Shard + B1 (GPU 0) ---")
    print(f"X shape: {X.shape}")
    print(f"W1_Shard shape: {W1_gpu0.shape}")
    print(f"B1 shape: {B1.shape}")
    print(f"\nComputed H:")
    print(H_computed)
    print(f"\nExpected H (from C++):")
    print(H_gpu0)
    
    h_diff = np.max(np.abs(H_computed - H_gpu0))
    h_match = h_diff < 0.001
    results['H_gpu0'] = ('PASS' if h_match else 'FAIL', h_diff)
    print(f"\nMax difference: {h_diff:.6f} -> {'PASS' if h_match else 'FAIL'}")
    
    # Verify second linear layer if available
    if all(k in tensors for k in ['W2_Shard_gpu0', 'B2', 'Y_before_sync_gpu0']):
        W2_gpu0 = tensors['W2_Shard_gpu0']
        B2 = tensors['B2']
        Y_gpu0 = tensors['Y_before_sync_gpu0']
        
        Y_computed = np.zeros_like(Y_gpu0)
        for b in range(H_gpu0.shape[0]):
            Y_computed[b] = H_gpu0[b] @ W2_gpu0[b] + B2[b]
        
        print("\n--- Layer 2: Y = H @ W2_Shard + B2 (GPU 0, before sync) ---")
        print(f"H shape: {H_gpu0.shape}")
        print(f"W2_Shard shape: {W2_gpu0.shape}")
        print(f"\nComputed Y:")
        print(Y_computed)
        print(f"\nExpected Y (from C++):")
        print(Y_gpu0)
        
        y_diff = np.max(np.abs(Y_computed - Y_gpu0))
        y_match = y_diff < 0.001
        results['Y_gpu0'] = ('PASS' if y_match else 'FAIL', y_diff)
        print(f"\nMax difference: {y_diff:.6f} -> {'PASS' if y_match else 'FAIL'}")
    
    # Verify sync (all-reduce sum) if both GPU outputs available
    if all(k in tensors for k in ['Y_before_sync_gpu0', 'Y_before_sync_gpu1', 'Y_after_sync']):
        Y0 = tensors['Y_before_sync_gpu0']
        Y1 = tensors['Y_before_sync_gpu1']
        Y_synced = tensors['Y_after_sync']
        
        Y_sync_computed = Y0 + Y1
        
        print("\n--- Sync (All-Reduce Sum) ---")
        print(f"\nComputed Y_sync (GPU0 + GPU1):")
        print(Y_sync_computed)
        print(f"\nExpected Y_sync (from C++):")
        print(Y_synced)
        
        sync_diff = np.max(np.abs(Y_sync_computed - Y_synced))
        sync_match = sync_diff < 0.001
        results['sync'] = ('PASS' if sync_match else 'FAIL', sync_diff)
        print(f"\nMax difference: {sync_diff:.6f} -> {'PASS' if sync_match else 'FAIL'}")
    
    return results


def verify_backward_pass(tensors):
    """Verify backward pass gradient calculations."""
    results = {}
    
    print("\n" + "="*45)
    print("\tBACKWARD PASS VERIFICATION")
    print("="*45)
    
    # Need X, H, W2, and gradients
    required = ['X', 'H_gpu0', 'W2_Shard_gpu0', 'Y_after_sync']
    if not all(k in tensors for k in required):
        print(f"Warning: Missing tensors for backward verification. Have: {list(tensors.keys())}")
        return results
    
    X = tensors['X']
    H = tensors['H_gpu0']
    W2 = tensors['W2_Shard_gpu0']
    Y = tensors['Y_after_sync']
    
    # Loss = mean(Y), so dL/dY = 1/N where N = total elements
    N = Y.size
    dL_dY = np.ones_like(Y) / N
    
    print(f"\nLoss = mean(Y), dL/dY = 1/{N} = {1/N:.6f}")
    
    # For layer 2: Y = H @ W2 + B2
    # dL/dW2 = H.T @ dL/dY (for each batch)
    if 'W2_Shard_grad' in tensors:
        W2_grad_expected = tensors['W2_Shard_grad']
        
        dL_dW2 = np.zeros_like(W2)
        for b in range(H.shape[0]):
            dL_dW2[b] = H[b].T @ dL_dY[b]
        
        print("\n--- W2_Shard Gradient: dL/dW2 = H.T @ dL/dY ---")
        print(f"\nComputed gradient:")
        print(dL_dW2)
        print(f"\nExpected gradient (from C++):")
        print(W2_grad_expected)
        
        w2_diff = np.max(np.abs(dL_dW2 - W2_grad_expected))
        w2_match = w2_diff < 0.001
        results['W2_grad'] = ('PASS' if w2_match else 'FAIL', w2_diff)
        print(f"\nMax difference: {w2_diff:.6f} -> {'PASS' if w2_match else 'FAIL'}")
    
    # For layer 1: H = X @ W1 + B1
    # First compute dL/dH = dL/dY @ W2.T
    # Then dL/dW1 = X.T @ dL/dH
    if 'W1_Shard_grad' in tensors:
        W1_grad_expected = tensors['W1_Shard_grad']
        W1 = tensors.get('W1_Shard_gpu0')
        
        if W1 is not None:
            dL_dH = np.zeros_like(H)
            for b in range(H.shape[0]):
                dL_dH[b] = dL_dY[b] @ W2[b].T
            
            dL_dW1 = np.zeros_like(W1)
            for b in range(X.shape[0]):
                dL_dW1[b] = X[b].T @ dL_dH[b]
            
            print("\n--- W1_Shard Gradient: dL/dW1 = X.T @ (dL/dY @ W2.T) ---")
            print(f"\nComputed gradient:")
            print(dL_dW1)
            print(f"\nExpected gradient (from C++):")
            print(W1_grad_expected)
            
            w1_diff = np.max(np.abs(dL_dW1 - W1_grad_expected))
            w1_match = w1_diff < 0.001
            results['W1_grad'] = ('PASS' if w1_match else 'FAIL', w1_diff)
            print(f"\nMax difference: {w1_diff:.6f} -> {'PASS' if w1_match else 'FAIL'}")
    
    return results


def print_summary(forward_results, backward_results):
    """Print overall verification summary."""
    print("\n" + "="*41)
    print("\tVERIFICATION SUMMARY")
    print("="*41)
    
    all_pass = True
    
    for name, (status, diff) in forward_results.items():
        # symbol = "✓" if status == 'PASS' else "✗"
        print(f"Forward {name}: {status} (max diff: {diff:.6f})")
        if status != 'PASS':
            all_pass = False
    
    for name, (status, diff) in backward_results.items():
        # symbol = "✓" if status == 'PASS' else "✗"
        print(f"Backward {name}: {status} (max diff: {diff:.6f})")
        if status != 'PASS':
            all_pass = False
    
    if all_pass and (forward_results or backward_results):
        print("\n ALL CALCULATIONS ARE CORRECT ")
    elif forward_results or backward_results:
        print("\n  Some calculations have discrepancies. Check the details above.")
    else:
        print("\n  No tensors were parsed. Make sure the output format matches expectations.")


def main():
    # Read from stdin
    print("Reading tensor_parallel_mlp output from stdin...")
    print("(Pipe output: mpirun -np 2 ./tensor_parallel_mlp 2>&1 | python3 verify_calculations.py)")
    print()
    
    text = sys.stdin.read()
    
    if not text:
        print("Error: No input received. Please pipe tensor_parallel_mlp output.")
        sys.exit(1)
    
    print(f"Read {len(text)} characters of output.")
    print()
    
    # Parse tensors
    tensors = parse_output(text)
    
    if not tensors:
        print("Error: No tensors were parsed from the output.")
        print("\nExpected format example:")
        print("  x - dtensor :")
        print("  Tensor(shape=(2, 4, 2), dtype=Float32, device='cuda:0')")
        print("  [[0.0700, 0.1601], ...]")
        sys.exit(1)
    
    print(f"\nParsed {len(tensors)} tensors: {list(tensors.keys())}")
    
    # Verify calculations
    forward_results = verify_forward_pass(tensors)
    backward_results = verify_backward_pass(tensors)
    
    # Print summary
    print_summary(forward_results, backward_results)


if __name__ == "__main__":
    main()
