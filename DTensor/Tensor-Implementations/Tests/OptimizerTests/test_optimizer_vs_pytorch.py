#!/usr/bin/env python3
"""
test_optimizer_vs_pytorch.py


PyTorch reference for Adam/AdamW optimizer output (CPU and GPU).
Run alongside C++ test and compare printed outputs visually.
"""
import torch


def print_tensor(name, t, max_elements=16):
   """Print tensor values with 7 decimal precision."""
   data = t.detach().cpu().flatten().numpy()
   n = min(max_elements, len(data))
  
   print(f"{name} [{len(data)} elements, showing first {n}]:")
   for i in range(n):
       print(f"{data[i]:.7f}", end="")
       if i < n - 1:
           print(", ", end="")
       if (i + 1) % 8 == 0:
           print()
   if n % 8 != 0:
       print()


def test_adam(device_str, num_steps=2):
   """Test Adam optimizer on specified device."""
   device = torch.device(device_str)
  
   print("\n" + "=" * 50)
   print(f"=== Adam Optimizer ({device_str.upper()}): PyTorch Reference ===")
   print("=" * 50 + "\n")
  
   # Hyperparameters (must match C++)
   lr = 0.001
   beta1 = 0.9
   beta2 = 0.999
   eps = 1e-8
   weight_decay = 0.01
  
   print("Hyperparameters:")
   print(f"  lr = {lr}")
   print(f"  beta1 = {beta1}")
   print(f"  beta2 = {beta2}")
   print(f"  eps = {eps}")
   print(f"  weight_decay = {weight_decay}")
   print(f"  num_steps = {num_steps}")
   print(f"  device = {device_str}")
   print()
  
   # Create weight tensor with same values as C++: [0.1, 0.2, ..., 1.6]
   W = torch.zeros(4, 4, dtype=torch.float32, requires_grad=True, device=device)
   with torch.no_grad():
       for i in range(16):
           W.view(-1)[i] = 0.1 * (i + 1)
  
   print("--- Initial Weights ---")
   print_tensor("W_initial", W)
  
   # Fixed gradient [1.0, 1.0, ..., 1.0]
   grad = torch.ones(4, 4, dtype=torch.float32, device=device)
   print("\n--- Gradient (fixed) ---")
   print_tensor("grad", grad)
  
   # Create Adam optimizer (weight_decay = L2 regularization, coupled)
   optimizer = torch.optim.Adam([W], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
  
   # Run multiple steps
   for step in range(1, num_steps + 1):
       optimizer.zero_grad()
       W.grad = grad.clone()
       optimizer.step()
      
       print(f"\n--- After Step {step} ---")
       print_tensor("W", W)


def test_adamw(device_str, num_steps=2):
   """Test AdamW optimizer on specified device."""
   device = torch.device(device_str)
  
   print("\n" + "=" * 50)
   print(f"=== AdamW Optimizer ({device_str.upper()}): PyTorch Reference ===")
   print("=" * 50 + "\n")
  
   # Hyperparameters (must match C++)
   lr = 0.001
   beta1 = 0.9
   beta2 = 0.999
   eps = 1e-8
   weight_decay = 0.01
  
   print("Hyperparameters:")
   print(f"  lr = {lr}")
   print(f"  beta1 = {beta1}")
   print(f"  beta2 = {beta2}")
   print(f"  eps = {eps}")
   print(f"  weight_decay = {weight_decay}")
   print(f"  num_steps = {num_steps}")
   print(f"  device = {device_str}")
   print()
  
   # Create weight tensor with same values as C++: [0.1, 0.2, ..., 1.6]
   W = torch.zeros(4, 4, dtype=torch.float32, requires_grad=True, device=device)
   with torch.no_grad():
       for i in range(16):
           W.view(-1)[i] = 0.1 * (i + 1)
  
   print("--- Initial Weights ---")
   print_tensor("W_initial", W)
  
   # Fixed gradient [1.0, 1.0, ..., 1.0]
   grad = torch.ones(4, 4, dtype=torch.float32, device=device)
   print("\n--- Gradient (fixed) ---")
   print_tensor("grad", grad)
  
   # Create AdamW optimizer (decoupled weight decay)
   optimizer = torch.optim.AdamW([W], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
  
   # Run multiple steps
   for step in range(1, num_steps + 1):
       optimizer.zero_grad()
       W.grad = grad.clone()
       optimizer.step()
      
       print(f"\n--- After Step {step} ---")
       print_tensor("W", W)


def main():
   num_steps = 2
  
   print("=" * 60)
   print("    PyTorch Optimizer Reference")
   print("=" * 60)
  
   # Check for CUDA availability
   cuda_available = torch.cuda.is_available()
   if cuda_available:
       print(f"\nCUDA is available: {torch.cuda.get_device_name(0)}")
   else:
       print("\nCUDA is NOT available, GPU tests will be skipped")
  
   # CPU Tests
   print("\n############### CPU TESTS ###############")
   test_adam("cpu", num_steps)
   test_adamw("cpu", num_steps)
  
   # GPU Tests
   if cuda_available:
       print("\n############### GPU TESTS ###############")
       test_adam("cuda", num_steps)
       test_adamw("cuda", num_steps)
  
   print("\n" + "=" * 60)
   print("    Done - Compare with OwnTensor output above")
   print("=" * 60)


if __name__ == "__main__":
   main()
