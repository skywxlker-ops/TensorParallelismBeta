
#!/usr/bin/env python3
"""
test_sgd_vs_pytorch.py


PyTorch reference for SGD optimizer variations.
Tests: Vanilla SGD, SGD+Momentum, SGD+WeightDecay, SGD+Momentum+WeightDecay
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


def test_sgd(name, lr, momentum, weight_decay):
   """Test SGD optimizer with specified parameters."""
   print("\n" + "=" * 50)
   print(f"=== SGD ({name}): PyTorch Reference ===")
   print("=" * 50 + "\n")
  
   print("Hyperparameters:")
   print(f"  lr = {lr}")
   print(f"  momentum = {momentum}")
   print(f"  weight_decay = {weight_decay}")
   print()
  
   # Create weight tensor: [0.1, 0.2, ..., 1.6]
   W = torch.zeros(4, 4, dtype=torch.float32, requires_grad=True)
   with torch.no_grad():
       for i in range(16):
           W.view(-1)[i] = 0.1 * (i + 1)
  
   print("--- Initial Weights ---")
   print_tensor("W_initial", W)
  
   # Fixed gradient [1.0, 1.0, ..., 1.0]
   grad = torch.ones(4, 4, dtype=torch.float32)
   print("\n--- Gradient (fixed) ---")
   print_tensor("grad", grad)
  
   # Create SGD optimizer
   optimizer = torch.optim.SGD([W], lr=lr, momentum=momentum, weight_decay=weight_decay)
  
   # Set gradient and step
   optimizer.zero_grad()
   W.grad = grad.clone()
   optimizer.step()
  
   print("\n--- After Step 1 ---")
   print_tensor("W", W)


def main():
   print("=" * 60)
   print("    PyTorch SGD Optimizer Reference")
   print("=" * 60)
  
   lr = 0.1
  
   # Test all SGD variations
   test_sgd("Vanilla", lr, momentum=0.0, weight_decay=0.0)
   test_sgd("Momentum", lr, momentum=0.9, weight_decay=0.0)
   test_sgd("WeightDecay", lr, momentum=0.0, weight_decay=0.01)
   test_sgd("Momentum+WeightDecay", lr, momentum=0.9, weight_decay=0.01)
  
   print("\n" + "=" * 60)
   print("    Done - Compare with OwnTensor output")
   print("=" * 60)


if __name__ == "__main__":
   main()
