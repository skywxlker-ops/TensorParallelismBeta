import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# 1. SETUP: Define tensors from the manual example
# --------------------------------------------------------------------------
# Set requires_grad=True to track gradients
X = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8]], dtype=torch.float32, requires_grad=True)

A = torch.tensor([[1, 2, 3, 4, 5, 6],
                  [7, 8, 9, 1, 2, 3],
                  [4, 5, 6, 7, 8, 9],
                  [1, 2, 3, 4, 5, 6]], dtype=torch.float32, requires_grad=True)

B = torch.tensor([[6, 5, 4, 3],
                  [2, 1, 7, 8],
                  [9, 8, 6, 5],
                  [4, 3, 2, 1],
                  [1, 2, 3, 4],
                  [5, 6, 7, 8]], dtype=torch.float32, requires_grad=True)

# NOTE: For verification, we use an identity function to match the manual
# calculation's approximation of GeLU(x) ≈ x. In a real model, you would
# use F.gelu(x).
gelu = lambda x: x

# Define the fixed dropout mask from the manual example
dropout_mask = torch.tensor([[1, 0, 0, 1],
                             [0, 1, 1, 0]], dtype=torch.float32)
dropout_rate = 0.5
dropout_scale = 1 / (1 - dropout_rate)

print("--- Initial Tensors ---")
print(f"X:\n{X}\n")
print(f"A:\n{A}\n")
print(f"B:\n{B}\n")


# --------------------------------------------------------------------------
# 2. FORWARD PASS
# --------------------------------------------------------------------------
print("--- FORWARD PASS ---")

# --- Simulate Tensor Parallelism on 2 GPUs ---

# Split A by columns (Column Parallelism)
# "GPU 1" gets A1, "GPU 2" gets A2
A1 = A[:, :3]
A2 = A[:, 3:]

# Split B by rows (Row Parallelism)
# "GPU 1" gets B1, "GPU 2" gets B2
B1 = B[:3, :]
B2 = B[3:, :]

# --- Layer 1: Y = GeLU(X * A) ---
# Operation is local to each GPU, no communication needed.

# "GPU 1" computes its part
Y_intermediate_1 = X @ A1
Y1 = gelu(Y_intermediate_1)

# "GPU 2" computes its part
Y_intermediate_2 = X @ A2
Y2 = gelu(Y_intermediate_2)

print(f"Y1 (on GPU 1):\n{Y1}\n")
print(f"Y2 (on GPU 2):\n{Y2}\n")

# --- Layer 2: Z = Dropout(Y * B) ---

# Local matrix multiplication on each GPU
# Note: Y is implicitly [Y1, Y2]
Z_partial_1 = Y1 @ B1
Z_partial_2 = Y2 @ B2

print(f"Z_partial_1 (on GPU 1):\n{Z_partial_1}\n")
print(f"Z_partial_2 (on GPU 2):\n{Z_partial_2}\n")

# *** All-Reduce Communication Step ***
# The partial results are summed across GPUs.
Z_unactivated = Z_partial_1 + Z_partial_2
print(f"Z_unactivated (after All-Reduce):\n{Z_unactivated}\n")

# Apply Dropout
Z_masked = Z_unactivated * dropout_mask
Z = Z_masked * dropout_scale

print(f"Final Output Z:\n{Z}\n")


# --------------------------------------------------------------------------
# 3. BACKWARD PASS
# --------------------------------------------------------------------------
print("--- BACKWARD PASS ---")

# Define a dummy loss. Z.sum() creates an initial gradient dZ of all ones.
loss = Z.sum()
print(f"Loss (for backprop):\n{loss}\n")

# Automatically compute gradients for all tensors with requires_grad=True
loss.backward()


# --------------------------------------------------------------------------
# 4. VERIFICATION: Check gradients against manual calculations
# --------------------------------------------------------------------------
print("--- VERIFYING GRADIENTS ---\n")

# Gradient for X (dX)
# This required an All-Reduce in the backward pass
print(f"Gradient for X (dX):\n{X.grad}\n")

# Gradients for A (dA1 and dA2)
# A.grad contains the gradients for the entire A matrix. We can verify
# by slicing it into the parts corresponding to A1 and A2.
print(f"Gradient for A1 (dA1):\n{A.grad[:, :3]}\n")
print(f"Gradient for A2 (dA2):\n{A.grad[:, 3:]}\n")

# Gradients for B (dB1 and dB2)
# Similarly, B.grad contains the full gradient.
print(f"Gradient for B1 (dB1):\n{B.grad[:3, :]}\n")
print(f"Gradient for B2 (dB2):\n{B.grad[3:, :]}\n")




