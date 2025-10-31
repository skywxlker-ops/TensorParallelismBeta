import torch
import torch.nn as nn
import torch.optim as optim

# Helper function to check GPU memory usage
def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Step 1: Create a model and optimizer, and allocate some tensors on the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.Linear(1000, 1000).to(device)  # Model on GPU
optimizer = optim.Adam(model.parameters())  # Optimizer for the model
tensor = torch.randn(1000, 1000, device=device)  # Tensor on GPU

print("Before deleting references:")
print_gpu_memory()

# Step 2: Delete the references to these objects
del model
del optimizer
del tensor

print("\nAfter deleting references:")
print_gpu_memory()

# Step 3: Free up GPU memory manually using torch.cuda.empty_cache()
torch.cuda.empty_cache()

print("\nAfter emptying cache:")
print_gpu_memory()


# import torch

# # Clear GPU cache
# torch.cuda.empty_cache()

# import torch
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
# else:
#     print("No GPU available. Training will run on CPU.")


