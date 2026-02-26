import torch
import torch.nn.functional as F

torch.manual_seed(42)
x = torch.randn(2, 5, requires_grad=True)
y = torch.randint(0, 5, (2,))

# Normal cross entropy
loss = F.cross_entropy(x, y)
loss.backward()
print("PyTorch F.cross_entropy grad:")
print(x.grad)

# Our manually implemented scaled version:
x2 = x.detach().clone()
x2.requires_grad = True
y2 = y.clone()

# manual computation of loss
log_probs = F.log_softmax(x2, dim=-1)
loss2 = -log_probs[torch.arange(2), y2].mean()
loss2.backward()
print("\nManual log softmax grad:")
print(x2.grad)

# The scale in F.cross_entropy is 1/N where N is the batch size!
# Vocab size does not scale the gradient!

