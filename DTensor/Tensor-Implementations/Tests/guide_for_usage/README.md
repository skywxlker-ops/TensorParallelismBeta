# Tensor Library Usage Guide

This folder contains comprehensive examples showing how to use the tensor library at its full potential.

## Files Overview

| File | Description |
|------|-------------|
| `01_tensor_basics.cpp` | Creating tensors, basic math, matmul, reductions |
| `02_autograd_basics.cpp` | Gradient tracking, forward/backward passes |
| `03_hooks_and_debugging.cpp` | Tensor/node hooks, AnomalyMode, SavedVariable |
| `04_custom_backward_functions.cpp` | Writing custom autograd ops (Square, Sigmoid, LeakyReLU) |
| `05_mlp_examples.cpp` | MLP architectures, Module system, training loops |
| `06_nn_module_extension.cpp` | Advanced layers: Dropout, BatchNorm, Attention, Residual |

## How to Run

```bash
cd /path/to/cgadimpl/tensor

# Run any guide:
make run-snippet FILE=Tests/guide_for_usage/01_tensor_basics.cpp
make run-snippet FILE=Tests/guide_for_usage/02_autograd_basics.cpp
make run-snippet FILE=Tests/guide_for_usage/03_hooks_and_debugging.cpp
make run-snippet FILE=Tests/guide_for_usage/04_custom_backward_functions.cpp
make run-snippet FILE=Tests/guide_for_usage/05_mlp_examples.cpp
make run-snippet FILE=Tests/guide_for_usage/06_nn_module_extension.cpp
```

## Quick Start

### 1. Basic Tensor Operations
```cpp
#include "core/Tensor.h"
using namespace OwnTensor;

Tensor x = Tensor::randn(Shape{{3, 3}}, TensorOptions());
Tensor y = Tensor::ones(Shape{{3, 3}}, TensorOptions());
Tensor z = x + y;
Tensor m = matmul(x, y);
```

### 2. Gradient Tracking
```cpp
TensorOptions opts = TensorOptions().with_req_grad(true);
Tensor w = Tensor::randn(Shape{{3, 3}}, opts);

Tensor y = autograd::matmul(x, w);
Tensor loss = autograd::mean(y);

autograd::backward(loss);
float* grad = w.grad<float>();  // Access gradient
```

### 3. Custom Backward Function
```cpp
class MyBackward : public Node {
public:
    MyBackward() : Node(1) {}
    std::string name() const override { return "MyBackward"; }
    
    variable_list apply(variable_list&& grads) override {
        // Compute gradient
        return {my_gradient};
    }
};
```

### 4. Building an MLP
```cpp
Sequential mlp;
mlp.add(std::make_shared<Linear>(784, 128));
mlp.add(std::make_shared<ReLU>());
mlp.add(std::make_shared<Linear>(128, 10));

Tensor output = mlp.forward(input);
```
