# CustomDNN Usage Examples

Flexible tensor-parallel neural network layers with per-parameter sharding control.

## ShardingType API

```cpp
#include "nn/CustomDNN.h"
using namespace OwnTensor::dnn;

// Two sharding types:
ShardingType::Shard(0)      // Row-wise sharding (split along dim 0)
ShardingType::Shard(1)      // Column-wise sharding (split along dim 1)
ShardingType::Replicated()  // Full copy on each GPU
```

## DLinear - Flexible Sharding

### New API (Recommended)

```cpp
// Column-parallel: shard weight columns, replicate bias
auto fc1 = DLinear(mesh, pg,
    768,                          // in_features
    3072,                         // out_features
    ShardingType::Shard(1),       // weight: column-wise (3072/num_gpus per GPU)
    ShardingType::Replicated(),   // bias: full copy on each GPU
    true);                        // has_bias

// Row-parallel: shard weight rows, with bias
auto fc2 = DLinear(mesh, pg,
    3072,                         // in_features
    768,                          // out_features
    ShardingType::Shard(0),       // weight: row-wise (3072/num_gpus per GPU)
    ShardingType::Replicated(),   // bias: replicated
    true);                        // has_bias

// Replicated layer (no sharding)
auto output_proj = DLinear(mesh, pg,
    768, vocab_size,
    ShardingType::Replicated(),   // weight: full copy
    ShardingType::Replicated(),   // bias: full copy
    true);
```

### Legacy API (Backward Compatible)

```cpp
// Still works! No bias, uses ParallelType enum
auto fc_old = DLinear(768, 3072, mesh, pg, ParallelType::COLUMN);
```

## Auto-Sync Behavior

The forward pass automatically handles synchronization based on weight sharding:

| Weight Sharding | Sync in forward() |
|-----------------|-------------------|
| `Shard(0)` (row-parallel) | **Yes** - AllReduce after matmul |
| `Shard(1)` (column-parallel) | No - output already distributed |
| `Replicated()` | No - no communication needed |

```cpp
// Row-parallel auto-syncs
DTensor y = fc2.forward(x);  // Internally: y = AllReduce(x @ W) + b
```

## Complete MLP Example

```cpp
#include "nn/CustomDNN.h"
using namespace OwnTensor::dnn;

class CustomMLP {
    std::unique_ptr<DLinear> fc1_;
    std::unique_ptr<DLinear> fc2_;
    
public:
    CustomMLP(std::shared_ptr<DeviceMesh> mesh,
              std::shared_ptr<ProcessGroupNCCL> pg,
              int64_t hidden_dim, int64_t ffn_dim)
    {
        // Column-parallel first layer (expand)
        fc1_ = std::make_unique<DLinear>(mesh, pg,
            hidden_dim, ffn_dim,
            ShardingType::Shard(1),       // shard columns
            ShardingType::Shard(0),       // shard bias too
            true);
        
        // Row-parallel second layer (reduce)
        fc2_ = std::make_unique<DLinear>(mesh, pg,
            ffn_dim, hidden_dim,
            ShardingType::Shard(0),       // shard rows
            ShardingType::Replicated(),   // replicate bias
            true);
    }
    
    DTensor forward(const DTensor& x) {
        DTensor h = fc1_->forward(x);
        h = h.gelu();
        return fc2_->forward(h);  // Auto AllReduce
    }
    
    std::vector<DTensor*> parameters() {
        std::vector<DTensor*> params;
        for (auto* p : fc1_->parameters()) params.push_back(p);
        for (auto* p : fc2_->parameters()) params.push_back(p);
        return params;
    }
};
```

## Training Loop

```cpp
// Setup
auto mesh = std::make_shared<DeviceMesh>(/*...*/);
auto pg = std::make_shared<ProcessGroupNCCL>(/*...*/);

// Model with new API
CustomMLP mlp(mesh, pg, 768, 3072);
SGD optimizer(0.001);

// Training
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : dataloader) {
        // Zero gradients
        for (auto* p : mlp.parameters()) p->zero_grad();
        
        // Forward
        DTensor pred = mlp.forward(batch.input);
        DTensor loss = pred.mse_loss(batch.target);
        
        // Backward
        loss.backward();
        
        // Update
        optimizer.step(mlp.parameters());
    }
}
```

## Comparison: Old vs New API

| Feature | Old API (ParallelType) | New API (ShardingType) |
|---------|------------------------|------------------------|
| Weight sharding | COLUMN or ROW | Shard(0), Shard(1), Replicated() |
| Bias support | No | Yes |
| Per-param control | No | Yes |
| Auto-sync | ROW only | Based on weight sharding |
| Flexibility | Low | High |
