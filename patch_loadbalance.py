with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/gpt2_cp_test/LoadBalanceBackward.h', 'r') as f:
    h = f.read()

new_h = """#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"
#include "tensor/dtensor.h"

using namespace OwnTensor;

class LoadBalanceBackward : public autograd::Node {
public:
  LoadBalanceBackward(int world_size, int rank, int chunk_dim, const Shape& full_shape) 
      : Node(1), world_size_(world_size), rank_(rank), chunk_dim_(chunk_dim), full_shape_(full_shape) {}
  
  const char *name() const override { return "LoadBalanceBackward"; }
  
  std::vector<Tensor> apply(std::vector<Tensor> &&grads) override {
    Tensor g_local = grads[0];
    Tensor g_full = Tensor::zeros(full_shape_, g_local.opts());
    HeadTail hb;
    hb.set_world_size(world_size_);
    hb.set_chunk_dim(chunk_dim_);
    // We use partial_update to write the local chunk directly into the full gradient tensor
    hb.partial_update(g_local, g_full, rank_, false);
    return {g_full};
  }
  
private:
  int world_size_;
  int rank_;
  int chunk_dim_;
  Shape full_shape_;
};
"""
with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/gpt2_cp_test/LoadBalanceBackward.h', 'w') as f:
    f.write(new_h)
