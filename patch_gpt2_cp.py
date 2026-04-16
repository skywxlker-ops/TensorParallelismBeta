with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/gpt2_cp_test/gpt2_cp_test.cpp', 'r') as f:
    cpp = f.read()

import re

# Find the block:
#     if (!config.cp_unshard && !is_in_generation_mode_) {
#       ...
#       x = autograd::contiguous(x_chunks[rank_]); // [B, T/n, C] — autograd-aware
#     }

old_block = """    if (!config.cp_unshard && !is_in_generation_mode_) {
      if (config.load_balancing) {
        Tensor x_lb = x.clone();
        HeadTail hb;
        hb.set_world_size(world_size_);
        hb.set_chunk_dim(1);
        hb.loadbalance(x_lb);
        if (x.requires_grad()) {
            auto grad_fn = std::make_shared<LoadBalanceBackward>(world_size_, 1);
            grad_fn->set_next_edge(0, autograd::get_grad_edge(x));
            x_lb.set_grad_fn(grad_fn);
            x_lb.set_requires_grad(true);
        }
        x = x_lb;
      }
      std::vector<Tensor> x_chunks =
          x.make_shards_inplace_axis(static_cast<size_t>(world_size_), 1);
      x = autograd::contiguous(x_chunks[rank_]); // [B, T/n, C] — autograd-aware
    }"""

new_block = """    if (!config.cp_unshard && !is_in_generation_mode_) {
      if (config.load_balancing) {
        Tensor x_lb = Tensor::empty(x.shape(), x.opts());
        // Do not use clone() to avoid autograd tracking this step.
        // We will perform raw data copy.
        cudaMemcpyAsync(x_lb.data_ptr(), x.data_ptr(), x.numel() * (x.dtype() == Dtype::Float32 ? 4 : 2), cudaMemcpyDeviceToDevice, 0);
        
        HeadTail hb;
        hb.set_world_size(world_size_);
        hb.set_chunk_dim(1);
        hb.loadbalance(x_lb);
        
        std::vector<Tensor> x_chunks =
            x_lb.make_shards_inplace_axis(static_cast<size_t>(world_size_), 1);
        
        // Disable autograd tracking for the chunking so we can intercept it
        Tensor x_local = x_chunks[rank_].clone(); // clone detaches the naive slice
        x_local = autograd::contiguous(x_local);
        
        if (x.requires_grad()) {
            auto grad_fn = std::make_shared<LoadBalanceBackward>(world_size_, rank_, 1, x.shape());
            grad_fn->set_next_edge(0, autograd::get_grad_edge(x));
            x_local.set_grad_fn(grad_fn);
            x_local.set_requires_grad(true);
        }
        x = x_local;
      } else {
        std::vector<Tensor> x_chunks =
            x.make_shards_inplace_axis(static_cast<size_t>(world_size_), 1);
        x = autograd::contiguous(x_chunks[rank_]); // [B, T/n, C]
      }
    }"""

if old_block in cpp:
    cpp = cpp.replace(old_block, new_block)
    with open('/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/gpt2_cp_test/gpt2_cp_test.cpp', 'w') as f:
        f.write(cpp)
    print("Replaced block perfectly.")
else:
    print("Could not find block.")
