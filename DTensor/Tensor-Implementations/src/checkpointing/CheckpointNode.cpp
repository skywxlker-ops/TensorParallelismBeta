#include "checkpointing/CheckpointNode.h"
#include "checkpointing/GradMode.h"
#include "autograd/Engine.h"
#include "autograd/ops_template.h"
// #include "autograd/operations/ReductionOps.h" // No longer needed
#include "ops/TensorOps.h"
#include <stdexcept>


namespace OwnTensor {
namespace autograd {


CheckpointNode::CheckpointNode(
   std::function<variable_list(const variable_list&)> forward_fn,
   const variable_list& inputs,
   RNGState rng_state,
   size_t num_outputs,
   bool offload_to_cpu)
   : Node(inputs.size()),
     forward_fn_(std::move(forward_fn)),
     rng_state_(std::move(rng_state)),
     num_outputs_(num_outputs),
     offload_to_cpu_(offload_to_cpu) {
  
   saved_inputs_.reserve(inputs.size());
   input_requires_grad_.reserve(inputs.size());
  
   for (size_t i = 0; i < inputs.size(); ++i) {
       const auto& input = inputs[i];
      
       // Save original device for restoration
       input_devices_.push_back(input.device());
      
       // Conditional offloading to CPU for storage
       if (offload_to_cpu_) {
           // Note: This adds PCIe overhead but saves VRAM
           Tensor cpu_input = input.to(Device::CPU).detach(); // Detach to save memory
           saved_inputs_.emplace_back(cpu_input, false);
       } else {
           // Detach to save memory (keep storage, drop graph history)
           Tensor detached_input = input.detach();
           saved_inputs_.emplace_back(detached_input, false);
       }
      
       input_requires_grad_.push_back(input.requires_grad());
      
       if (input.requires_grad()) {
           Tensor& input_mut = const_cast<Tensor&>(input);
           Edge e = get_grad_edge(input_mut);
           set_next_edge(i, e);
       } else {
           set_next_edge(i, Edge{});
       }
   }
}


variable_list CheckpointNode::apply(variable_list&& grads) {
   // std::cout << "[DEBUG] CheckpointNode::apply started. Grads size: " << grads.size() << "\n";
  
   // 1. Restore RNG state to ensure deterministic recomputation (e.g., Dropout).
   RNGStateGuard rng_guard;
   RNG::set_state(rng_state_);


   // 3. Unpack inputs and create views to isolate the recomputation graph.
   // This prevents the local backward pass from propagating into the global graph.
   variable_list recompute_inputs;
   recompute_inputs.reserve(saved_inputs_.size());
   for (size_t i = 0; i < saved_inputs_.size(); ++i) {
       const auto& sv = saved_inputs_[i];
       Tensor input = sv.unpack(shared_from_this());
       if (input.unsafeGetTensorImpl()) {
           // Restore to original device (likely GPU)
           Tensor gpu_input = input.to(input_devices_[i]);
          
           // Create a view that shares storage but has its own AutogradMeta.
           Tensor input_view = gpu_input.view(gpu_input.shape()).detach();
                     // Important: We set requires_grad on the view BEFORE enabling GradMode
            // so that it acts as a leaf in the local recomputation graph.
            input_view.set_requires_grad(input_requires_grad_[i]);
            
            recompute_inputs.push_back(input_view);
        } else {
           recompute_inputs.push_back(Tensor());
       }
   }
  
   // OPTIMIZATION: Early release of saved inputs (CPU tensors)
   // We have unpacked them to GPU, so we don't need the CPU copies anymore.
   // This reduces peak memory usage during recomputation.
   for (auto& sv : saved_inputs_) {
       sv.reset();
   }
   input_devices_.clear();


   // 4. Enable gradients for recomputation.
   // The initial forward pass was done in no_grad mode, so we must
   // re-enable it here to build the local computational graph.
   GradModeGuard grad_guard(true);


   // RAII guard to ensure release_saved_variables() is called even if recompute or backward fails.
   struct ReleaseGuard {
       CheckpointNode* node;
       ~ReleaseGuard() { if (node) node->release_saved_variables(); }
   } release_guard{this};


    // 5. Re-run forward pass to build the local graph.
    variable_list outputs = forward_fn_(recompute_inputs);

   if (outputs.size() != grads.size()) {
       throw std::runtime_error(
           "CheckpointNode::apply: Number of recomputed outputs (" +
           std::to_string(outputs.size()) + ") does not match number of gradients (" +
           std::to_string(grads.size()) + ")");
   }


  
   // 6. Disable dependency tracking for local graph
   // set_dependency_tracking_enabled(false);
  
   // Set dependencies to a high value for local nodes to prevent eager release
   // for (size_t i = 0; i < outputs.size(); ++i) {
   //    if (outputs[i].requires_grad() && outputs[i].grad_fn()) {
   //        outputs[i].grad_fn()->set_dependencies(999999);
   //    }
   // }


   // 7. Run local backward (multi-root)
   std::vector<Tensor> valid_outputs;
   std::vector<Tensor> valid_grads;
   valid_outputs.reserve(grads.size());
   valid_grads.reserve(grads.size());


   for (size_t i = 0; i < outputs.size(); ++i) {
       if (i >= grads.size()) break;
      
       // Only trigger backward if we have a valid gradient and the output requires grad
       if (outputs[i].requires_grad() && outputs[i].unsafeGetTensorImpl() && grads[i].unsafeGetTensorImpl()) {
            valid_outputs.push_back(outputs[i]);
            valid_grads.push_back(grads[i]);
       }
   }


   if (!valid_outputs.empty()) {
       autograd::backward(valid_outputs, valid_grads);
   }
  
   // Re-enable dependency tracking
   // set_dependency_tracking_enabled(true);


   // 8. Manually release local graph nodes after backward completes
   // Note: backward() releases variables for nodes it executes.
   // However, we might want to ensure the root nodes (outputs) are cleaned up if they weren't leaves?
   // The roots are leaves of the LOCAL graph (outputs of forward_fn).
   // fast_backward_sequential releases variables of nodes it visits.
   // We can keep this loop just in case, or rely on the engine.
   // Engine releases variables of the node *after* it executes.
   // The roots of the backward pass (outputs of forward) are the *starts* of the BFS.
   // If they have next_edges, they are processed.
   for (size_t i = 0; i < outputs.size(); ++i) {
       if (outputs[i].requires_grad() && outputs[i].grad_fn()) {
           outputs[i].grad_fn()->release_saved_variables();
       }
   }


   // 9. Collect gradients for inputs
   variable_list input_grads;
   input_grads.reserve(recompute_inputs.size());
   for (size_t i = 0; i < recompute_inputs.size(); ++i) {
       if (input_requires_grad_[i]) {
           if (recompute_inputs[i].has_grad()) {
               Tensor g = recompute_inputs[i].grad_view();
               input_grads.push_back(g);
           } else {
               input_grads.push_back(Tensor());
           }
       } else {
           input_grads.push_back(Tensor());
       }
   }


   // 10. Clear saved data to free memory.
   // release_saved_variables(); // Already called early
   release_guard.node = nullptr;


   return input_grads;
}


void CheckpointNode::release_saved_variables() {
   // Release the forward function.
   forward_fn_ = nullptr;
  
   // Release the saved inputs.
   for (auto& sv : saved_inputs_) {
       sv.reset();
   }
   input_devices_.clear();
}


} // namespace autograd
} // namespace OwnTensor