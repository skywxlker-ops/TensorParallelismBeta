#include "autograd/backward/GradAccumulator.h"
#include "core/AutogradMeta.h"
#include "ops/TensorOps.h"

namespace OwnTensor {
namespace autograd {

GradAccumulator::GradAccumulator(TensorImpl* impl)
    : Node(1), leaf_impl_(impl) {}

std::vector<GradAccumulator*> GradAccumulator::pool_;
std::mutex GradAccumulator::pool_mutex_;

void GradAccumulator::reset(TensorImpl* impl) {
    leaf_impl_ = intrusive_ptr<TensorImpl>(impl);
    // Reset Node state if necessary (clearing edges, hooks etc.)
    // For now assuming Node state is clean or doesn't matter for new usage as leaf
    // Important: Node construction increments sequence_nr. Reuse means sequence_nr is stale?
    // Engine uses topological sort which re-computes dependencies. 
    // Sequence nr is mostly for debug or deterministic ties.
    // Ideally we should re-assign a new sequence number.
    // Accessing protected member in Node? 
    // We can just leave it. If Engine relies heavily on strict increasing seq number for correctness it might issue.
    // Engine uses topological sort based on structure, sequence_nr is secondary.
    clear_edges(); 
    // Reset edge to empty/invalid if any
}

std::shared_ptr<GradAccumulator> GradAccumulator::make(TensorImpl* impl) {
    // Temporarily disabled pooling for debugging
    GradAccumulator* ptr = new GradAccumulator(impl);
    
    // Return shared_ptr with default deleter (delete ptr)
    return std::shared_ptr<GradAccumulator>(ptr);
}

std::vector<Tensor> GradAccumulator::apply(std::vector<Tensor>&& grads) {
    // fprintf(stderr, "DEBUG: GradAccumulator::apply EMPTY\n");
    // return {};

    // Assuming grads contains a single grad_output for this leaf
    // And leaf_impl_ is the TensorImpl for which we are accumulating gradients
    Tensor grad_output = std::move(grads[0]); // Take ownership of the grad

    // Accumulate gradient into leaf tensor
    if (leaf_impl_->has_autograd_meta()) {
        auto* meta = static_cast<AutogradMeta*>(leaf_impl_->autograd_meta());
        
        // Use optimized accumulation (1 lock instead of 3)
        meta->accumulate_grad(std::move(grad_output));
        
        meta->trigger_post_acc_hooks(meta->grad());
    }

    // Leaf nodes don't propagate gradients further up the graph
    return {};
}

} // namespace autograd
} // namespace OwnTensor
