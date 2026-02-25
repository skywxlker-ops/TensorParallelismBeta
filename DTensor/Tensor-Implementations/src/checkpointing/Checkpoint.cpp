
#include "checkpointing/CheckpointNode.h"
#include "checkpointing/GradMode.h"
#include "nn/NN.h"
#include <algorithm>
#include <iostream>

namespace OwnTensor {
namespace autograd {

variable_list checkpoint(
    std::function<variable_list(const variable_list&)> fn,
    const variable_list& inputs,
    bool offload_to_cpu) {
    
    // 1. Check if any input requires gradient.
    // If none do, we don't need to checkpoint at all.
    bool any_requires_grad = std::any_of(inputs.begin(), inputs.end(),
        [](const Tensor& t) { return t.requires_grad(); });
    
    if (!any_requires_grad || !GradMode::is_enabled()) {
        return fn(inputs);
    }

    // 2. Capture current RNG state.
    RNGState rng_state = RNG::get_state();

    // 3. Run the function in NO_GRAD mode.
    // This is the "memory saving" part: intermediate activations are NOT stored.
    variable_list outputs;
    {
        NoGradGuard guard;
        outputs = fn(inputs);
    }

    // 4. Create the CheckpointNode.
    // This node will handle the recomputation during the backward pass.
    auto checkpoint_node = std::make_shared<CheckpointNode>(
        std::move(fn),
        inputs,
        std::move(rng_state),
        outputs.size(),
        offload_to_cpu
    );

    // 5. Connect outputs to the CheckpointNode.
    // We set the grad_fn of the outputs to our CheckpointNode so the 
    // autograd engine knows to call it during backward.
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].unsafeGetTensorImpl()) {
            outputs[i].set_requires_grad(true);
            outputs[i].set_grad_fn(checkpoint_node);
            outputs[i].set_output_nr(i);
        }
    }

    return outputs;
}

variable_list checkpoint_sequential(
    std::shared_ptr<nn::Sequential> model,
    int segments,
    const variable_list& inputs,
    bool offload_to_cpu) {
    
    const auto& modules = model->modules();
    int num_modules = static_cast<int>(modules.size());
    
    if (segments <= 0) {
        throw std::invalid_argument("segments must be greater than 0");
    }
    
    if (segments > num_modules) {
        segments = num_modules;
    }

    int modules_per_segment = num_modules / segments;
    int remainder = num_modules % segments;

    variable_list current_inputs = inputs;
    int start_idx = 0;

    for (int i = 0; i < segments; ++i) {
        int segment_size = modules_per_segment + (i < remainder ? 1 : 0);
        int end_idx = start_idx + segment_size;

        auto segment_fn = [modules, start_idx, end_idx](const variable_list& segment_inputs) {
            Tensor x = segment_inputs[0];
            for (int j = start_idx; j < end_idx; ++j) {
                x = modules[j]->forward(x);
            }
            return variable_list{x};
        };

        if (i < segments - 1) {
            current_inputs = checkpoint(segment_fn, current_inputs, offload_to_cpu);
        } else {
            // Last segment doesn't necessarily need checkpointing if we want to save memory 
            // but PyTorch checkpoints all segments in checkpoint_sequential.
            current_inputs = checkpoint(segment_fn, current_inputs, offload_to_cpu);
        }

        start_idx = end_idx;
    }

    return current_inputs;
}

} // namespace autograd
} // namespace OwnTensor