#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for tril(input, diagonal)
 * 
 * Forward: result = tril(input, diagonal)
 * Backward: grad_input = tril(grad_output, diagonal)
 */
class TrilBackward : public Node {
private:
    int64_t diagonal_;
    double value_;

public:
    TrilBackward(int64_t diagonal, double value = 0.0)
        : diagonal_(diagonal), value_(value) {}
    
    const char* name() const override { return "TrilBackward"; }
    
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    
    void release_saved_variables() override {}
};

} // namespace autograd
} // namespace OwnTensor
