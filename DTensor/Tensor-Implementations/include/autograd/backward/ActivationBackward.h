#pragma once

#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {
namespace autograd {

/**
 * @brief Backward function for ReLU: max(0, x)
 * 
 * Forward: out = max(0, x)
 * Backward: grad_x = grad_out * (x > 0)
 */
class ReluBackward : public Node {
private:
    Tensor saved_input_;
    
public:
    explicit ReluBackward(const Tensor& input);
    
    const char* name() const override { return "ReluBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override { saved_input_ = Tensor(); }
};

/**
 * @brief Backward function for GeLU activation
 * 
 * Forward: out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Backward: Uses chain rule with tanh derivative
 */
class GeLUBackward : public Node {
private:
    Tensor saved_input_;
    
public:
    explicit GeLUBackward(const Tensor& input);
    
    const char* name() const override { return "GeLUBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override { saved_input_ = Tensor(); }
};

/**
 * @brief Backward function for sigmoid: 1 / (1 + exp(-x))
 * 
 * Forward: out = sigmoid(x)
 * Backward: grad_x = grad_out * sigmoid(x) * (1 - sigmoid(x))
 */
class SigmoidBackward : public Node {
private:
    Tensor saved_output_;  // Save output for efficient backward
    
public:
    explicit SigmoidBackward(const Tensor& output);
    
    const char* name() const override { return "SigmoidBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override { saved_output_ = Tensor(); }
};

/**
 * @brief Backward function for softmax
 * 
 * Forward: out = exp(x) / sum(exp(x))
 * Backward: Jacobian-based computation
 */
class SoftmaxBackward : public Node {
private:
    Tensor saved_output_;
    int64_t dim_;
    
public:
    SoftmaxBackward(const Tensor& output, int64_t dim);
    
    const char* name() const override { return "SoftmaxBackward"; }
    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override;
    void release_saved_variables() override { saved_output_ = Tensor(); }
};

} // namespace autograd
} // namespace OwnTensor
