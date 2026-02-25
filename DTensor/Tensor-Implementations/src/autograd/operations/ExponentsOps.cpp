#include "autograd/operations/ExponentsOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/ExponentsBackward.h"
#include "ops/UnaryOps/Exponents.h"

namespace OwnTensor {
namespace autograd {

Tensor exp(const Tensor& input) {
    // ExpBackward takes Output (y).
    // ops_template makes result first.
    // If I pass output, I need result to be available.
    // Standard make_unary_op passes 'args' to constructor.
    // It calls `make_shared<BackwardNode>(args...)`.
    // And `Tensor result = forward_op(x)`.
    // It does NOT pass result to args automatically.
    // So if ExpBackward needs output, I can't use standard make_unary_op pattern easily unless I change ExpBackward to Input (recompute y) OR use a manual impl.
    // Calculating exp(x) again in backward is cheap (or same cost as storing).
    // But wait, if I use saved_output, I avoid recomputation.
    // But given the helper constraint, I'll switch ExpBackward to take INPUT?
    // Or I construct manually. 
    // Manual construction is better to save recomputation for exp.
    
    /* Manual Implementation for Exp */
    Tensor result = OwnTensor::exp(input);
    if (input.requires_grad()) {
        auto grad_fn = std::make_shared<ExpBackward>(result); // Pass result!
        Tensor& x_mut = const_cast<Tensor&>(input);
        grad_fn->set_next_edge(0, get_grad_edge(x_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

Tensor log(const Tensor& input) {
    return make_unary_op<LogBackward>(input,
        [](const Tensor& x) { return OwnTensor::log(x); },
        input);
}

Tensor exp2(const Tensor& input) {
    // Similar to exp, uses output for efficiency.
    Tensor result = OwnTensor::exp2(input);
    if (input.requires_grad()) {
        auto grad_fn = std::make_shared<Exp2Backward>(result);
        Tensor& x_mut = const_cast<Tensor&>(input);
        grad_fn->set_next_edge(0, get_grad_edge(x_mut));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

Tensor log2(const Tensor& input) {
    return make_unary_op<Log2Backward>(input,
        [](const Tensor& x) { return OwnTensor::log2(x); },
        input);
}

Tensor log10(const Tensor& input) {
    return make_unary_op<Log10Backward>(input,
        [](const Tensor& x) { return OwnTensor::log10(x); },
        input);
}

} // namespace autograd
} // namespace OwnTensor
