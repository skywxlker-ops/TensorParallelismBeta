#include "autograd/operations/ReshapeOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/ReshapeBackward.h"
#include "autograd/backward/TransposeBackward.h"

namespace OwnTensor {
namespace autograd {

Tensor transpose(const Tensor& input, int dim0, int dim1) {
    return make_unary_op<TransposeBackward>(input,
        [dim0, dim1](const Tensor& x) { return x.transpose(dim0, dim1); },
        dim0, dim1);
}

Tensor reshape(const Tensor& input, Shape new_shape) {
    return make_unary_op<ReshapeBackward>(input,
        [&new_shape](const Tensor& x) { return x.reshape(new_shape); },
        input.shape());
}

Tensor view(const Tensor& input, Shape new_shape) {
    return make_unary_op<ReshapeBackward>(input,
        [&new_shape](const Tensor& x) { return x.view(new_shape); },
        input.shape());
}

std::vector<Tensor> ReshapeBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty() || !grads[0].is_valid()) {
        return {Tensor()};
    }
    // Backward of reshape is reshape back to original shape
    return {grads[0].reshape(input_shape_)};
}

} // namespace autograd
} // namespace OwnTensor
