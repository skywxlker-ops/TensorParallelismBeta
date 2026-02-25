#include "autograd/backward/TransposeBackward.h"

namespace OwnTensor {
namespace autograd {

TransposeBackward::TransposeBackward(int dim0, int dim1)
    : dim0_(dim0), dim1_(dim1) {}

std::vector<Tensor> TransposeBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty() || !grads[0].is_valid()) {
        return {Tensor()};
    }
    
    // The backward of transpose(dim0, dim1) is transpose(dim0, dim1) 
    // because transpose is its own inverse for the same pair of dimensions.
    return {grads[0].transpose(dim0_, dim1_)};
}

} // namespace autograd
} // namespace OwnTensor
