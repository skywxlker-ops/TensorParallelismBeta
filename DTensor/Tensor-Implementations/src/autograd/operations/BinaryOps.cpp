#include "autograd/operations/BinaryOps.h"
#include "autograd/ops_template.h"
#include "autograd/backward/BinaryBackward.h"
#include "ops/TensorOps.h"

namespace OwnTensor {
namespace autograd {

Tensor add(const Tensor& a, const Tensor& b) {
    return make_binary_op<AddBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return operator+(x, y); },
        a.shape(), b.shape());
}

Tensor mul(const Tensor& a, const Tensor& b) {
    return make_binary_op<MulBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return operator*(x, y); },
        a, b);  // Pass a, b to MulBackward constructor
}

Tensor sub(const Tensor& a, const Tensor& b) {
    return make_binary_op<SubBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return operator-(x, y); },
        a.shape(), b.shape());
}

Tensor div(const Tensor& a, const Tensor& b) {
    return make_binary_op<DivBackward>(a, b,
        [](const Tensor& x, const Tensor& y) { return operator/(x, y); },
        a, b);  // Pass a, b to DivBackward constructor
}

} // namespace autograd
} // namespace OwnTensor
