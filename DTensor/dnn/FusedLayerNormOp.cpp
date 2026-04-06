#include "dnn/FusedLayerNormOp.h"
#include "dnn/VectorizedLayerNormKernel.h"
#include "autograd/Node.h"
#include "autograd/ops_template.h"
#include "core/Tensor.h"
#include "device/DeviceCore.h"

namespace OwnTensor {
namespace dnn {

// ---------------------------------------------------------------------------
// Backward node — inherits OwnTensor::Node (not OwnTensor::autograd::Node)
// ---------------------------------------------------------------------------
class FusedLayerNormBackward : public OwnTensor::Node {
public:
    FusedLayerNormBackward(const Tensor& x, const Tensor& mean, const Tensor& rstd,
                           const Tensor& gamma, int cols)
        : saved_x_(x), saved_mean_(mean), saved_rstd_(rstd),
          saved_gamma_(gamma), cols_(cols) {}

    const char* name() const override { return "FusedLayerNormBackward"; }

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        Tensor dy = grads[0];

        int64_t rows = saved_x_.numel() / cols_;
        Tensor dx     = Tensor::zeros(saved_x_.shape(),     saved_x_.opts());
        Tensor dgamma = Tensor::zeros(saved_gamma_.shape(), saved_gamma_.opts());
        Tensor dbeta  = Tensor::zeros(saved_gamma_.shape(), saved_gamma_.opts());

        device::set_cuda_device(saved_x_.device().index);
        OwnTensor::cuda::vln_backward_f32(
            dy.data<float>(),
            saved_x_.data<float>(),
            saved_mean_.data<float>(),
            saved_rstd_.data<float>(),
            saved_gamma_.data<float>(),
            dx.data<float>(),
            dgamma.data<float>(),
            dbeta.data<float>(),
            static_cast<int>(rows),
            cols_);

        return {dx, dgamma, dbeta};
    }

    void release_saved_variables() override {
        saved_x_     = Tensor();
        saved_mean_  = Tensor();
        saved_rstd_  = Tensor();
        saved_gamma_ = Tensor();
    }

private:
    Tensor saved_x_;
    Tensor saved_mean_;
    Tensor saved_rstd_;
    Tensor saved_gamma_;
    int    cols_;
};

// ---------------------------------------------------------------------------
// Forward op
// ---------------------------------------------------------------------------
Tensor fused_layer_norm(
    const Tensor& x,
    const Tensor& gamma,
    const Tensor& beta,
    int cols,
    float eps)
{
    int64_t rows = x.numel() / cols;

    Tensor y    = Tensor(x.shape(), x.opts());
    Tensor mean = Tensor(Shape{{rows}}, x.opts().with_req_grad(false));
    Tensor rstd = Tensor(Shape{{rows}}, x.opts().with_req_grad(false));

    device::set_cuda_device(x.device().index);
    OwnTensor::cuda::vln_forward_f32(
        x.data<float>(),
        gamma.data<float>(),
        beta.data<float>(),
        y.data<float>(),
        mean.data<float>(),
        rstd.data<float>(),
        static_cast<int>(rows),
        cols,
        eps);

    if (x.requires_grad() || gamma.requires_grad() || beta.requires_grad()) {
        auto grad_fn = std::make_shared<FusedLayerNormBackward>(x, mean, rstd, gamma, cols);

        if (x.requires_grad()) {
            Tensor& xm = const_cast<Tensor&>(x);
            grad_fn->set_next_edge(0, OwnTensor::autograd::get_grad_edge(xm));
        }
        if (gamma.requires_grad()) {
            Tensor& gm = const_cast<Tensor&>(gamma);
            grad_fn->set_next_edge(1, OwnTensor::autograd::get_grad_edge(gm));
        }
        if (beta.requires_grad()) {
            Tensor& bm = const_cast<Tensor&>(beta);
            grad_fn->set_next_edge(2, OwnTensor::autograd::get_grad_edge(bm));
        }

        y.set_grad_fn(grad_fn);
        y.set_requires_grad(true);
    }

    return y;
}

} // namespace dnn
} // namespace OwnTensor
