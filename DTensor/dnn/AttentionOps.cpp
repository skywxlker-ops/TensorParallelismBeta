#include "autograd/operations/AttentionOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/ops_template.h"
#include "checkpointing/GradMode.h"
#include "utils/Profiler.h"

#ifdef WITH_CUDA
#include "ops/helpers/AttentionKernels.h"
#include "autograd/backward/AttentionBackward.h"
#endif

#include <cmath>
#include <limits>
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

// ============================================================================
// Math Backend: composes existing autograd ops
// ============================================================================

static Tensor sdpa_math(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal)
{
    // std::cout<<"hi"<<std::endl;
    // query, key, value: (B, nh, T, hd)
    if(query.device() != key.device() && query.device()!= value.device())
    {
    
        throw std::runtime_error(
            "Device Mismatch, Q, K, V do not reside in the same device");
    
    }
    int64_t hd = query.shape().dims[3];
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    // std::cout<<scale<<std::endl;
    // Scale query
    Tensor scale_t = Tensor::full(Shape{{1}},
        TensorOptions().with_dtype(query.dtype()).with_device(query.device()),
        scale);
    Tensor q_scaled = autograd::mul(query, scale_t);

    // Q @ K^T -> (B, nh, T, T)
    Tensor k_t = autograd::transpose(key, -2, -1);
    Tensor attn_weights = autograd::matmul(q_scaled, k_t);

    // Apply causal mask + softmax
    Tensor attn_probs;
    if (is_causal) {
        float neg_inf = -std::numeric_limits<float>::infinity();
        attn_probs = autograd::fused_tril_softmax(attn_weights, 0, neg_inf);
    } else {
        attn_probs = autograd::softmax(attn_weights, -1);
    }

    // Attn @ V -> (B, nh, T, hd)
    return autograd::matmul(attn_probs, value);
}

// ============================================================================
// Memory-Efficient Backend (requires CUDA)
// ============================================================================

#ifdef WITH_CUDA
static Tensor sdpa_memory_efficient(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal)
{
    if(query.device() != key.device() && query.device()!= value.device())
    {
        throw std::runtime_error("Device Mismatch, Q, K, V do not reside in the same device");
    }
    if (!query.device().is_cuda() || query.dtype() != Dtype::Float32) {
        throw std::runtime_error(
            "Memory-efficient attention requires CUDA float32 tensors");
    }

    int64_t B  = query.shape().dims[0];
    int64_t nh = query.shape().dims[1];
    int64_t T  = query.shape().dims[2];
    int64_t hd = query.shape().dims[3];

    //* print the dimensions
    // printf("The dimensions of Q, K, V, O are: {%d, %d, %d, %d}\n", B, nh, T, hd);
    // exit(1);
    //* Kernel requires contiguous (B*nh, T, hd) layout — make contiguous if needed
    // Tensor q_contig = query.is_contiguous() ? query : query.contiguous();
    // Tensor k_contig = key.is_contiguous()   ? key   : key.contiguous();
    // Tensor v_contig = value.is_contiguous() ? value : value.contiguous();

    auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(query.device());

    // Allocate output and LSE — NO T×T allocation anywhere
    Tensor output = Tensor::empty(Shape{{B, nh, T, hd}}, opts);
    Tensor lse    = Tensor::empty(Shape{{B, nh, T}}, opts);

    // Single fused kernel: Q @ K^T → scale → mask → online softmax → @ V
    // Never materializes the T×T attention matrix.
    cuda::mem_efficient_attn_forward(
        query.data<float>(), key.data<float>(), value.data<float>(),
        output.data<float>(), lse.data<float>(),
        B, nh, T, hd, is_causal);

    // Build autograd graph with memory-efficient backward
    if (GradMode::is_enabled() &&
        (query.requires_grad() || key.requires_grad() || value.requires_grad()))
    {
        // std::cout<<"inside grad mode"<<std::endl;
        auto grad_fn = std::make_shared<MemEfficientAttentionBackward>(
            query.detach(), key.detach(), value.detach(),
            output.detach(), lse.detach(),
            B, nh, T, hd, is_causal);

        Tensor& q_mut = const_cast<Tensor&>(query);
        Tensor& k_mut = const_cast<Tensor&>(key);
        Tensor& v_mut = const_cast<Tensor&>(value);

        if (query.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(q_mut));
        }
        if (key.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(k_mut));
        }
        if (value.requires_grad()) {
            grad_fn->set_next_edge(2, get_grad_edge(v_mut));
        }

        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }

    return output;
}
#endif

// ============================================================================
// Public Dispatch
// ============================================================================

Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal,
    SDPBackend backend)
{
    GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention");

    switch (backend) {
        case SDPBackend::Math:
            // std::cout << "Using Math Backend" << std::endl;
            return sdpa_math(query, key, value, is_causal);

        case SDPBackend::MemoryEfficient:
#ifdef WITH_CUDA
            // std::cout << "Using Memory-Efficient Attention Backend" << std::endl;
            return sdpa_memory_efficient(query, key, value, is_causal);
#else
            throw std::runtime_error(
                "Memory-efficient attention requires CUDA. "
                "Falling back to Math backend.");
            //std::cout << "Using Math Backend   _____1 " << std::endl;
            return sdpa_math(query, key, value, is_causal);
#endif

        default:
            throw std::runtime_error("Unknown SDPBackend");
    }
}

} // namespace autograd
} // namespace OwnTensor