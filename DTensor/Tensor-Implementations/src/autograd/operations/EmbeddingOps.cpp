#include "autograd/operations/EmbeddingOps.h"
#include "autograd/backward/EmbeddingBackward.h"
#include "autograd/ops_template.h"
#include "autograd/Variable.h"
#include "core/TensorImpl.h"
#include "core/AutogradMeta.h"
#include "ops/helpers/EmbeddingKernels.h"
#include <stdexcept>
#include <cstring>  // std::memcpy, std::memset

namespace OwnTensor {
namespace autograd {

Tensor embedding(const Tensor& weight, const Tensor& indices, int padding_idx) {
    // Get dimensions
    auto weight_shape = weight.shape().dims;
    if (weight_shape.size() != 2) {
        throw std::runtime_error("embedding: weight must be 2D [vocab_size, embed_dim]");
    }
    int64_t vocab_size = weight_shape[0];
    int64_t embed_dim = weight_shape[1];
    
    auto indices_shape = indices.shape().dims;
    if (indices_shape.size() < 1 || indices_shape.size() > 2) {
        throw std::runtime_error("embedding: indices must be 1D or 2D");
    }
    
    // Compute output shape: indices_shape + [embed_dim]
    std::vector<int64_t> output_dims = indices_shape;
    output_dims.push_back(embed_dim);
    Shape output_shape{output_dims};
    
    // Create output tensor
    TensorOptions opts = TensorOptions()
        .with_dtype(weight.dtype())
        .with_device(weight.device());
    Tensor output(output_shape, opts);
    
    // Get total number of lookups
    int64_t num_indices = indices.numel();
    
    // Forward pass: lookup weight rows by indices
    if (weight.device().is_cpu()) {
        const float* weight_data = weight.data<float>();
        float* output_data = output.data<float>();

        // OPTIMIZATION 1: Unified CPU lookup lambda.
        // Previously, three near-identical for-loops existed for Int64, Int32,
        // and UInt16 index types.  We collapse them into a single lambda
        // `scatter_lookup` that accepts a typed index-reader `get_idx`, keeping
        // the hot memcpy/memset path in one place and eliminating ~50 lines of
        // duplicated code.
        auto scatter_lookup = [&](auto get_idx) {
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t token_id = get_idx(i);
                float* out_row = output_data + i * embed_dim;
                if (token_id == (int64_t)padding_idx) {
                    std::memset(out_row, 0, embed_dim * sizeof(float));
                    continue;
                }
                if (token_id < 0 || token_id >= vocab_size) {
                    throw std::runtime_error(
                        "embedding: index out of range: " + std::to_string(token_id));
                }
                const float* row = weight_data + token_id * embed_dim;
                std::memcpy(out_row, row, embed_dim * sizeof(float));
            }
        };

        if (indices.dtype() == Dtype::Int64) {
            const int64_t* idx_data = indices.data<int64_t>();
            scatter_lookup([idx_data](int64_t i) -> int64_t { return idx_data[i]; });
        } else if (indices.dtype() == Dtype::Int32) {
            const int32_t* idx_data = indices.data<int32_t>();
            scatter_lookup([idx_data](int64_t i) -> int64_t {
                return static_cast<int64_t>(idx_data[i]);
            });
        } else if (indices.dtype() == Dtype::UInt16) {
            const uint16_t* idx_data = indices.data<uint16_t>();
            scatter_lookup([idx_data](int64_t i) -> int64_t {
                return static_cast<int64_t>(idx_data[i]);
            });
        } else {
            throw std::runtime_error("embedding: indices must be Int32, Int64, or UInt16");
        }
    } else {
        // CUDA: Use optimized CUDA kernel
        if (indices.dtype() == Dtype::UInt16) {
            // Already the correct type; just ensure it lives on the GPU.
            Tensor indices_cuda = indices.device().is_cpu()
                ? indices.to(weight.device())
                : indices;

            cuda::embedding_forward_cuda(
                indices_cuda.data<uint16_t>(),
                weight.data<float>(),
                output.data<float>(),
                num_indices, embed_dim, vocab_size, padding_idx,
                weight.stride().strides[0], weight.stride().strides[1]
            );
        } else {
            // OPTIMIZATION 2: Move to GPU first, then cast dtype on-device.
            // The old code converted on the CPU (Int32/Int64 → UInt16) then
            // moved the smaller result to the GPU.  That still paid the full
            // host→device PCIe transfer cost for the original large dtype.
            // We now move the original tensor to the GPU first (same PCIe cost)
            // and perform the cast entirely on-chip via a fast CUDA kernel,
            // saving the extra CPU-side allocation and dtype-conversion pass.
            Tensor indices_gpu = indices.device().is_cpu()
                ? indices.to(weight.device())
                : indices;
            Tensor indices_u16 = indices_gpu.as_type(Dtype::UInt16);

            cuda::embedding_forward_cuda(
                indices_u16.data<uint16_t>(),
                weight.data<float>(),
                output.data<float>(),
                num_indices, embed_dim, vocab_size, padding_idx,
                weight.stride().strides[0], weight.stride().strides[1]
            );
        }
    }
    
    // Set up autograd if needed
    if (weight.requires_grad()) {
        auto grad_fn = std::make_shared<EmbeddingBackward>(indices, vocab_size, embed_dim, padding_idx);
        
        Tensor& weight_mut = const_cast<Tensor&>(weight);
        grad_fn->set_next_edge(0, get_grad_edge(weight_mut));
        
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    
    return output;
}

} // namespace autograd
} // namespace OwnTensor