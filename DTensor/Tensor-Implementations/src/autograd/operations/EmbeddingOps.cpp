#include "autograd/operations/EmbeddingOps.h"
#include "autograd/backward/EmbeddingBackward.h"
#include "autograd/ops_template.h"
#include "autograd/Variable.h"
#include "core/TensorImpl.h"
#include "core/AutogradMeta.h"
#include "ops/helpers/EmbeddingKernels.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

Tensor embedding(const Tensor& weight, const Tensor& indices) {
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
        
        if (indices.dtype() == Dtype::Int64) {
            const int64_t* idx_data = indices.data<int64_t>();
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t token_id = idx_data[i];
                if (token_id < 0 || token_id >= vocab_size) {
                    throw std::runtime_error("embedding: index out of range: " + std::to_string(token_id));
                }
                const float* row = weight_data + token_id * embed_dim;
                float* out_row = output_data + i * embed_dim;
                for (int64_t c = 0; c < embed_dim; ++c) {
                    out_row[c] = row[c];
                }
            }
        } else if (indices.dtype() == Dtype::Int32) {
            const int32_t* idx_data = indices.data<int32_t>();
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t token_id = static_cast<int64_t>(idx_data[i]);
                if (token_id < 0 || token_id >= vocab_size) {
                    throw std::runtime_error("embedding: index out of range");
                }
                const float* row = weight_data + token_id * embed_dim;
                float* out_row = output_data + i * embed_dim;
                for (int64_t c = 0; c < embed_dim; ++c) {
                    out_row[c] = row[c];
                }
            }
        } else if (indices.dtype() == Dtype::UInt16) {
            const uint16_t* idx_data = indices.data<uint16_t>();
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t token_id = static_cast<int64_t>(idx_data[i]);
                if (token_id < 0 || token_id >= vocab_size) {
                    throw std::runtime_error("embedding: index out of range");
                }
                const float* row = weight_data + token_id * embed_dim;
                float* out_row = output_data + i * embed_dim;
                for (int64_t c = 0; c < embed_dim; ++c) {
                    out_row[c] = row[c];
                }
            }
        } else {
            throw std::runtime_error("embedding: indices must be Int32, Int64, or UInt16");
        }
    } else {
        // CUDA: Use optimized CUDA kernel
        if (indices.dtype() == Dtype::UInt16) {
            // Ensure indices are on same device as weight
            Tensor indices_cuda = indices.device().is_cpu() ? indices.to(weight.device()) : indices;
            
            cuda::embedding_forward_cuda(
                indices_cuda.data<uint16_t>(),
                weight.data<float>(),
                output.data<float>(),
                num_indices, embed_dim, vocab_size, -1,  // padding_idx = -1 (none)
                weight.stride().strides[0], weight.stride().strides[1]
            );
        } else {
            // For other index types, convert to UInt16 on device
            Tensor indices_u16 = indices.as_type(Dtype::UInt16);
            if (indices_u16.device().is_cpu()) {
                indices_u16 = indices_u16.to(weight.device());
            }
            
            cuda::embedding_forward_cuda(
                indices_u16.data<uint16_t>(),
                weight.data<float>(),
                output.data<float>(),
                num_indices, embed_dim, vocab_size, -1,
                weight.stride().strides[0], weight.stride().strides[1]
            );
        }
    }
    
    // Set up autograd if needed
    if (weight.requires_grad()) {
        auto grad_fn = std::make_shared<EmbeddingBackward>(indices, vocab_size, embed_dim);
        
        Tensor& weight_mut = const_cast<Tensor&>(weight);
        grad_fn->set_next_edge(0, get_grad_edge(weight_mut));
        
        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }
    
    return output;
}

} // namespace autograd
} // namespace OwnTensor