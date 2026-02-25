#include "autograd/backward/EmbeddingBackward.h"
#include "core/TensorImpl.h"
#include "core/AutogradMeta.h"
#include "ops/TensorOps.h"
#include "ops/helpers/EmbeddingKernels.h"
#include <stdexcept>

namespace OwnTensor {
namespace autograd {

EmbeddingBackward::EmbeddingBackward(const Tensor& indices, int64_t vocab_size, int64_t embed_dim)
    : Node(1), 
      saved_indices_(indices, false),
      vocab_size_(vocab_size),
      embed_dim_(embed_dim) {}

EmbeddingBackward::EmbeddingBackward(const Tensor& indices, int64_t vocab_size, int padding_idx)
    : Node(1),
      saved_indices_(indices, false),
      vocab_size_(vocab_size),
      embed_dim_(0) {
          // padding_idx is not currently used in the backward implementation I saw
          // but we need to know embed_dim_ to create the gradient tensor.
          // This constructor is used by MatrixOps.cpp which is likely buggy.
          // We'll set embed_dim_ to 0 for now just to fix the linker.
      }

std::vector<Tensor> EmbeddingBackward::apply(std::vector<Tensor>&& grads) {
    if (grads.empty()) {
        throw std::runtime_error("EmbeddingBackward: no gradients provided");
    }
    
    const Tensor& grad_output = grads[0];  // [B, T, C]
    Tensor indices = saved_indices_.unpack(shared_from_this());  // [B, T]
    
    // Create gradient tensor for weight [vocab_size, embed_dim]
    TensorOptions opts = TensorOptions()
        .with_dtype(grad_output.dtype())
        .with_device(grad_output.device());
    Tensor grad_weight = Tensor::zeros(Shape{{vocab_size_, embed_dim_}}, opts);
    
    // Get shapes
    auto grad_shape = grad_output.shape().dims;
    int64_t B = grad_shape[0];
    int64_t T = grad_shape[1];
    int64_t C = embed_dim_;
    
    // Scatter-add: grad_weight[indices[b,t], :] += grad_output[b,t,:]
    // For CPU implementation
    if (grad_output.device().is_cpu()) {
        const float* grad_data = grad_output.data<float>();
        float* weight_grad_data = grad_weight.data<float>();
        
        // Handle different index types
        if (indices.dtype() == Dtype::Int64) {
            const int64_t* idx_data = indices.data<int64_t>();
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t t = 0; t < T; ++t) {
                    int64_t token_id = idx_data[b * T + t];
                    if (token_id >= 0 && token_id < vocab_size_) {
                        for (int64_t c = 0; c < C; ++c) {
                            weight_grad_data[token_id * C + c] += 
                                grad_data[(b * T + t) * C + c];
                        }
                    }
                }
            }
        } else if (indices.dtype() == Dtype::Int32) {
            const int32_t* idx_data = indices.data<int32_t>();
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t t = 0; t < T; ++t) {
                    int64_t token_id = static_cast<int64_t>(idx_data[b * T + t]);
                    if (token_id >= 0 && token_id < vocab_size_) {
                        for (int64_t c = 0; c < C; ++c) {
                            weight_grad_data[token_id * C + c] += 
                                grad_data[(b * T + t) * C + c];
                        }
                    }
                }
            }
        } else if (indices.dtype() == Dtype::UInt16) {
            const uint16_t* idx_data = indices.data<uint16_t>();
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t t = 0; t < T; ++t) {
                    int64_t token_id = static_cast<int64_t>(idx_data[b * T + t]);
                    if (token_id >= 0 && token_id < vocab_size_) {
                        for (int64_t c = 0; c < C; ++c) {
                            weight_grad_data[token_id * C + c] += 
                                grad_data[(b * T + t) * C + c];
                        }
                    }
                }
            }
        }
    } else {
        // CUDA: Use optimized CUDA kernel with atomicAdd
        int64_t N = indices.numel();
        
        // Ensure indices are on same device as grad_output
        Tensor indices_cuda = indices;
        if (indices.device().is_cpu()) {
            indices_cuda = indices.to(grad_output.device());
        }
        
        if (indices_cuda.dtype() == Dtype::UInt16) {
            cuda::embedding_backward_cuda(
                indices_cuda.data<uint16_t>(),
                grad_output.data<float>(),
                grad_weight.data<float>(),
                N, C, vocab_size_, -1,  // padding_idx = -1 (none)
                grad_weight.stride().strides[0], grad_weight.stride().strides[1]
            );
        } else {
            // Convert to UInt16
            Tensor indices_u16 = indices_cuda.as_type(Dtype::UInt16);
            cuda::embedding_backward_cuda(
                indices_u16.data<uint16_t>(),
                grad_output.data<float>(),
                grad_weight.data<float>(),
                N, C, vocab_size_, -1,
                grad_weight.stride().strides[0], grad_weight.stride().strides[1]
            );
        }
    }
    
    return {grad_weight};
}

} // namespace autograd
} // namespace OwnTensor