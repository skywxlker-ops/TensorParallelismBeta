#include "autograd/backward/EmbeddingBackward.h"
#include "core/TensorImpl.h"
#include "core/AutogradMeta.h"
#include "ops/TensorOps.h"
#include "ops/helpers/EmbeddingKernels.h"
#include <stdexcept>
#include <vector>
#include <omp.h>  // OPTIMIZATION 3: OpenMP for CPU multi-threaded scatter-add

namespace OwnTensor {
namespace autograd {

EmbeddingBackward::EmbeddingBackward(const Tensor& indices, int64_t vocab_size, int64_t embed_dim, int padding_idx)
    : Node(1), 
      saved_indices_(indices, false),
      vocab_size_(vocab_size),
      embed_dim_(embed_dim),
      padding_idx_(padding_idx) {}

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
    int64_t C = embed_dim_;
    
    // Scatter-add: grad_weight[indices[n], :] += grad_output[n, :]
    int64_t N = indices.numel();
    
    // For CPU implementation
    if (grad_output.device().is_cpu()) {
        const float* grad_data = grad_output.data<float>();
        float* weight_grad_data = grad_weight.data<float>();

        // OPTIMIZATION 3: OpenMP parallel scatter-add with per-thread
        // private accumulators to eliminate data races.
        //
        // The original single-threaded loop serialized all N scatter writes onto
        // one CPU core.  With large batches (N = B*T >> thousands) and a wide
        // embed_dim (C = 768/1024), this was a significant bottleneck.
        //
        // Strategy:
        //   1. Each OpenMP thread owns a PRIVATE gradient table of the same
        //      shape [vocab_size, embed_dim] initialized to zero.
        //   2. Every thread processes its share of N tokens into its private
        //      table — no locking needed because no two threads write the
        //      same row of the SAME table.
        //   3. After the parallel region ends, we sequentially merge
        //      (reduce-add) all private tables into the final grad_weight.
        //      This serial reduction is O(num_threads * vocab_size * C) and is
        //      typically << the cost of the parallel scatter loop.
        //
        // OPTIMIZATION 4: SIMD vectorization hint on the inner channel loop.
        //   The `#pragma omp simd` directive tells the compiler auto-vectorizer
        //   to pack adjacent float operations into SIMD registers (SSE/AVX),
        //   processing 4–8 floats per CPU clock instead of 1.  This is safe
        //   here because each `c` iteration writes to a distinct memory offset.

        auto scatter_add = [&](auto get_idx) {
            int num_threads = omp_get_max_threads();

            // Allocate one private accumulator per thread: flat 1-D storage.
            std::vector<std::vector<float>> private_grads(
                num_threads,
                std::vector<float>(static_cast<size_t>(vocab_size_) * C, 0.0f)
            );

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                float* local_grad = private_grads[tid].data();

                #pragma omp for schedule(static)
                for (int64_t n = 0; n < N; ++n) {
                    int64_t token_id = get_idx(n);
                    if (token_id == (int64_t)padding_idx_) continue;
                    if (token_id >= 0 && token_id < vocab_size_) {
                        const float* src = grad_data + n * C;
                        float*       dst = local_grad + token_id * C;

                        // OPTIMIZATION 4: SIMD vectorization of the inner loop
                        #pragma omp simd
                        for (int64_t c = 0; c < C; ++c) {
                            dst[c] += src[c];
                        }
                    }
                }
            } // end parallel

            // Serial reduction: merge all private tables into grad_weight.
            for (int t = 0; t < num_threads; ++t) {
                const float* local_grad = private_grads[t].data();
                for (int64_t row = 0; row < vocab_size_; ++row) {
                    float*       dst = weight_grad_data + row * C;
                    const float* src = local_grad       + row * C;
                    #pragma omp simd
                    for (int64_t c = 0; c < C; ++c) {
                        dst[c] += src[c];
                    }
                }
            }
        };
        
        // Handle different index types
        if (indices.dtype() == Dtype::Int64) {
            const int64_t* idx_data = indices.data<int64_t>();
            scatter_add([idx_data](int64_t n) -> int64_t { return idx_data[n]; });
        } else if (indices.dtype() == Dtype::Int32) {
            const int32_t* idx_data = indices.data<int32_t>();
            scatter_add([idx_data](int64_t n) -> int64_t { return static_cast<int64_t>(idx_data[n]); });
        } else if (indices.dtype() == Dtype::UInt16) {
            const uint16_t* idx_data = indices.data<uint16_t>();
            scatter_add([idx_data](int64_t n) -> int64_t { return static_cast<int64_t>(idx_data[n]); });
        }
    } else {
        // CUDA: Use optimized CUDA kernel with atomicAdd
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
                N, C, vocab_size_, padding_idx_, 
                grad_weight.stride().strides[0], grad_weight.stride().strides[1]
            );
        } else {
            // Convert to UInt16
            Tensor indices_u16 = indices_cuda.as_type(Dtype::UInt16);
            cuda::embedding_backward_cuda(
                indices_u16.data<uint16_t>(),
                grad_output.data<float>(),
                grad_weight.data<float>(),
                N, C, vocab_size_, padding_idx_,
                grad_weight.stride().strides[0], grad_weight.stride().strides[1]
            );
        }
    }
    
    return {grad_weight};
}

} // namespace autograd
} // namespace OwnTensor