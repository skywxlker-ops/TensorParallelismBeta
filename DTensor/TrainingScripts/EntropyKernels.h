#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace OwnTensor {
namespace cuda {

void launch_extract_target_logits(const float* logits, const float* targets, float* out,
                                  int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream);

void launch_sparse_subtract(float* grad, const float* targets, float g_out,
                           int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream);

void launch_vocab_parallel_embedding_fwd(
    const int64_t* input, const float* weight, float* output,
    int64_t B, int64_t T, int64_t D,
    int64_t start_v, int64_t end_v, cudaStream_t stream);

void launch_vocab_parallel_embedding_bwd(
    const int64_t* input, const float* grad_output, float* grad_weight,
    int64_t B, int64_t T, int64_t D,
    int64_t start_v, int64_t end_v, cudaStream_t stream);

// Fused forward: sum_exp + target_logit in one pass over logits
void launch_fused_sum_exp_target(
    const float* logits, const float* global_max, const float* targets,
    float* packed_out, int64_t B, int64_t T, int64_t V_local,
    int64_t start_v, cudaStream_t stream);

// Fused backward: softmax + grad_scale + sparse_subtract (reads grad_output from device)
void launch_vocab_parallel_ce_backward(
    const float* logits, const float* targets,
    const float* sum_exp, const float* max_logits,
    const float* grad_output, float* grad_logits,
    int64_t B, int64_t T, int64_t V_local,
    int64_t start_v, float scale, cudaStream_t stream);

} // namespace cuda
} // namespace OwnTensor
