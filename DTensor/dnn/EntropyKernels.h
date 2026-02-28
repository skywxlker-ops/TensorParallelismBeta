#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace OwnTensor {
namespace cuda {

void launch_extract_target_logits(const float* logits, const float* targets, float* out,
                                  int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream);

void launch_sparse_subtract(float* grad, const float* targets, float g_out,
                           int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream);

void launch_vocab_parallel_ce_backward(
    const float* probs, const float* targets, const float* grad_out,
    float scale, float* grad_logits,
    int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream);

void launch_vocab_parallel_ce_fused_forward(
    const float* logits, const float* targets,
    float* probs, float* local_max, float* local_sum_exp, float* target_logit,
    int64_t B, int64_t T, int64_t V_local, int64_t start_v, cudaStream_t stream);

void launch_normalize_probs(
    float* probs, const float* global_sum_exp,
    int64_t BT, int64_t V_local, cudaStream_t stream);

void launch_rescale_probs(
    float* probs, const float* scale,
    int64_t BT, int64_t V_local, cudaStream_t stream);

} // namespace cuda
} // namespace OwnTensor
