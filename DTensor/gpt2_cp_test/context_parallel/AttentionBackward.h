#pragma once

namespace OwnTensor {
namespace cp {
namespace cuda {

void mem_efficient_attn_backward(
    const float* query, const float* key, const float* value,
    const float* output, const float* grad_output, const float* lse,
    float* grad_query, float* grad_key, float* grad_value,
    float* D_buf,
    int64_t B, int64_t nh,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    int64_t hd,
    bool is_causal);

} // namespace cuda
} // namespace cp
} // namespace OwnTensor
