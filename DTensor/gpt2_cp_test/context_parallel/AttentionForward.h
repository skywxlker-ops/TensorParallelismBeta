#pragma once

namespace OwnTensor {
namespace cp {
namespace cuda {

void mem_efficient_attn_forward_tc(
    const float* query, const float* key, const float* value,
    float* output, float* lse,
    int64_t B, int64_t nh,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    int64_t hd,
    bool is_causal,
    float dropout_p, const float* dropout_mask);



} // namespace cuda
} // namespace cp
} // namespace OwnTensor
