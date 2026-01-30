#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace DistributedLoss {

template <typename T, typename IndexT>
void launch_distributed_sparse_ce_forward(
    const T* logits_shard,
    const IndexT* targets,
    T* target_logits_shard,
    int64_t batch_size,
    int64_t num_classes_local,
    int64_t vocab_offset,
    cudaStream_t stream = 0
);

template <typename T, typename IndexT>
void launch_distributed_sparse_ce_backward(
    T* grad_logits_shard,
    const T* logits_shard,
    const IndexT* targets,
    const T* max_logits,
    const T* sum_exps,
    const T* grad_output,
    int64_t batch_size,
    int64_t num_classes_local,
    int64_t vocab_offset,
    cudaStream_t stream = 0
);

} // namespace cuda
