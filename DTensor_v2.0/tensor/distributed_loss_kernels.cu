#include "tensor/distributed_loss_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

namespace DistributedLoss {

template <typename T, typename IndexT>
__global__ void distributed_sparse_ce_forward_kernel(
    const T* logits_shard,
    const IndexT* targets,
    T* target_logits_shard,
    int64_t batch_size,
    int64_t num_classes_local,
    int64_t vocab_offset) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    
    IndexT target = targets[i];
    if (target >= (IndexT)vocab_offset && target < (IndexT)(vocab_offset + num_classes_local)) {
        target_logits_shard[i] = logits_shard[i * num_classes_local + (target - vocab_offset)];
    } else {
        target_logits_shard[i] = 0;
    }
}

template <typename T, typename IndexT>
__global__ void distributed_sparse_ce_backward_kernel(
    T* grad_logits_shard,
    const T* logits_shard,
    const IndexT* targets,
    const T* max_logits,
    const T* sum_exps,
    const T* grad_output,
    int64_t batch_size,
    int64_t num_classes_local,
    int64_t vocab_offset) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_classes_local) return;
    
    int i = idx / num_classes_local;
    int j = idx % num_classes_local;
    
    T max_l = max_logits[i];
    T sum_e = sum_exps[i];
    T g_out = grad_output[0];
    
    T logit = logits_shard[idx];
    T softmax_val = expf((float)logit - (float)max_l) / (float)sum_e;
    
    IndexT target = targets[i];
    T indicator = (target == (IndexT)(j + vocab_offset)) ? 1.0f : 0.0f;
    
    grad_logits_shard[idx] = (T)((softmax_val - indicator) * (float)g_out / (float)batch_size);
}

template <typename T, typename IndexT>
void launch_distributed_sparse_ce_forward(
    const T* logits_shard,
    const IndexT* targets,
    T* target_logits_shard,
    int64_t batch_size,
    int64_t num_classes_local,
    int64_t vocab_offset,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    distributed_sparse_ce_forward_kernel<T, IndexT><<<blocks, threads, 0, stream>>>(
        logits_shard, targets, target_logits_shard, batch_size, num_classes_local, vocab_offset);
}

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
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (batch_size * num_classes_local + threads - 1) / threads;
    distributed_sparse_ce_backward_kernel<T, IndexT><<<blocks, threads, 0, stream>>>(
        grad_logits_shard, logits_shard, targets, max_logits, sum_exps, grad_output, batch_size, num_classes_local, vocab_offset);
}

// Explicit instantiations
template void launch_distributed_sparse_ce_forward<float, uint16_t>(const float*, const uint16_t*, float*, int64_t, int64_t, int64_t, cudaStream_t);
template void launch_distributed_sparse_ce_backward<float, uint16_t>(float*, const float*, const uint16_t*, const float*, const float*, const float*, int64_t, int64_t, int64_t, cudaStream_t);
template void launch_distributed_sparse_ce_forward<float, int64_t>(const float*, const int64_t*, float*, int64_t, int64_t, int64_t, cudaStream_t);
template void launch_distributed_sparse_ce_backward<float, int64_t>(float*, const float*, const int64_t*, const float*, const float*, const float*, int64_t, int64_t, int64_t, cudaStream_t);

} // namespace cuda
