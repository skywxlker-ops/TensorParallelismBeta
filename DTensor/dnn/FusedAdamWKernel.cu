#include "dnn/FusedAdamWKernel.h"
#include "device/DeviceCore.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <algorithm>

namespace OwnTensor {
namespace cuda {

// ---------------------------------------------------------------------------
// Device helper: binary search prefix-sum offsets to map global_idx to
// (tensor_id, local_idx). Identical layout to MultiTensorKernels.cu.
// ---------------------------------------------------------------------------
__device__ __forceinline__
void faw_find_tensor(
    const int64_t* __restrict__ offsets,
    int num_tensors,
    int64_t global_idx,
    int& tensor_id,
    int64_t& local_idx)
{
    int lo = 0, hi = num_tensors - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (offsets[mid] <= global_idx) lo = mid;
        else                             hi = mid - 1;
    }
    tensor_id = lo;
    local_idx = global_idx - offsets[lo];
}

// ---------------------------------------------------------------------------
// Fused AdamW kernel with inline unscale + clip
// ---------------------------------------------------------------------------
__global__ void fused_adamw_unscale_kernel(
    const TensorInfo* __restrict__ params,
    const TensorInfo* __restrict__ grads,
    const TensorInfo* __restrict__ ms,
    const TensorInfo* __restrict__ vs,
    const int64_t* __restrict__ offsets,
    int num_tensors,
    int64_t total_work,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2,
    float inv_scale, float clip_coeff)
{
    const float g_scale = inv_scale * clip_coeff;
    const int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t global_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         global_idx < total_work;
         global_idx += grid_stride)
    {
        int t;
        int64_t local_idx;
        faw_find_tensor(offsets, num_tensors, global_idx, t, local_idx);

        float* p = params[t].ptr;
        const float* g_ptr = grads[t].ptr;
        float* m_ptr = ms[t].ptr;
        float* v_ptr = vs[t].ptr;

        float grad  = g_ptr[local_idx] * g_scale;
        float param = p[local_idx];

        float m_val = beta1 * m_ptr[local_idx] + (1.0f - beta1) * grad;
        float v_val = beta2 * v_ptr[local_idx] + (1.0f - beta2) * grad * grad;

        m_ptr[local_idx] = m_val;
        v_ptr[local_idx] = v_val;

        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;

        // AdamW decoupled weight decay
        p[local_idx] = param - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

// ---------------------------------------------------------------------------
// Persistent GPU metadata + offset buffers (identical scheme to MultiTensorKernels)
// ---------------------------------------------------------------------------

static const size_t FAW_MAX_TENSORS = 512;

static TensorInfo* faw_d_params  = nullptr;
static TensorInfo* faw_h_params  = nullptr;
static TensorInfo* faw_d_grads   = nullptr;
static TensorInfo* faw_h_grads   = nullptr;
static TensorInfo* faw_d_ms      = nullptr;
static TensorInfo* faw_h_ms      = nullptr;
static TensorInfo* faw_d_vs      = nullptr;
static TensorInfo* faw_h_vs      = nullptr;
static int64_t*    faw_d_offsets = nullptr;
static int64_t*    faw_h_offsets = nullptr;
static int         faw_num_sms   = 0;

static void faw_ensure_buffers() {
    if (faw_d_params) return;
    cudaMalloc    (&faw_d_params,  FAW_MAX_TENSORS * sizeof(TensorInfo));
    cudaMallocHost(&faw_h_params,  FAW_MAX_TENSORS * sizeof(TensorInfo));
    cudaMalloc    (&faw_d_grads,   FAW_MAX_TENSORS * sizeof(TensorInfo));
    cudaMallocHost(&faw_h_grads,   FAW_MAX_TENSORS * sizeof(TensorInfo));
    cudaMalloc    (&faw_d_ms,      FAW_MAX_TENSORS * sizeof(TensorInfo));
    cudaMallocHost(&faw_h_ms,      FAW_MAX_TENSORS * sizeof(TensorInfo));
    cudaMalloc    (&faw_d_vs,      FAW_MAX_TENSORS * sizeof(TensorInfo));
    cudaMallocHost(&faw_h_vs,      FAW_MAX_TENSORS * sizeof(TensorInfo));
    cudaMalloc    (&faw_d_offsets, (FAW_MAX_TENSORS + 1) * sizeof(int64_t));
    cudaMallocHost(&faw_h_offsets, (FAW_MAX_TENSORS + 1) * sizeof(int64_t));
}

static int faw_get_sms() {
    if (faw_num_sms == 0) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&faw_num_sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return faw_num_sms;
}

static int faw_grid(int64_t total_work, int block_size) {
    int min_blocks = faw_get_sms() * 4;
    int work_blocks = (int)std::min((int64_t)INT32_MAX,
                                    (total_work + block_size - 1) / block_size);
    return std::max(min_blocks, std::min(work_blocks, 1024));
}

static void faw_build_offsets(const TensorInfo* tensors, int n, cudaStream_t stream) {
    faw_h_offsets[0] = 0;
    for (int i = 0; i < n; i++)
        faw_h_offsets[i + 1] = faw_h_offsets[i] + tensors[i].numel;
    cudaMemcpyAsync(faw_d_offsets, faw_h_offsets,
                    (n + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
}

// ---------------------------------------------------------------------------
// Public launcher
// ---------------------------------------------------------------------------

void fused_adamw_with_unscale_cuda(
    const std::vector<TensorInfo>& params,
    const std::vector<TensorInfo>& grads,
    const std::vector<TensorInfo>& ms,
    const std::vector<TensorInfo>& vs,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2,
    float inv_scale, float clip_coeff)
{
    if (params.empty()) return;
    faw_ensure_buffers();

    int n = std::min((int)params.size(), (int)FAW_MAX_TENSORS);
    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();

    std::memcpy(faw_h_params, params.data(), n * sizeof(TensorInfo));
    cudaMemcpyAsync(faw_d_params, faw_h_params, n * sizeof(TensorInfo),
                    cudaMemcpyHostToDevice, stream);

    std::memcpy(faw_h_grads, grads.data(), n * sizeof(TensorInfo));
    cudaMemcpyAsync(faw_d_grads, faw_h_grads, n * sizeof(TensorInfo),
                    cudaMemcpyHostToDevice, stream);

    std::memcpy(faw_h_ms, ms.data(), n * sizeof(TensorInfo));
    cudaMemcpyAsync(faw_d_ms, faw_h_ms, n * sizeof(TensorInfo),
                    cudaMemcpyHostToDevice, stream);

    std::memcpy(faw_h_vs, vs.data(), n * sizeof(TensorInfo));
    cudaMemcpyAsync(faw_d_vs, faw_h_vs, n * sizeof(TensorInfo),
                    cudaMemcpyHostToDevice, stream);

    faw_build_offsets(params.data(), n, stream);
    int64_t total_work = faw_h_offsets[n];

    int threads = 256;
    int blocks  = faw_grid(total_work, threads);

    fused_adamw_unscale_kernel<<<blocks, threads, 0, stream>>>(
        faw_d_params, faw_d_grads, faw_d_ms, faw_d_vs,
        faw_d_offsets, n, total_work,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2,
        inv_scale, clip_coeff);
}

} // namespace cuda
} // namespace OwnTensor
