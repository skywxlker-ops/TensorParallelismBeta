#include "dtensor_opt.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cmath>


#define CUDA_CHECK_THROW(x) do { auto _e=(x); if(_e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(_e)); } while(0)

// ===================== GELU helpers =====================
static __device__ __forceinline__ float gelu_f(float x) {
  const float k = sqrtf(2.f / M_PI);
  return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}
static __device__ __forceinline__ float gelu_prime_f(float x) {
  const float k = sqrtf(2.f / M_PI);
  float s = k * (x + 0.044715f * x * x * x);
  float t = tanhf(s);
  float ds = k * (1.f + 3.f * 0.044715f * x * x);
  return 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * ds;
}

// ===================== KERNELS =====================
__global__ void k_add_bias(float* __restrict__ Y, const float* __restrict__ b, int B, int F) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = B * F;
  for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
    int f = i % F;
    Y[i] += b[f];
  }
}

__global__ void k_gelu(float* __restrict__ Y, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) Y[i] = gelu_f(Y[i]);
}

__global__ void k_gelu_bw(const float* __restrict__ Y, float* __restrict__ dY, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) dY[i] *= gelu_prime_f(Y[i]);
}

__global__ void k_mse_grad_and_loss(float* __restrict__ dZ,
                                    const float* __restrict__ Z,
                                    const float* __restrict__ T,
                                    int N, float invN, float* __restrict__ loss) {
  extern __shared__ float ssum[];
  float local = 0.f;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int idx = i; idx < N; idx += gridDim.x * blockDim.x) {
    float diff = Z[idx] - T[idx];
    dZ[idx] = 2.f * diff * invN;
    local += diff * diff;
  }
  int tid = threadIdx.x;
  ssum[tid] = local;
  __syncthreads();
  for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
    if (tid < s) ssum[tid] += ssum[tid + s];
    __syncthreads();
  }
  // last warp
  if (tid < 32) {
    volatile float* v = ssum;
    v[tid] += v[tid + 16];
    v[tid] += v[tid + 8];
    v[tid] += v[tid + 4];
    v[tid] += v[tid + 2];
    v[tid] += v[tid + 1];
  }
  if (tid == 0) atomicAdd(loss, ssum[0]);
}

__global__ void k_reduce_cols_sum(const float* __restrict__ A,
                                  float* __restrict__ colsum,
                                  int B, int F) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= F) return;
  float s = 0.f;
  for (int b = 0; b < B; ++b) s += A[b*F + f];
  colsum[f] = s;
}


// ---------------- Work ----------------
Work::Work(cudaStream_t stream) : stream_(stream) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
}
Work::~Work() { CUDA_CHECK(cudaEventDestroy(event_)); }

void Work::markCompleted(bool success) {
    success_ = success;
    completed_ = true;
    CUDA_CHECK(cudaEventRecord(event_, stream_));
}
bool Work::wait() {
    if (!completed_) return false;
    CUDA_CHECK(cudaEventSynchronize(event_));
    return success_;
}

// ---------------- ProcessGroup ----------------
ProcessGroup::ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id)
    : rank_(rank), world_size_(world_size), device_(device) {
    CUDA_CHECK(cudaSetDevice(device_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    NCCL_CHECK(ncclCommInitRank(&comm_, world_size_, id, rank_));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
}

ProcessGroup::~ProcessGroup() {
    if (cublas_) { cublasDestroy(cublas_); }
    if (comm_)   { ncclCommDestroy(comm_); }
    if (stream_) { CUDA_CHECK(cudaStreamDestroy(stream_)); }
}

// Row-major GEMM using cuBLAS (which is column-major).
// We compute C = A * B with row-major storage by swapping operands:
//   cublas (col-major): C = op(B)^T * op(A)^T  -> request (n x m)
// So call sgemm as: (n,m,k), with opB/opA chosen from trans flags.
void ProcessGroup::gemm_f32_rowmajor(const float* A, const float* B, float* C,
                                     int m, int n, int k, bool transA, bool transB)
{
    // op flags (row-major A[m,k], B[k,n])
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

    // In cuBLAS column-major call:
    // sizes are (n, m, k), operands swapped
    int lda = (opA == CUBLAS_OP_N) ? k : m;   // leading dim of A (row-major)
    int ldb = (opB == CUBLAS_OP_N) ? n : k;   // leading dim of B (row-major)
    int ldc = n;                               // leading dim of C (row-major)

    const float alpha = 1.0f, beta = 0.0f;

    // Note: pass B then A, and request (n x m)
    CUBLAS_CHECK(cublasSgemm(
        cublas_,
        opB,                // op on B
        opA,                // op on A
        n,                  // rows of op(B)
        m,                  // cols of op(A)
        k,                  // shared dim
        &alpha,
        B, ldb,
        A, lda,
        &beta,
        C, ldc
    ));
}

// ---------------- DTensor ----------------
DTensor::DTensor(int world_size, int slice_size, int rank)
    : world_size_(world_size), slice_size_(slice_size), rank_(rank)
{
    h_data_.resize(slice_size_);
    for (int j = 0; j < slice_size_; ++j) {
        h_data_[j] = float(rank_ * slice_size_ + j);
    }
    CUDA_CHECK(cudaMalloc(&d_data_, slice_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data_, h_data_.data(), slice_size_ * sizeof(float), cudaMemcpyHostToDevice));
}
DTensor::~DTensor() { if (d_data_) CUDA_CHECK(cudaFree(d_data_)); }

float* DTensor::deviceData() { return d_data_; }
std::vector<float>& DTensor::hostData() { return h_data_; }
size_t DTensor::size() const { return static_cast<size_t>(slice_size_); }
void DTensor::copyDeviceToHost() {
    CUDA_CHECK(cudaMemcpy(h_data_.data(), d_data_, slice_size_ * sizeof(float), cudaMemcpyDeviceToHost));
}
void DTensor::copyFromHost(const float* src) {
    CUDA_CHECK(cudaMemcpy(d_data_, src, slice_size_ * sizeof(float), cudaMemcpyHostToDevice));
}


void DeviceOps::set_stream(std::uintptr_t stream_ptr) {
  stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
}
std::uintptr_t DeviceOps::get_stream() const { return reinterpret_cast<std::uintptr_t>(stream_); }

void DeviceOps::add_bias_inplace(std::uintptr_t Y, std::uintptr_t b, int B, int F) {
  int N = B * F, th = 256, bl = (N + th - 1) / th;
  k_add_bias<<<bl, th, 0, stream_>>>(reinterpret_cast<float*>(Y), reinterpret_cast<const float*>(b), B, F);
}

void DeviceOps::gelu_inplace(std::uintptr_t Y, int N) {
  int th = 256, bl = (N + th - 1) / th;
  k_gelu<<<bl, th, 0, stream_>>>(reinterpret_cast<float*>(Y), N);
}

void DeviceOps::gelu_backward_inplace(std::uintptr_t Y, std::uintptr_t dY, int N) {
  int th = 256, bl = (N + th - 1) / th;
  k_gelu_bw<<<bl, th, 0, stream_>>>(reinterpret_cast<const float*>(Y),
                                    reinterpret_cast<float*>(dY), N);
}

void DeviceOps::mse_grad_and_loss(std::uintptr_t dZ, std::uintptr_t Z, std::uintptr_t T,
                                  int N, float invN, std::uintptr_t loss_dev) {
  CUDA_CHECK_THROW(cudaMemsetAsync(reinterpret_cast<void*>(loss_dev), 0, sizeof(float), stream_));
  int th = 256;
  int bl = (N + th - 1) / th;
  size_t sh = th * sizeof(float);
  k_mse_grad_and_loss<<<bl, th, sh, stream_>>>(reinterpret_cast<float*>(dZ),
                                               reinterpret_cast<const float*>(Z),
                                               reinterpret_cast<const float*>(T),
                                               N, invN, reinterpret_cast<float*>(loss_dev));
}

void DeviceOps::reduce_cols_sum(std::uintptr_t A, std::uintptr_t colsum, int B, int F) {
  int th = 256;
  int bl = (F + th - 1) / th;
  k_reduce_cols_sum<<<bl, th, 0, stream_>>>(reinterpret_cast<const float*>(A),
                                            reinterpret_cast<float*>(colsum),
                                            B, F);
}

void DeviceOps::memcpy_d2d(std::uintptr_t dst, std::uintptr_t src, size_t N) {
  CUDA_CHECK_THROW(cudaMemcpyAsync(reinterpret_cast<void*>(dst),
                                   reinterpret_cast<const void*>(src),
                                   N * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_));
}

// ===================== AdamW =====================
__global__ void k_adamw(float* __restrict__ p,
                        const float* __restrict__ g,
                        float* __restrict__ m,
                        float* __restrict__ v,
                        size_t n, float lr, float b1, float b2,
                        float eps, float wd, float b1t, float b2t)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t idx = i; idx < n; idx += gridDim.x * blockDim.x) {
    float gi = g[idx];
    float mi = m[idx] = b1 * m[idx] + (1.f - b1) * gi;
    float vi = v[idx] = b2 * v[idx] + (1.f - b2) * gi * gi;
    float mhat = mi / (1.f - b1t);
    float vhat = vi / (1.f - b2t);
    float up = mhat / (sqrtf(vhat) + eps) + wd * p[idx]; // decoupled wd
    p[idx] -= lr * up;
  }
}

AdamW::AdamW(float lr, float beta1, float beta2, float eps, float wd)
: lr_(lr), b1_(beta1), b2_(beta2), eps_(eps), wd_(wd) {}

AdamW::~AdamW() {
  if (m_) cudaFree(m_);
  if (v_) cudaFree(v_);
}

void AdamW::attach_buffers(size_t n) {
  if (n <= cap_) return;
  if (m_) cudaFree(m_);
  if (v_) cudaFree(v_);
  CUDA_CHECK_THROW(cudaMalloc(&m_, n * sizeof(float)));
  CUDA_CHECK_THROW(cudaMalloc(&v_, n * sizeof(float)));
  CUDA_CHECK_THROW(cudaMemset(m_, 0, n * sizeof(float)));
  CUDA_CHECK_THROW(cudaMemset(v_, 0, n * sizeof(float)));
  cap_ = n;
}

void AdamW::step(std::uintptr_t param, std::uintptr_t grad, size_t n,
                 float t, std::uintptr_t stream_ptr) {
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
  attach_buffers(n);
  float b1t = std::pow(b1_, t);
  float b2t = std::pow(b2_, t);
  int th = 256;
  int bl = (int)std::min((n + th - 1) / th, (size_t)65535);
  k_adamw<<<bl, th, 0, s>>>(reinterpret_cast<float*>(param),
                            reinterpret_cast<const float*>(grad),
                            m_, v_, n, lr_, b1_, b2_, eps_, wd_, b1t, b2t);
}

void AdamW::step_bias(std::uintptr_t bias, std::uintptr_t grad, size_t n,
                      float t, std::uintptr_t stream_ptr) {
  float wd_saved = wd_;
  wd_ = 0.f;
  step(bias, grad, n, t, stream_ptr);
  wd_ = wd_saved;
}