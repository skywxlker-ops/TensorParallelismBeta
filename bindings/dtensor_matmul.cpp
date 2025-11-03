#include "dtensor_matmul.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

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
