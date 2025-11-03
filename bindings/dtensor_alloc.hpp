#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

// --------- light helpers ----------
static inline void CUDA_CHECK(cudaError_t e){
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static inline void NCCL_CHECK(ncclResult_t r){
  if(r!=ncclSuccess) throw std::runtime_error(ncclGetErrorString(r));
}
static inline void CUBLAS_CHECK(cublasStatus_t s){
  if(s!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error");
}

// ---------------- Work ----------------
class Work {
public:
  explicit Work(cudaStream_t stream);
  ~Work();

  void markCompleted(bool success = true);
  bool wait();

private:
  cudaStream_t stream_{};
  cudaEvent_t  event_{};
  bool         completed_{false};
  bool         success_{true};
};

// ---------------- ProcessGroup ----------------
class ProcessGroup {
public:
  ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id);
  ~ProcessGroup();

  template<typename T>
  std::shared_ptr<Work> allReduce(T* data, size_t count, ncclDataType_t dtype) {
    auto w = std::make_shared<Work>(stream_);
    NCCL_CHECK(ncclAllReduce(data, data, count, dtype, ncclSum, comm_, stream_));
    w->markCompleted(true);
    return w;
  }
  template<typename T>
  std::shared_ptr<Work> reduceScatter(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
    auto w = std::make_shared<Work>(stream_);
    NCCL_CHECK(ncclReduceScatter(send_buf, recv_buf, count_per_rank, dtype, ncclSum, comm_, stream_));
    w->markCompleted(true);
    return w;
  }
  template<typename T>
  std::shared_ptr<Work> allGather(T* recv_buf, T* send_buf, size_t count_per_rank, ncclDataType_t dtype) {
    auto w = std::make_shared<Work>(stream_);
    NCCL_CHECK(ncclAllGather(send_buf, recv_buf, count_per_rank, dtype, comm_, stream_));
    w->markCompleted(true);
    return w;
  }
  template<typename T>
  std::shared_ptr<Work> broadcast(T* data, size_t count, int root, ncclDataType_t dtype) {
    auto w = std::make_shared<Work>(stream_);
    NCCL_CHECK(ncclBroadcast(data, data, count, dtype, root, comm_, stream_));
    w->markCompleted(true);
    return w;
  }

  // Row-major GEMM (your original)
  void gemm_f32_rowmajor(const float* A, const float* B, float* C,
                         int m, int n, int k, bool transA=false, bool transB=false);

  // Convenience wrapper used by Python: C[m,n] = A[m,k] * B[k,n]
  inline void gemm_f32(const float* A, const float* B, float* C,
                       int m, int n, int k) {
    gemm_f32_rowmajor(A, B, C, m, n, k, false, false);
  }

  cudaStream_t getStream() { return stream_; }
  int rank() const { return rank_; }
  int worldSize() const { return world_size_; }

private:
  int            rank_{0};
  int            world_size_{1};
  int            device_{0};
  cudaStream_t   stream_{};
  ncclComm_t     comm_{};
  cublasHandle_t cublas_{nullptr};
};

// ---------------- DTensor ----------------
class DTensor {
public:
  DTensor(int world_size, int slice_size, int rank);
  ~DTensor();

  float* deviceData();
  std::vector<float>& hostData();
  size_t size() const;

  void copyDeviceToHost();
  void copyFromHost(const float* src);

private:
  int world_size_{1};
  int slice_size_{0};
  int rank_{0};
  std::vector<float> h_data_;
  float* d_data_{nullptr};

  // track bytes for allocator
  size_t bytes_{0};
};

// -------- Allocator stats helpers (callable from Python via bindings) --------
// Global-namespace versions:
std::string allocator_stats();
void        allocator_print_stats();

// Also provide dt::-qualified aliases for compatibility with bindings that
// expect dt::allocator_stats() / dt::allocator_print_stats().
namespace dt {
  std::string allocator_stats();
  void        allocator_print_stats();
}
