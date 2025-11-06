#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <vector>
#include <string>
#include <memory>   
#include <stdexcept>
#include "TensorLib.h" 
#include "cachingAllocator.hpp"  

// ---- helpers (declare before any use) ----
static inline void CUDA_CHECK(cudaError_t e){
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static inline void NCCL_CHECK(ncclResult_t r){
  if(r!=ncclSuccess) throw std::runtime_error(ncclGetErrorString(r));
}
static inline void CUBLAS_CHECK(cublasStatus_t s){
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error");
}
static inline void CUBLAS_CHECK(cublasStatus_t s, const char* what){
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("cuBLAS error in ") + what);
}

// ---- Work ----
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

// ---- ProcessGroup ----
class ProcessGroup {
public:
  ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id);
  ~ProcessGroup();

// ---- Collectives (templated inline; float wrappers for pybind) ----
template<typename T>
std::shared_ptr<Work> allReduce(T* data, size_t count, ncclDataType_t dtype) {
  auto w = std::make_shared<Work>(stream_);
  NCCL_CHECK(ncclAllReduce((const void*)data, (void*)data, count, dtype, ncclSum, comm_, stream_));
  w->markCompleted(true);
  return w;
}


template<typename T>
std::shared_ptr<Work> reduceScatter(T* send_buf, T* recv_buf, size_t recv_count, ncclDataType_t dtype) {
  auto w = std::make_shared<Work>(stream_);
  NCCL_CHECK(ncclReduceScatter((const void*)send_buf, (void*)recv_buf, recv_count, dtype, ncclSum, comm_, stream_));
  w->markCompleted(true);
  return w;
}

template<typename T>
std::shared_ptr<Work> allGather(T* send_buf, T* recv_buf, size_t send_count, ncclDataType_t dtype) {
  auto w = std::make_shared<Work>(stream_);
  NCCL_CHECK(ncclAllGather((const void*)send_buf, (void*)recv_buf, send_count, dtype, comm_, stream_));
  w->markCompleted(true);
  return w;
}

template<typename T>
std::shared_ptr<Work> broadcast(T* buf, size_t count, int root, ncclDataType_t dtype) {
  auto w = std::make_shared<Work>(stream_);
  NCCL_CHECK(ncclBroadcast((void*)buf, (void*)buf, count, dtype, root, comm_, stream_));
  w->markCompleted(true);
  return w;
}

// GEMMs (row-major)
void gemm_f32_rowmajor(const float* A, const float* B, float* C,
                       int m, int n, int k, bool transA=false, bool transB=false);
                       
// inline void gemm_f32(const float* A, const float* B, float* C,
//                      int m, int n, int k){
//   gemm_f32_rowmajor(A,B,C,m,n,k,false,false);
// }

void gemm_strided_batched_f32_rowmajor(
    const float* A, const float* B, float* C,
    int m, int n, int k,
    long long strideA, long long strideB, long long strideC,
    int batchCount) const;
    
std::shared_ptr<Work> allReduce_f32(float* p, size_t n)      { return allReduce<float>(p, n, ncclFloat32); }
std::shared_ptr<Work> reduceScatter_f32(float* s, float* r, size_t rc) { return reduceScatter<float>(s, r, rc, ncclFloat32); }
std::shared_ptr<Work> allGather_f32(float* s, float* r, size_t sc)     { return allGather<float>(s, r, sc, ncclFloat32); }
std::shared_ptr<Work> broadcast_f32(float* p, size_t n, int root)      { return broadcast<float>(p, n, root, ncclFloat32); }

// utils
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

// ---- DTensor ----
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
  //OwnTensor::Tensor tensor_;
  std::vector<float> h_data_;
  float* d_data_{nullptr};
  size_t bytes_{0};
};

// allocator stats (global + dt:: aliases)
std::string allocator_stats();
void        allocator_print_stats();
namespace dt { std::string allocator_stats(); void allocator_print_stats(); }


