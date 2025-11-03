#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <cublas_v2.h>   // <-- needed for cuBLAS

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>


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

    // NEW: row-major GEMM wrapper (float32)
    // Computes: C[m,n] = A[m,k] * B[k,n]
    void gemm_f32_rowmajor(const float* A, const float* B, float* C,
                           int m, int n, int k, bool transA=false, bool transB=false);

    cudaStream_t getStream() { return stream_; }
    int rank() const { return rank_; }
    int worldSize() const { return world_size_; }

private:
    int            rank_{0};
    int            world_size_{1};
    int            device_{0};
    cudaStream_t   stream_{};
    ncclComm_t     comm_{};
    cublasHandle_t cublas_{nullptr};   // <-- keep cuBLAS handle here
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
};


class DeviceOps {
public:
  DeviceOps() = default;
  ~DeviceOps() = default;

  // Set the CUDA stream used by all ops (pass your ProcessGroup's stream)
  void set_stream(std::uintptr_t stream_ptr);
  std::uintptr_t get_stream() const;

  // Elementwise / reductions
  // Y += b (B x F) += (F)
  void add_bias_inplace(std::uintptr_t Y, std::uintptr_t b, int B, int F);

  // GELU forward in-place over N elements
  void gelu_inplace(std::uintptr_t Y, int N);

  // dY *= gelu'(Y) in-place over N elements
  void gelu_backward_inplace(std::uintptr_t Y, std::uintptr_t dY, int N);

  // dZ = 2*(Z - T)/N, returns loss = mean((Z - T)^2) into *loss_dev (device scalar)
  void mse_grad_and_loss(std::uintptr_t dZ,
                         std::uintptr_t Z,
                         std::uintptr_t T,
                         int N,
                         float invN,
                         std::uintptr_t loss_dev);

  // column-wise sum over batch: colsum[f] = sum_b A[b, f]
  void reduce_cols_sum(std::uintptr_t A, std::uintptr_t colsum, int B, int F);

  // simple device-to-device copy of N floats
  void memcpy_d2d(std::uintptr_t dst, std::uintptr_t src, size_t N);

private:
  cudaStream_t stream_{nullptr};
};

// Simple AdamW optimizer that keeps m,v on device
class AdamW {
public:
  AdamW(float lr, float beta1, float beta2, float eps, float weight_decay);
  ~AdamW();

  // ensure m,v capacity
  void attach_buffers(size_t n);

  // param update: p -= lr*(mhat/(sqrt(vhat)+eps) + wd*p)
  void step(std::uintptr_t param, std::uintptr_t grad, size_t n,
            float t, std::uintptr_t stream_ptr);

  // same but wd=0 (bias)
  void step_bias(std::uintptr_t bias, std::uintptr_t grad, size_t n,
                 float t, std::uintptr_t stream_ptr);

private:
  float lr_, b1_, b2_, eps_, wd_;
  float* m_{nullptr};
  float* v_{nullptr};
  size_t cap_{0};
};
