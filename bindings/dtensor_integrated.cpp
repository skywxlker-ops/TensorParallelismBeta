#include "dtensor_integrated.hpp"
#include "cachingAllocator.hpp"
#include "TensorLib.h"
#include "core/Tensor.h"
#include "core/Views/ViewUtils.h"
#include "ops/Matmul.cuh"

using namespace OwnTensor;

// #ifdef KT_CALL_MATMUL
// #undef KT_CALL_MATMUL
// #endif
// #define KT_CALL_MATMUL GenMatmul

using dt::CachingAllocator;

// ---- Work ----
Work::Work(cudaStream_t stream) : stream_(stream) {

  CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));

}

Work::~Work(){ CUDA_CHECK(cudaEventDestroy(event_)); }
void Work::markCompleted(bool success){ success_ = success; completed_ = true; CUDA_CHECK(cudaEventRecord(event_, stream_)); }
bool Work::wait(){ if(!completed_) return false; CUDA_CHECK(cudaEventSynchronize(event_)); return success_; }


// ---- ProcessGroup ----
ProcessGroup::ProcessGroup(int rank, int world_size, int device, const ncclUniqueId &id)
: rank_(rank), world_size_(world_size), device_(device){
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaStreamCreate(&stream_));

  // 2) CUDA/NCCL sanity prints (BEFORE NCCL init)
  int drv = 0, rtv = 0;
  cudaDriverGetVersion(&drv);
  cudaRuntimeGetVersion(&rtv);
  fprintf(stderr, "[rank %d] CUDA versions: driver=%d runtime=%d\n", rank_, drv, rtv);

  cudaError_t e1 = cudaFree(0);
  fprintf(stderr, "[rank %d] cudaFree(0) -> %s\n", rank_, cudaGetErrorString(e1));

  int ncclVer = 0;
  ncclGetVersion(&ncclVer);
  fprintf(stderr, "[rank %d] NCCL version: %d\n", rank_, ncclVer);

  // 3) cuBLAS on the same stream
  CUBLAS_CHECK(cublasCreate(&cublas_), "cublasCreate");
  CUBLAS_CHECK(cublasSetStream(cublas_, stream_), "cublasSetStream");

  // 4) NCCL init (print exact failure)
  ncclResult_t st = ncclCommInitRank(&comm_, world_size_, id, rank_);
  if (st != ncclSuccess) {
    fprintf(stderr, "[rank %d] ncclCommInitRank -> %s\n", rank_, ncclGetErrorString(st));
    throw std::runtime_error("ncclCommInitRank failed");
  }

}

ProcessGroup::~ProcessGroup(){
  if (cublas_) cublasDestroy(cublas_);
  if (comm_)   ncclCommDestroy(comm_);
  if (stream_) CUDA_CHECK(cudaStreamDestroy(stream_));
}

void ProcessGroup::gemm_f32_rowmajor(const float* A, const float* B, float* C,
                                       int m, int n, int k, bool transA, bool transB){
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = (opA == CUBLAS_OP_N) ? k : m;
  int ldb = (opB == CUBLAS_OP_N) ? n : k;
  int ldc = n;
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(
      cublas_, opB, opA,  // swap for col-major
      n, m, k,
      &alpha,
      B, ldb,
      A, lda,
      &beta,
      C, ldc), "cublasSgemm");
}

void ProcessGroup::gemm_strided_batched_f32_rowmajor(
    const float* A_ptr, const float* B_ptr, float* C_ptr,
    int m, int n, int k,
    long long strideA, long long strideB, long long strideC,
    int batchCount) const
{
  using namespace OwnTensor;

  // Shapes in elements (row-major)
  Shape a_shape; a_shape.dims = { (int64_t)batchCount, (int64_t)m, (int64_t)k };
  Shape b_shape; b_shape.dims = { (int64_t)((strideB==0)?1:batchCount), (int64_t)k, (int64_t)n };
  Shape c_shape; c_shape.dims = { (int64_t)batchCount, (int64_t)m, (int64_t)n };

  // Strides in elements (0 batch-stride = broadcast B)
  Stride a_stride; a_stride.strides = { (int64_t)strideA, (int64_t)k, 1 };
  Stride b_stride; b_stride.strides = { (int64_t)(strideB==0 ? 0 : strideB), (int64_t)n, 1 };
  Stride c_stride; c_stride.strides = { (int64_t)strideC, (int64_t)n, 1 };

  // Wrap raw device memory as NON-OWNING views
  auto noop = [](uint8_t*){};
  std::shared_ptr<uint8_t[]> A_sh(reinterpret_cast<uint8_t*>(const_cast<float*>(A_ptr)), noop);
  std::shared_ptr<uint8_t[]> B_sh(reinterpret_cast<uint8_t*>(const_cast<float*>(B_ptr)), noop);
  std::shared_ptr<uint8_t[]> C_sh(reinterpret_cast<uint8_t*>(C_ptr),                     noop);

  Dtype dt = Dtype::Float32;
  DeviceIndex dev(Device::CUDA, device_);
  bool requires_grad = false;

  Tensor A = Tensor::from_external(A_sh, a_shape, a_stride, 0, dt, dev, requires_grad);
  Tensor B = Tensor::from_external(B_sh, b_shape, b_stride, 0, dt, dev, requires_grad);
  Tensor C = Tensor::from_external(C_sh, c_shape, c_stride, 0, dt, dev, requires_grad);

  // Call the repo’s host API (declared in ops/Matmul.cuh, defined in GenMatmul.cu)
  OwnTensor::cuda_matmul(A, B, C);
}

  

// void ProcessGroup::gemm_strided_batched_f32_rowmajor(
//     const float* A_ptr, const float* B_ptr, float* C_ptr,
//     int m, int n, int k,
//     long long strideA, long long strideB, long long strideC,
//     int batchCount) const
// {
//   // using namespace OwnTensor;

//   // Shape a_shape; a_shape.dims = { (int64_t)batchCount, (int64_t)m, (int64_t)k };
//   // Shape b_shape; b_shape.dims = { (int64_t)((strideB==0)?1:batchCount), (int64_t)k, (int64_t)n };
//   // Shape c_shape; c_shape.dims = { (int64_t)batchCount, (int64_t)m, (int64_t)n };

//   // Stride a_stride; a_stride.strides = { (int64_t)strideA, (int64_t)k, 1 };
//   // Stride b_stride; b_stride.strides = { (int64_t)(strideB==0 ? 0 : strideB), (int64_t)n, 1 };
//   // Stride c_stride; c_stride.strides = { (int64_t)strideC, (int64_t)n, 1 };

//   // auto noop = [](uint8_t*){};
//   // std::shared_ptr<uint8_t[]> A_sh(reinterpret_cast<uint8_t*>(const_cast<float*>(A_ptr)), noop);
//   // std::shared_ptr<uint8_t[]> B_sh(reinterpret_cast<uint8_t*>(const_cast<float*>(B_ptr)), noop);
//   // std::shared_ptr<uint8_t[]> C_sh(reinterpret_cast<uint8_t*>(C_ptr),                     noop);

//   // Dtype dt = Dtype::Float32;
//   // DeviceIndex dev(Device::CUDA, device_);
//   // bool requires_grad = false;

//   // Tensor A = Tensor::from_external(A_sh, a_shape, a_stride, /*offset=*/0, dt, dev, requires_grad);
//   // Tensor B = Tensor::from_external(B_sh, b_shape, b_stride, /*offset=*/0, dt, dev, requires_grad);
//   // Tensor C = Tensor::from_external(C_sh, c_shape, c_stride, /*offset=*/0, dt, dev, requires_grad);

//   // KT_CALL_MATMUL(A, B, C);

//   cudaMatmul(A, B, C);
// }

 
// ---- DTensor ----
DTensor::DTensor(int world_size, int slice_size, int rank)
: world_size_(world_size), slice_size_(slice_size), rank_(rank),
  bytes_(static_cast<size_t>(slice_size) * sizeof(float)) {
  h_data_.resize(slice_size_);
  for (int j=0;j<slice_size_;++j) h_data_[j] = float(rank_*slice_size_ + j);
  // {
  //   OwnTensor::TensorOptions opts{ OwnTensor::Dtype::Float32, OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_) };
  //   OwnTensor::Shape s({ static_cast<size_t>(slice_size_) });
  //   tensor_ = OwnTensor::Tensor::zeros(s, opts);
  // }
  // copyFromHost(h_data_.data());
  // Allocate device memory from the GPU caching allocator
   CUDA_CHECK(cudaSetDevice(rank_));
   d_data_ = static_cast<float*>(CachingAllocator::instance().malloc(bytes_, rank_));
   if (!d_data_) throw std::runtime_error("DTensor device allocation failed");
 
   // Initialize device memory from host using device::copy_memory
   copyFromHost(h_data_.data());
}

DTensor::~DTensor(){
  //h_data_.clear();
  // free device memory back to caching allocator
   if (d_data_) {
     // ensure correct device when freeing
     CUDA_CHECK(cudaSetDevice(rank_));
     CachingAllocator::instance().free(d_data_, bytes_, rank_);
     d_data_ = nullptr;
   }
   h_data_.clear();
}
//float* DTensor::deviceData(){ return reinterpret_cast<float*>(tensor_.data_ptr()); }

float* DTensor::deviceData(){ return d_data_; }
std::vector<float>& DTensor::hostData(){ return h_data_; }
size_t DTensor::size() const { return static_cast<size_t>(slice_size_); }
// void DTensor::copyDeviceToHost(){ void* dev_ptr = tensor_.data_ptr();
//   OwnTensor::device::copy_memory(h_data_.data(), OwnTensor::Device::CPU,
//                                  dev_ptr,                 OwnTensor::Device::CUDA,
//                                  bytes_); }
// void DTensor::copyFromHost(const float* src){ void* dev_ptr = tensor_.data_ptr();
//   OwnTensor::device::copy_memory(dev_ptr,                 OwnTensor::Device::CUDA,
//                                  src,                    OwnTensor::Device::CPU,
//                                  bytes_); }
void DTensor::copyDeviceToHost(){
  if (!d_data_) throw std::runtime_error("device buffer is null");
  OwnTensor::device::copy_memory(h_data_.data(),        OwnTensor::Device::CPU,
                                 d_data_,               OwnTensor::Device::CUDA,
                                 bytes_);
}
void DTensor::copyFromHost(const float* src){
  if (!d_data_) throw std::runtime_error("device buffer is null");
  OwnTensor::device::copy_memory(d_data_,               OwnTensor::Device::CUDA,
                                 src,                   OwnTensor::Device::CPU,
                                 bytes_);
}

// allocator stats
std::string allocator_stats(){ return CachingAllocator::instance().stats_string(); }
void allocator_print_stats(){ CachingAllocator::instance().print_stats(); }
namespace dt {
  std::string allocator_stats(){ return CachingAllocator::instance().stats_string(); }
  void allocator_print_stats(){ CachingAllocator::instance().print_stats(); }
}



