#include "dtensor_3d.hpp"
#include "cachingAllocator.hpp"
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
  NCCL_CHECK(ncclCommInitRank(&comm_, world_size_, id, rank_));
  CUBLAS_CHECK(cublasCreate(&cublas_), "cublasCreate");
  CUBLAS_CHECK(cublasSetStream(cublas_, stream_), "cublasSetStream");
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
    const float* A, const float* B, float* C,
    int m, int n, int k,
    long long strideA, long long strideB, long long strideC,
    int batchCount) const {
  const float alpha = 1.0f, beta = 0.0f;
  const int m_col = n, n_col = m, k_col = k;
  const int lda = n;   // rows of B_col (n x k)
  const int ldb = k;   // rows of A_col (k x m)
  const int ldc = n;   // rows of C_col (n x m)
  CUBLAS_CHECK(cublasSgemmStridedBatched(
      cublas_,
      CUBLAS_OP_N, CUBLAS_OP_N,
      m_col, n_col, k_col,
      &alpha,
      B, lda, strideB,     // B_col
      A, ldb, strideA,     // A_col
      &beta,
      C, ldc, strideC,
      batchCount), "cublasSgemmStridedBatched");
}

// ---- DTensor ----
DTensor::DTensor(int world_size, int slice_size, int rank)
: world_size_(world_size), slice_size_(slice_size), rank_(rank),
  bytes_(static_cast<size_t>(slice_size) * sizeof(float)) {
  h_data_.resize(slice_size_);
  for (int j=0;j<slice_size_;++j) h_data_[j] = float(rank_*slice_size_ + j);
  {
    CachingAllocator::DeviceGuard g(rank_);
    d_data_ = static_cast<float*>(CachingAllocator::instance().malloc(bytes_, rank_));
    if (!d_data_) throw std::runtime_error("CachingAllocator::malloc failed");
  }
  copyFromHost(h_data_.data());
}
DTensor::~DTensor(){
  if (d_data_){
    CachingAllocator::DeviceGuard g(rank_);
    CachingAllocator::instance().free(d_data_, bytes_, rank_);
    d_data_ = nullptr;
  }
}
float* DTensor::deviceData(){ return d_data_; }
std::vector<float>& DTensor::hostData(){ return h_data_; }
size_t DTensor::size() const { return static_cast<size_t>(slice_size_); }
void DTensor::copyDeviceToHost(){ CUDA_CHECK(cudaMemcpy(h_data_.data(), d_data_, bytes_, cudaMemcpyDeviceToHost)); }
void DTensor::copyFromHost(const float* src){ CUDA_CHECK(cudaMemcpy(d_data_, src, bytes_, cudaMemcpyHostToDevice)); }

// allocator stats
std::string allocator_stats(){ return CachingAllocator::instance().stats_string(); }
void allocator_print_stats(){ CachingAllocator::instance().print_stats(); }
namespace dt {
  std::string allocator_stats(){ return CachingAllocator::instance().stats_string(); }
  void allocator_print_stats(){ CachingAllocator::instance().print_stats(); }
}
