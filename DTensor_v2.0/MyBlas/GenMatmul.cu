// ============================================================================
// MYBLAS MATRIX MULTIPLICATION
// ============================================================================
// Simple wrapper around MyBlas GEMM functions
// ============================================================================

#ifdef WITH_CUDA

#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdexcept>
#include <memory>
#include <vector>
#include <algorithm>
#include <mpi.h>

#include "mycublas.h"
#include "ops/Matmul.cuh"
#include "ops/MatmulBackward.cuh"
#include "ops/LinearKernels.cuh"
#include "ops/TensorOps.h"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "device/DeviceCore.h"
#include "ops/helpers/GenMatmulUtils.h"

namespace OwnTensor {

// ============================================================================
// MYBLAS HANDLE MANAGEMENT (Thread-Safe Singleton)
// ============================================================================

class MyBlasHandleManager {
private:
   mycublasHandle_t handle_;
   bool initialized_ = false;

   MyBlasHandleManager() {
       if (mycublasCreate(&handle_) != MYCUBLAS_STATUS_SUCCESS) {
           throw std::runtime_error("Failed to create mycublas handle");
       }
       initialized_ = true;
   }

   ~MyBlasHandleManager() {
       if (initialized_) {
           mycublasDestroy(handle_);
       }
   }

public:
   static MyBlasHandleManager& getInstance() {
       static MyBlasHandleManager instance;
       return instance;
   }

   mycublasHandle_t get() { return handle_; }

   MyBlasHandleManager(const MyBlasHandleManager&) = delete;
   MyBlasHandleManager& operator=(const MyBlasHandleManager&) = delete;
};

void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream)
{
   int rank = 0;
   int mpi_initialized = 0;
   MPI_Initialized(&mpi_initialized);
   if (mpi_initialized) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   // printf("[Rank %d] MYBLAS_ENTERED: M=%d, N=%d, K=%d\n", rank, (int)A.shape().dims[A.ndim()-2], (int)output.shape().dims[output.ndim()-1], (int)A.shape().dims[A.ndim()-1]);
   
   Tensor A_cont = A.is_contiguous() ? A : A.contiguous();
   Tensor B_cont = B.is_contiguous() ? B : B.contiguous();
  
   const auto& a_shape = A_cont.shape().dims;
   const auto& b_shape = B_cont.shape().dims;
   const auto& out_shape = output.shape().dims;
  
   const auto& a_strides = A_cont.stride().strides;
   const auto& b_strides = B_cont.stride().strides;
   const auto& out_strides = output.stride().strides;
  
   size_t a_ndim = a_shape.size();
   size_t b_ndim = b_shape.size();
   size_t out_ndim = out_shape.size();
  
   int M = static_cast<int>(a_shape[a_ndim - 2]);
   int K = static_cast<int>(a_shape[a_ndim - 1]);
   int N = static_cast<int>(b_shape[b_ndim - 1]);
  
   int lda = static_cast<int>(a_strides[a_ndim - 2]);
   int ldb = static_cast<int>(b_strides[b_ndim - 2]);
   int ldc = static_cast<int>(out_strides[out_ndim - 2]);
  
   int batch_count = 1;
   for (size_t i = 0; i < out_ndim - 2; ++i) {
       batch_count *= static_cast<int>(out_shape[i]);
   }

   long long int strideA = 0;
   long long int strideB = 0;
   long long int strideC = (long long int)M * N;

   size_t size_A_matrix = (size_t)M * K;
   size_t size_T_A = A_cont.numel();
  
   if (size_T_A == (size_t)batch_count * size_A_matrix) {
       strideA = size_A_matrix;
   } else if (size_T_A == size_A_matrix) {
       strideA = 0; 
   } else {
       throw std::runtime_error("cuda_matmul: Input A has unsupported shape for Strided Batched GEMM");
   }

   size_t size_B_matrix = (size_t)K * N;
   size_t size_T_B = B_cont.numel();
  
   if (size_T_B == (size_t)batch_count * size_B_matrix) {
       strideB = size_B_matrix;
   } else if (size_T_B == size_B_matrix) {
       strideB = 0; 
   } else {
       throw std::runtime_error("cuda_matmul: Input B has unsupported shape for Strided Batched GEMM");
   }
  
   if (output.numel() != (size_t)batch_count * strideC) {
         throw std::runtime_error("cuda_matmul: Output tensor size mismatch for Strided Batched GEMM");
   }

    mycublasHandle_t handle = MyBlasHandleManager::getInstance().get();
    mycublasSetStream(handle, stream);
   
    // Ensure no previous errors are pending
    if (cudaDeviceSynchronize() != cudaSuccess) {
        // fprintf(stderr, "MyBlas ERROR: Detected pre-launch sync failure (asynchronous error from previous layer?): %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    dispatch_by_dtype(A_cont.dtype(), [&](auto dummy) {
        using T = decltype(dummy);
        const T* a_ptr = A_cont.data<T>();
        const T* b_ptr = B_cont.data<T>();
        T* out_ptr = output.data<T>();
       
        if (a_ptr == nullptr || b_ptr == nullptr || out_ptr == nullptr) {
             // fprintf(stderr, "MyBlas FATAL: NULL pointer detected! A=%p, B=%p, out=%p\n", a_ptr, b_ptr, out_ptr);
             return;
        }

        if constexpr (std::is_same<T, float>::value) {
            if (batch_count == 1) {
                mycublasSgemm(handle, M, N, K, 1.0f, (const float*)a_ptr, lda, (const float*)b_ptr, ldb, 0.0f, (float*)out_ptr, ldc);
            } else {
                mycublasSgemmStridedBatched(handle, M, N, K, 1.0f, (const float*)a_ptr, lda, strideA, (const float*)b_ptr, ldb, strideB, 0.0f, (float*)out_ptr, ldc, strideC, batch_count);
            }
        } else if constexpr (std::is_same<T, __half>::value || std::is_same<T, OwnTensor::float16_t>::value) {
            __half alpha = __float2half(1.0f);
            __half beta = __float2half(0.0f);
            if (batch_count == 1) {
                mycublasHgemm(handle, M, N, K, alpha, (const __half*)a_ptr, lda, (const __half*)b_ptr, ldb, beta, (__half*)out_ptr, ldc);
            } else {
                mycublasHgemmStridedBatchedV2(handle, M, N, K, alpha, (const __half*)a_ptr, lda, strideA, (const __half*)b_ptr, ldb, strideB, beta, (__half*)out_ptr, ldc, strideC, batch_count);
            }
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, OwnTensor::bfloat16_t>::value) {
            __nv_bfloat16 alpha = __float2bfloat16(1.0f);
            __nv_bfloat16 beta = __float2bfloat16(0.0f);
            if (batch_count == 1) {
                mycublasBgemm(handle, M, N, K, alpha, (const __nv_bfloat16*)a_ptr, lda, (const __nv_bfloat16*)b_ptr, ldb, beta, (__nv_bfloat16*)out_ptr, ldc);
            } else {
                mycublasBgemmStridedBatched(handle, M, N, K, alpha, (const __nv_bfloat16*)a_ptr, lda, strideA, (const __nv_bfloat16*)b_ptr, ldb, strideB, beta, (__nv_bfloat16*)out_ptr, ldc, strideC, batch_count);
            }
        } else if constexpr (std::is_same<T, double>::value) {
            if (batch_count == 1) {
                mycublasDgemm(handle, M, N, K, 1.0, (const double*)a_ptr, lda, (const double*)b_ptr, ldb, 0.0, (double*)out_ptr, ldc);
            } else {
                mycublasDgemmStridedBatched(handle, M, N, K, 1.0, (const double*)a_ptr, lda, strideA, (const double*)b_ptr, ldb, strideB, 0.0, (double*)out_ptr, ldc, strideC, batch_count);
            }
        } else {
            throw std::runtime_error("cuda_matmul: Unsupported dtype for batched GEMM.");
        }
    });

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // fprintf(stderr, "MyBlas ERROR: Post-launch sync failed: %s | M=%d, N=%d, K=%d, batch=%d\n", cudaGetErrorString(err), M, N, K, batch_count);
    }
}

// ============================================================================
// OVERRIDING OwnTensor::matmul to force MyBlas redirection
// ============================================================================

Tensor matmul(const Tensor& A, const Tensor& B, [[maybe_unused]]cudaStream_t stream)
{
    if (A.dtype() != B.dtype()) throw std::runtime_error("Matmul: Inputs must be of same datatypes");

    const auto& a_dims = A.shape().dims;
    const auto& b_dims = B.shape().dims;

    if (a_dims.size() < 2 || b_dims.size() < 2) throw std::runtime_error("Matmul: Both Tensors must be at least 2 Dimensional");
    if (a_dims.back() != b_dims[b_dims.size() - 2]) throw std::runtime_error("Incompatible dimensions for Matrix Multiplication");

    size_t a_ndim = a_dims.size();
    size_t b_ndim = b_dims.size();
    size_t max_ndim = std::max(a_ndim, b_ndim);

    std::vector<int64_t> output_dims(max_ndim);
    for (size_t i = 0; i < max_ndim - 2; ++i) {
        int64_t a_dim = (i >= max_ndim - a_ndim) ? a_dims[i - (max_ndim - a_ndim)] : 1;
        int64_t b_dim = (i >= max_ndim - b_ndim) ? b_dims[i - (max_ndim - b_ndim)] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) throw std::runtime_error("Incompatible batch dimensions");
        output_dims[i] = std::max(a_dim, b_dim);
    }
    output_dims[max_ndim - 2] = a_dims[a_ndim - 2];
    output_dims[max_ndim - 1] = b_dims[b_ndim - 1];

    Tensor output(Shape{output_dims}, A.dtype(), A.device(), A.requires_grad());

    if (A.device().is_cuda() && B.device().is_cuda()) {
        cudaStream_t current_stream = OwnTensor::cuda::getCurrentStream();
        cuda_matmul(A, B, output, current_stream);
    } else {
        cpu_matmul(A, B, output);
    }

    return output;
}

// ============================================================================
// OVERRIDING Backward and Linear to force MyBlas redirection
// ============================================================================

void cuda_matmul_backward(
    const Tensor& grad_output,
    const Tensor& A,
    const Tensor& B,
    Tensor& grad_A,
    Tensor& grad_B,
    cudaStream_t stream)
{
    // Diagnostics
    if (!grad_output.is_valid() || !A.is_valid() || !B.is_valid()) {
        int rank = 0;
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // fprintf(stderr, "[Rank %d] MyBlas ERROR: cuda_matmul_backward received invalid tensor(s)\n", rank);
        return;
    }

    if (grad_output.device().is_cuda()) {
        // grad_A = grad_output @ B.T
        // grad_B = A.T @ grad_output
        
        // For now, we use the simple redirection to our matmul 
        // which handles batched/strided correctly.
        // We need to handle transposes. Our `matmul` handles basic (A, B).
        // For B.T, if B is [K, N], B.T is [N, K]. 
        // If we call B.t(), it returns a view. Our `cuda_matmul` handles views?
        // Let's check if our `cuda_matmul` handles non-contiguous. 
        // YES, it calls `contiguous()` if not contiguous.
        
        if (grad_A.is_valid()) {
            Tensor Bt = B.t();
            // grad_A = grad_output @ Bt
            cuda_matmul(grad_output, Bt, grad_A, stream);
        }
        
        if (grad_B.is_valid()) {
            Tensor At = A.t();
            // grad_B = At @ grad_output
            cuda_matmul(At, grad_output, grad_B, stream);
        }
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            // fprintf(stderr, "MyBlas ERROR: Post-backward sync failed: %s\n", cudaGetErrorString(err));
        }
    }
}

void cuda_linear_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output,
    cudaStream_t stream)
{
    if (!input.is_valid() || !weight.is_valid()) {
        int rank = 0;
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // fprintf(stderr, "[Rank %d] MyBlas ERROR: cuda_linear_forward received invalid tensor(s)\n", rank);
        return;
    }

    if (input.device().is_cuda()) {
        // F.linear(x, w) = x @ w.T
        Tensor wt = weight.t();
        cuda_matmul(input, wt, output, stream);
        
        if (bias.is_valid()) {
             // We need a bias add. For now, let's use the simple operator+ 
             // but we should ideally have a kernel.
             // Since we are overriding at this level, we can just do:
             output = output + bias;
        }
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            // fprintf(stderr, "MyBlas ERROR: Post-linear sync failed: %s\n", cudaGetErrorString(err));
        }
    }
}

} // namespace OwnTensor
#endif
