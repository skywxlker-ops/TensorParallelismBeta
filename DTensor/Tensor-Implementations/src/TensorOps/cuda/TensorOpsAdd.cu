#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "ops/TensorOps.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"

#include <stdio.h>
#include <algorithm>

namespace OwnTensor
{
    // Helper to get number of SMs
    inline int get_num_sms() {
        int deviceId;
        cudaGetDevice(&deviceId);
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);
        return numSMs;
    }

    // Dynamic grid size calculation based on occupancy
    template <typename T_kernel>
    inline size_t get_optimal_grid_size(T_kernel kernel, int block_size, size_t total_elems) {
        int deviceId;
        cudaGetDevice(&deviceId);
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, deviceId);

        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0);

        // Aim for ~2-4 waves to hide memory latency
        const int wave_factor = 4;
        size_t optimal_grid = (size_t)num_sms * max_active_blocks * wave_factor;
        
        // Clamp to total_elems/block_size (don't launch more blocks than elements)
        size_t max_useful_blocks = (total_elems + block_size - 1) / block_size;
        return std::min(optimal_grid, max_useful_blocks > 0 ? max_useful_blocks : 1);
    }

    // Simplified metadata for collapsed dimensions
    struct SimplifiedBroadcastingMetadata {
        size_t a_shape[8];
        size_t b_shape[8];
        size_t out_shape[8];
        size_t a_strides[8];
        size_t b_strides[8];
        int ndim;
    };

    inline SimplifiedBroadcastingMetadata align_and_collapse_dims(
        const Shape& A_shape, const Stride& A_stride,
        const Shape& B_shape, const Stride& B_stride,
        const Shape& Out_shape, const Stride& Out_stride) 
    {
        size_t a_ndim = A_shape.dims.size();
        size_t b_ndim = B_shape.dims.size();
        size_t out_ndim = Out_shape.dims.size();

        size_t a_dims_padded[8], a_strides_padded[8];
        size_t b_dims_padded[8], b_strides_padded[8];
        size_t out_dims_padded[8], out_strides_padded[8];

        for (size_t i = 0; i < out_ndim; ++i) {
            int a_idx = (int)i - ((int)out_ndim - (int)a_ndim);
            if (a_idx >= 0) {
                a_dims_padded[i] = A_shape.dims[a_idx];
                a_strides_padded[i] = A_stride.strides[a_idx];
            } else {
                a_dims_padded[i] = 1;
                a_strides_padded[i] = 0;
            }

            int b_idx = (int)i - ((int)out_ndim - (int)b_ndim);
            if (b_idx >= 0) {
                b_dims_padded[i] = B_shape.dims[b_idx];
                b_strides_padded[i] = B_stride.strides[b_idx];
            } else {
                b_dims_padded[i] = 1;
                b_strides_padded[i] = 0;
            }

            out_dims_padded[i] = Out_shape.dims[i];
            out_strides_padded[i] = Out_stride.strides[i];
        }

        SimplifiedBroadcastingMetadata meta;
        int current_dim = 0;
        meta.out_shape[0] = out_dims_padded[0];
        meta.a_shape[0] = a_dims_padded[0];
        meta.b_shape[0] = b_dims_padded[0];
        meta.a_strides[0] = a_strides_padded[0];
        meta.b_strides[0] = b_strides_padded[0];

        for (size_t i = 1; i < out_ndim; ++i) {
            bool can_collapse = true;
            if (out_strides_padded[i-1] != out_strides_padded[i] * out_dims_padded[i]) can_collapse = false;
            
            auto check_mergeable = [](size_t d_prev, size_t d_curr, size_t s_prev, size_t s_curr) {
                if (d_curr == 1 && d_prev == 1) return true;
                if (d_curr > 1 && d_prev > 1 && s_prev == s_curr * d_curr) return true;
                return false; 
            };

            if (!check_mergeable(a_dims_padded[i-1], a_dims_padded[i], a_strides_padded[i-1], a_strides_padded[i])) can_collapse = false;
            if (!check_mergeable(b_dims_padded[i-1], b_dims_padded[i], b_strides_padded[i-1], b_strides_padded[i])) can_collapse = false;

            if (can_collapse) {
                meta.out_shape[current_dim] *= out_dims_padded[i];
                meta.a_shape[current_dim] *= a_dims_padded[i];
                meta.b_shape[current_dim] *= b_dims_padded[i];
                meta.a_strides[current_dim] = a_strides_padded[i];
                meta.b_strides[current_dim] = b_strides_padded[i];
            } else {
                current_dim++;
                meta.out_shape[current_dim] = out_dims_padded[i];
                meta.a_shape[current_dim] = a_dims_padded[i];
                meta.b_shape[current_dim] = b_dims_padded[i];
                meta.a_strides[current_dim] = a_strides_padded[i];
                meta.b_strides[current_dim] = b_strides_padded[i];
            }
        }
        meta.ndim = current_dim + 1;
        return meta;
    }

    template <typename T>
    __global__ __launch_bounds__(256)
    void add_kernel(const T *a, const T *b, T *output, size_t n)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
        {
            output[i] = a[i] + b[i];
        }
    }

    template <>
    __global__ __launch_bounds__(256)
    void add_kernel<__half>(const __half *a, const __half *b, __half *output, size_t n)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
        {
            output[i] = __hadd(a[i], b[i]);
        }
    }

    template <>
    __global__ __launch_bounds__(256)
    void add_kernel<__nv_bfloat16>(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *output, size_t n)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
        {
            output[i] = __hadd(a[i], b[i]);
        }
    }

    template <typename T>
    __global__ __launch_bounds__(256)
    void add_kernel_nd_broadcast(const T *a, const T *b, T *output,
                                 SimplifiedBroadcastingMetadata meta,
                                 size_t total_elems)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;

        for (size_t linear_idx = idx; linear_idx < total_elems; linear_idx += stride)
        {
            size_t temp_idx = linear_idx;
            size_t a_offset = 0;
            size_t b_offset = 0;

            for (int dim = (int)meta.ndim - 1; dim >= 0; --dim)
            {
                size_t coord = temp_idx % meta.out_shape[dim];
                temp_idx /= meta.out_shape[dim];
                
                if (meta.a_shape[dim] > 1) {
                    a_offset += coord * meta.a_strides[dim];
                }
                if (meta.b_shape[dim] > 1) {
                    b_offset += coord * meta.b_strides[dim];
                }
            }

            output[linear_idx] = a[a_offset] + b[b_offset];
        }
    }

    template <>
    __global__ __launch_bounds__(256)
    void add_kernel_nd_broadcast<__half>(const __half *a, const __half *b, __half *output,
                                         SimplifiedBroadcastingMetadata meta,
                                         size_t total_elems)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;

        for (size_t linear_idx = idx; linear_idx < total_elems; linear_idx += stride)
        {
            size_t temp_idx = linear_idx;
            size_t a_offset = 0;
            size_t b_offset = 0;

            for (int dim = (int)meta.ndim - 1; dim >= 0; --dim)
            {
                size_t coord = temp_idx % meta.out_shape[dim];
                temp_idx /= meta.out_shape[dim];
                
                if (meta.a_shape[dim] > 1) {
                    a_offset += coord * meta.a_strides[dim];
                }
                if (meta.b_shape[dim] > 1) {
                    b_offset += coord * meta.b_strides[dim];
                }
            }

            output[linear_idx] = __hadd(a[a_offset], b[b_offset]);
        }
    }

    template <>
    __global__ __launch_bounds__(256)
    void add_kernel_nd_broadcast<__nv_bfloat16>(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *output,
                                                SimplifiedBroadcastingMetadata meta,
                                                size_t total_elems)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;

        for (size_t linear_idx = idx; linear_idx < total_elems; linear_idx += stride)
        {
            size_t temp_idx = linear_idx;
            size_t a_offset = 0;
            size_t b_offset = 0;

            for (int dim = (int)meta.ndim - 1; dim >= 0; --dim)
            {
                size_t coord = temp_idx % meta.out_shape[dim];
                temp_idx /= meta.out_shape[dim];
                
                if (meta.a_shape[dim] > 1) {
                    a_offset += coord * meta.a_strides[dim];
                }
                if (meta.b_shape[dim] > 1) {
                    b_offset += coord * meta.b_strides[dim];
                }
            }

            output[linear_idx] = __hadd(a[a_offset], b[b_offset]);
        }
    }

    void cuda_add_tensor(const Tensor &A, const Tensor &B, Tensor &output, cudaStream_t stream)
    {
        size_t total_elems = output.numel();
        if (total_elems == 0) return;

        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        int block_size = 256;

        dispatch_by_dtype(A.dtype(), [&](auto dummy)
                          {
        using T = decltype(dummy);
        const T* a_ptr = A.data<T>();
        const T* b_ptr = B.data<T>();
        T* output_ptr = output.data<T>();
        
        if (!needs_broadcasting && A.is_contiguous() && B.is_contiguous()) {
            size_t grid_size = get_optimal_grid_size(add_kernel<T>, block_size, total_elems);
            add_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, output_ptr, total_elems);
        } else {
            SimplifiedBroadcastingMetadata meta = align_and_collapse_dims(
                A.shape(), A.stride(), B.shape(), B.stride(), output.shape(), output.stride()
            );

            if (meta.ndim == 1 && meta.a_shape[0] == meta.out_shape[0] && meta.b_shape[0] == meta.out_shape[0] && 
                meta.a_strides[0] == 1 && meta.b_strides[0] == 1) {
                size_t grid_size = get_optimal_grid_size(add_kernel<T>, block_size, total_elems);
                add_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, output_ptr, total_elems);
                return;
            }

            size_t grid_size = get_optimal_grid_size(add_kernel_nd_broadcast<T>, block_size, total_elems);

            add_kernel_nd_broadcast<<<grid_size, block_size, 0, stream>>>(
                a_ptr, b_ptr, output_ptr, meta, total_elems
            );
        } });
    }

    /*########################################################
                TENSOR INPLACE CUDA KERNELS
    ##########################################################*/

    template <typename T>
    __global__ __launch_bounds__(256)
    void add_inplace_kernel(T *lhs, const T *rhs, size_t n)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
        {
            lhs[i] += rhs[i];
        }
    }

    template <>
    __global__ __launch_bounds__(256)
    void add_inplace_kernel<__half>(__half *lhs, const __half *rhs, size_t n)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
        {
            lhs[i] = __hadd(lhs[i], rhs[i]);
        }
    }

    template <>
    __global__ __launch_bounds__(256)
    void add_inplace_kernel<__nv_bfloat16>(__nv_bfloat16 *lhs, const __nv_bfloat16 *rhs, size_t n)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;
        for (size_t i = idx; i < n; i += stride)
        {
            lhs[i] = __hadd(lhs[i], rhs[i]);
        }
    }

    template <typename T>
    __global__ __launch_bounds__(256)
    void add_inplace_kernel_broadcast(T *lhs, const T *rhs,
                                      SimplifiedBroadcastingMetadata meta,
                                      size_t total_elems)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;

        for (size_t i = idx; i < total_elems; i += stride)
        {
            size_t temp_idx = i;
            size_t lhs_offset = 0;
            size_t rhs_offset = 0;

            for (int dim = (int)meta.ndim - 1; dim >= 0; --dim)
            {
                size_t coord = temp_idx % meta.out_shape[dim];
                temp_idx /= meta.out_shape[dim];
                
                if (meta.a_shape[dim] > 1) {
                    lhs_offset += coord * meta.a_strides[dim];
                }
                if (meta.b_shape[dim] > 1) {
                    rhs_offset += coord * meta.b_strides[dim];
                }
            }

            lhs[lhs_offset] += rhs[rhs_offset];
        }
    }

    template <>
    __global__ __launch_bounds__(256)
    void add_inplace_kernel_broadcast<__half>(__half *lhs, const __half *rhs,
                                              SimplifiedBroadcastingMetadata meta,
                                              size_t total_elems)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;

        for (size_t i = idx; i < total_elems; i += stride)
        {
            size_t temp_idx = i;
            size_t lhs_offset = 0;
            size_t rhs_offset = 0;

            for (int dim = (int)meta.ndim - 1; dim >= 0; --dim)
            {
                size_t coord = temp_idx % meta.out_shape[dim];
                temp_idx /= meta.out_shape[dim];
                
                if (meta.a_shape[dim] > 1) {
                    lhs_offset += coord * meta.a_strides[dim];
                }
                if (meta.b_shape[dim] > 1) {
                    rhs_offset += coord * meta.b_strides[dim];
                }
            }

            lhs[lhs_offset] = __hadd(lhs[lhs_offset], rhs[rhs_offset]);
        }
    }

    template <>
    __global__ __launch_bounds__(256)
    void add_inplace_kernel_broadcast<__nv_bfloat16>(__nv_bfloat16 *lhs, const __nv_bfloat16 *rhs,
                                                     SimplifiedBroadcastingMetadata meta,
                                                     size_t total_elems)
    {
        size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
        size_t stride = (size_t)blockDim.x * gridDim.x;

        for (size_t i = idx; i < total_elems; i += stride)
        {
            size_t temp_idx = i;
            size_t lhs_offset = 0;
            size_t rhs_offset = 0;

            for (int dim = (int)meta.ndim - 1; dim >= 0; --dim)
            {
                size_t coord = temp_idx % meta.out_shape[dim];
                temp_idx /= meta.out_shape[dim];
                
                if (meta.a_shape[dim] > 1) {
                    lhs_offset += coord * meta.a_strides[dim];
                }
                if (meta.b_shape[dim] > 1) {
                    rhs_offset += coord * meta.b_strides[dim];
                }
            }

            lhs[lhs_offset] = __hadd(lhs[lhs_offset], rhs[rhs_offset]);
        }
    }

    void cuda_add_tensor_inplace(Tensor &A, const Tensor &B, cudaStream_t stream)
    {
        size_t total_elems = A.numel();
        if (total_elems == 0) return;

        bool needs_broadcasting = (A.shape().dims != B.shape().dims);
        int block_size = 256;

        dispatch_by_dtype(A.dtype(), [&](auto dummy)
                          {
            using T = decltype(dummy);
            T* a_ptr = A.data<T>();
            const T* b_ptr = B.data<T>();
            
            if (!needs_broadcasting && A.is_contiguous() && B.is_contiguous()) {
                size_t grid_size = get_optimal_grid_size(add_inplace_kernel<T>, block_size, total_elems);
                add_inplace_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, total_elems);
            } else {
                SimplifiedBroadcastingMetadata meta = align_and_collapse_dims(
                    A.shape(), A.stride(), B.shape(), B.stride(), A.shape(), A.stride()
                );

                if (meta.ndim == 1 && meta.a_shape[0] == meta.out_shape[0] && meta.b_shape[0] == meta.out_shape[0] && 
                    meta.a_strides[0] == 1 && meta.b_strides[0] == 1) {
                    size_t grid_size = get_optimal_grid_size(add_inplace_kernel<T>, block_size, total_elems);
                    add_inplace_kernel<<<grid_size, block_size, 0, stream>>>(a_ptr, b_ptr, total_elems);
                    return;
                }

                size_t grid_size = get_optimal_grid_size(add_inplace_kernel_broadcast<T>, block_size, total_elems);

            add_inplace_kernel_broadcast<<<grid_size, block_size, 0, stream>>>(
                a_ptr, b_ptr, meta, total_elems
            );
        } });
    }

}
#endif