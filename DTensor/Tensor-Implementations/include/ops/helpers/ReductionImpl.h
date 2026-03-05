#pragma once

#ifndef OWNTENSOR_REDUCTIONS_IMPL_H
#define OWNTENSOR_REDUCTIONS_IMPL_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <numeric>
#include <omp.h>

#include "core/Tensor.h" 
#include "dtype/Types.h" 
#include "ops/helpers/ReductionUtils.h" 
#include "ops/helpers/ReductionOps.h"

// #ifdef WITH_CUDA
// #include "ReductionImplGPU.h" 
// #endif


namespace OwnTensor {
namespace detail {
// Forward declarations only (no implementation here!)
template <typename T, template <typename> class OpType>
Tensor dispatch_reduction_gpu(const Tensor& input, 
                               const std::vector<int64_t>& normalized_axes, 
                               bool keepdim, cudaStream_t stream);//✨✨✨

template <typename T, template <typename> class OpType>
Tensor dispatch_index_reduction_gpu(const Tensor& input, 
                                     const std::vector<int64_t>& normalized_axes, 
                                     bool keepdim, cudaStream_t stream);//✨✨✨

template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_gpu(const Tensor& input, 
                         const std::vector<int64_t>& normalized_axes, 
                         bool keepdim, cudaStream_t stream);//✨✨✨
 template <typename T, template <typename> class VarianceOpType>
Tensor dispatch_variance_gpu(const Tensor& input, 
                             const std::vector<int64_t>& normalized_axes, 
                             bool keepdim,
                             int64_t correction,
                             cudaStream_t stream);//✨✨✨                        

constexpr size_t MAX_DIMS = 16;
// =================================================================
// HELPER: Safe isnan check for both real and complex types
// =================================================================
template<typename T>
inline bool safe_isnan(const T& val) {
    if constexpr (std::is_same_v<T, complex32_t> || 
                  std::is_same_v<T, complex64_t> || 
                  std::is_same_v<T, complex128_t>) {
        // For complex types, check if either real or imaginary parts are NaN
        // However, for reduction operations with complex types, they typically
        // aren't supported for NaN-aware operations, so return false
        return false;
    } else if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
        return std::isnan(static_cast<float>(val));
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(val);
    } else {
        // Non-floating types (integers, bool) can't be NaN
        return false;
    }
}

// =================================================================
// HELPER: Convert any type to bool (for all/any reductions)
// =================================================================
template<typename T>
inline bool to_bool_value(const T& val) {
    if constexpr (std::is_same_v<T, complex32_t>) {
        // Complex is true if either real or imaginary part is non-zero
        float r = static_cast<float>(val.real());
        float i = static_cast<float>(val.imag());
        return (r != 0.0f) || (i != 0.0f);
    } else if constexpr (std::is_same_v<T, complex64_t>) {
        return (val.real() != 0.0f) || (val.imag() != 0.0f);
    } else if constexpr (std::is_same_v<T, complex128_t>) {
        return (val.real() != 0.0) || (val.imag() != 0.0);
    } else if constexpr (std::is_same_v<T, bool>) {
        return val;
    } else {
        // For all other types (int, float, etc.), non-zero is true
        return val != T(0.0f);
    }
}

// =================================================================
// HELPER: Check if we should use double accumulation for better precision
// =================================================================
template <typename T>
constexpr bool should_use_double_accumulation() {
    return std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t> || 
           std::is_same_v<T, float4_e2m1_t> || std::is_same_v<T, float4_e2m1_2x_t>;
}

// =================================================================
// --- CORE REDUCTION KERNEL (TENSOR -> TENSOR) ---
// =================================================================

template <typename T, template <typename> class OpType, typename AccT = T>
Tensor reduce_kernel(
    const Tensor& input, 
    const std::vector<int64_t>& normalized_axes, 
    const Shape& output_shape) 
{
    using Op = OpType<T>;

    // 1. Determine output dtype
    Dtype output_dtype = input.dtype();
    if constexpr (std::is_same_v<T, bool>) {
        output_dtype = Dtype::Bool;  //  Boolean operations return Bool
    } else if constexpr (std::is_same_v<AccT, ValueIndex<T>>) {
        // Index reductions always output Int64
        output_dtype = Dtype::Int64;
    } else if constexpr (std::is_integral_v<T>) {
        // Integer reductions widen to Int64
        output_dtype = Dtype::Int64;
    } 
    
    Tensor output({output_shape}, TensorOptions().with_dtype(output_dtype).with_device(input.device()).with_req_grad(input.requires_grad()));

    // 2. Setup
    const T* input_data = input.data<T>();
    const std::vector<int64_t>& input_dims = input.shape().dims;
    const std::vector<int64_t>& input_strides = input.stride().strides;
    
    const int64_t reduced_count = calculate_reduced_count(input_dims, normalized_axes);
    
    if (reduced_count == 0 && input.numel() > 0) {
        throw std::runtime_error("Reduction error: reduced count is zero but input has " + 
                                std::to_string(input.numel()) + " elements.");
    }
    
    // Determine output C++ type
    using OutputCppT = typename std::conditional<
        std::is_same_v<AccT, ValueIndex<T>>, 
        int64_t,
        typename std::conditional<
            std::is_integral_v<T>,
            int64_t,
            T
        >::type
    >::type;
    
    OutputCppT* output_data = output.data<OutputCppT>(); 

    Op op;
    const int64_t num_slices = output.numel();
    const bool rank_preserved = input_dims.size() == output_shape.dims.size();
    
    // Calculate reduced_dims once
    std::vector<int64_t> reduced_dims;
    for(size_t dim = 0; dim < input_dims.size(); ++dim) {
        bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
        if (is_reduced) {
            reduced_dims.push_back(input_dims[dim]);
        }
    }
    const size_t ndim = output_shape.dims.size();
    
    if (input_dims.size() > MAX_DIMS) {
        throw std::runtime_error("Tensor rank exceeds maximum supported dimensions (16)");
    }
    // =================================================================
    // Use double accumulation for FP16/BF16 for maximum precision
    // =================================================================
    // Use AccT directly if it's explicitly provided, otherwise compute based on T
    using AccumulatorT = AccT;

    // =================================================================
    // Kahan summation for floating point sum operations (numerical stability)
    // =================================================================
    constexpr bool use_kahan = std::is_same_v<OpType<T>, SumOp<T>> && 
                               !std::is_same_v<AccT, ValueIndex<T>> &&
                               (std::is_floating_point_v<AccumulatorT> || 
                                std::is_same_v<AccumulatorT, double>);

    // 3. Parallel execution
    #pragma omp parallel for
    for (int64_t output_index = 0; output_index < num_slices; ++output_index) 
    {
          //  STACK-ALLOCATED BUFFER (no heap allocation)
        int64_t out_coords_buf[MAX_DIMS];
        unravel_index_stack(output_index, output_shape.dims.data(), ndim, out_coords_buf);
        if constexpr (std::is_same_v<AccT, ValueIndex<T>>) {
            // =========================================================
            // INDEX REDUCTIONS PATH (argmax, argmin, etc.)
            // =========================================================
            ValueIndex<T> accumulator = op.identity();
          
            for (int64_t i = 0; i < reduced_count; ++i) {
                // Use stack-allocated buffers for indices
                int64_t slice_coords_buf[MAX_DIMS];
                int64_t full_input_coords_buf[MAX_DIMS];
                
                unravel_index_stack(i, reduced_dims.data(), reduced_dims.size(), slice_coords_buf);
                
                int out_coord_idx = 0;
                int slice_coord_idx = 0;
                
                for (size_t dim = 0; dim < input_dims.size(); ++dim) {
                    bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
                    if (is_reduced) {
                        full_input_coords_buf[dim] = slice_coords_buf[slice_coord_idx++];
                    } else {
                        if (rank_preserved) {
                            full_input_coords_buf[dim] = out_coords_buf[dim];
                        } else {
                            full_input_coords_buf[dim] = out_coords_buf[out_coord_idx];
                        }
                        out_coord_idx++;
                    }
                }
                
                int64_t input_lin_idx = ravel_index_stack(full_input_coords_buf, input_strides.data(), input_dims.size());
                T input_value = input_data[input_lin_idx];
                ValueIndex<T> current_val_index = {input_value, i};
                accumulator = op.reduce(accumulator, current_val_index);
            }
            
            output_data[output_index] = accumulator.index;
            
        } else {
            // =========================================================
            // VALUE REDUCTIONS PATH (sum, max, mean, etc.)
            // =========================================================
            
           // std::vector<int64_t> out_coords = unravel_index(output_index, output_shape.dims);(commented out as no use)
           // out_coords_buf already computed above

            if constexpr (use_kahan) {
                // Kahan state and initialization (used only for SumOp)
                AccumulatorT kahan_sum = 0;
                AccumulatorT kahan_c = 0;
                
                // Kahan Loop
                for (int64_t i = 0; i < reduced_count; ++i) {
                    int64_t slice_coords_buf[MAX_DIMS];
                    int64_t full_input_coords_buf[MAX_DIMS];
                    
                    unravel_index_stack(i, reduced_dims.data(), reduced_dims.size(), slice_coords_buf);
                    
                    int out_coord_idx = 0;
                    int slice_coord_idx = 0;
                    
                    for (size_t dim = 0; dim < input_dims.size(); ++dim) {
                        bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
                        if (is_reduced) {
                            full_input_coords_buf[dim] = slice_coords_buf[slice_coord_idx++];
                        } else {
                            if (rank_preserved) {
                                full_input_coords_buf[dim] = out_coords_buf[dim];
                            } else {
                                full_input_coords_buf[dim] = out_coords_buf[out_coord_idx];
                            }
                            out_coord_idx++;
                        }
                    }
                    
                    int64_t input_lin_idx = ravel_index_stack(full_input_coords_buf, input_strides.data(), input_dims.size());
                    T input_value = input_data[input_lin_idx];

                    // Kahan summation for maximum numerical stability
                    AccumulatorT val_acc = static_cast<AccumulatorT>(input_value);
                    
                    // Overflow/NaN detection for numerical stability
                    if (std::isinf(kahan_sum) || std::isnan(kahan_sum)) {
                        kahan_sum += val_acc;  // Fallback to simple accumulation
                    } else {
                        AccumulatorT y = val_acc - kahan_c;
                        AccumulatorT t = kahan_sum + y;
                        kahan_c = (t - kahan_sum) - y;
                        kahan_sum = t;
                    }
                }
                
                // =================================================================
                // CRITICAL: Safe conversion back to output type (Kahan path)
                // =================================================================
                if constexpr (std::is_same_v<T, float16_t>) {
                    // FP16: overflow→inf handled by float_to_float16
                    output_data[output_index] = static_cast<OutputCppT>(
                        static_cast<T>(static_cast<float>(kahan_sum))
                    );
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    // BF16: overflow→inf handled by float_to_bfloat16
                    output_data[output_index] = static_cast<OutputCppT>(
                        static_cast<T>(static_cast<float>(kahan_sum))
                    );
                } else {
                    output_data[output_index] = static_cast<OutputCppT>(kahan_sum);
                }

            } else {
                // Initialize standard accumulator (used for all other reductions)
                AccumulatorT accumulator;
                    //  FIX: Don't cast identity for index reductions (ValueIndex type)
                if constexpr (std::is_same_v<AccT, ValueIndex<T>>) {
                    // This path should never be reached since index reductions
                    // are handled in the first if branch above
                    accumulator = op.identity();
                } else if constexpr (should_use_double_accumulation<T>()) {
                    accumulator = static_cast<double>(op.identity());
                } else if constexpr (std::is_integral_v<T>) {
                    accumulator = static_cast<int64_t>(op.identity());
                } else {
                    accumulator = op.identity();
                }

                // Standard Loop
                for (int64_t i = 0; i < reduced_count; ++i) {
                    int64_t slice_coords_buf[MAX_DIMS];
                    int64_t full_input_coords_buf[MAX_DIMS];
                    
                    unravel_index_stack(i, reduced_dims.data(), reduced_dims.size(), slice_coords_buf);
                    
                    int out_coord_idx = 0;
                    int slice_coord_idx = 0;
                    
                    for (size_t dim = 0; dim < input_dims.size(); ++dim) {
                        bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
                        if (is_reduced) {
                            full_input_coords_buf[dim] = slice_coords_buf[slice_coord_idx++];
                        } else {
                            if (rank_preserved) {
                                full_input_coords_buf[dim] = out_coords_buf[dim];
                            } else {
                                full_input_coords_buf[dim] = out_coords_buf[out_coord_idx];
                            }
                            out_coord_idx++;
                        }
                    }
                    
                    int64_t input_lin_idx = ravel_index_stack(full_input_coords_buf, input_strides.data(), input_dims.size());
                    T input_value = input_data[input_lin_idx];


                      //  FIX: Proper type conversion based on AccumulatorT
                    if constexpr (std::is_same_v<AccT, ValueIndex<T>>) {
                        // Should never reach here - handled above
                    } else if constexpr (std::is_same_v<AccT, bool>) {
                        // For AllOp/AnyOp: convert input value to bool
                        bool val_as_bool = to_bool_value(input_value);
                        accumulator = op.reduce(accumulator, val_as_bool);
                    } else {
                        AccumulatorT val_acc = static_cast<AccumulatorT>(input_value);
                        accumulator = op.reduce(accumulator, val_acc);
                    }
                }
                

                
                // =================================================================
                // CRITICAL: Safe conversion back to output type (Standard path)
                // =================================================================
                if constexpr (std::is_same_v<T, float16_t>) {
                    output_data[output_index] = static_cast<OutputCppT>(
                        static_cast<T>(static_cast<float>(accumulator))
                    );
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    output_data[output_index] = static_cast<OutputCppT>(
                        static_cast<T>(static_cast<float>(accumulator))
                    );
                } else if constexpr (std::is_same_v<OutputCppT, complex32_t>) {
                    if constexpr (std::is_same_v<AccumulatorT, complex32_t>) {
                        output_data[output_index] = accumulator;
                    } else if constexpr (std::is_same_v<AccumulatorT, complex64_t> || std::is_same_v<AccumulatorT, complex128_t>) {
                        output_data[output_index] = complex32_t(static_cast<float>(accumulator.real()), static_cast<float>(accumulator.imag()));
                    } else {
                        output_data[output_index] = complex32_t(static_cast<float>(accumulator), 0.0f);
                    }
                } else if constexpr (std::is_same_v<OutputCppT, complex64_t>) {
                    if constexpr (std::is_same_v<AccumulatorT, complex64_t>) {
                        output_data[output_index] = accumulator;
                    } else if constexpr (std::is_same_v<AccumulatorT, complex32_t>) {
                        output_data[output_index] = complex64_t(static_cast<float>(accumulator.real()), static_cast<float>(accumulator.imag()));
                    } else if constexpr (std::is_same_v<AccumulatorT, complex128_t>) {
                        output_data[output_index] = complex64_t(static_cast<float>(accumulator.real()), static_cast<float>(accumulator.imag()));
                    } else {
                        output_data[output_index] = complex64_t(static_cast<float>(accumulator), 0.0f);
                    }
                } else if constexpr (std::is_same_v<OutputCppT, complex128_t>) {
                    if constexpr (std::is_same_v<AccumulatorT, complex128_t>) {
                        output_data[output_index] = accumulator;
                    } else if constexpr (std::is_same_v<AccumulatorT, complex32_t> || std::is_same_v<AccumulatorT, complex64_t>) {
                        output_data[output_index] = complex128_t(static_cast<double>(accumulator.real()), static_cast<double>(accumulator.imag()));
                    } else {
                        output_data[output_index] = complex128_t(static_cast<double>(accumulator), 0.0);
                    }
                } else if constexpr (std::is_same_v<OutputCppT, float4_e2m1_2x_t> || std::is_same_v<OutputCppT, float4_e2m1_t>) {
                    output_data[output_index] = static_cast<OutputCppT>(static_cast<float>(accumulator));
                } else {
                    output_data[output_index] = static_cast<OutputCppT>(accumulator);
                }
            }
        }
    }

    return output;
}


// =================================================================
// --- DISPATCHER TEMPLATES WITH TYPE VALIDATION ---
// =================================================================

template <typename T, template <typename> class OpType>                                                 
Tensor dispatch_reduction(const Tensor& input, const std::vector<int64_t>& normalized_axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    constexpr bool is_all_any_op = 
        std::is_same_v<OpType<T>, AllOp<T>> ||
        std::is_same_v<OpType<T>, AnyOp<T>>;

    // constexpr bool is_complex_type =
    //     std::is_same_v<T, complex32_t> ||
    //     std::is_same_v<T, complex64_t> ||
    //     std::is_same_v<T, complex128_t>;

    // if constexpr (is_complex_type) {
    //     throw std::runtime_error(
    //         "Comparison-based reduction operations (min, max, argmin, argmax) are not supported for complex types. "
    //         "Complex numbers do not have a natural ordering. "
    //         "Got: " + get_dtype_name(input.dtype())
    //     );
    // }

    if constexpr (is_all_any_op && !std::is_same_v<T, bool>) {
        // Convert non-Bool tensor to Bool tensor (0 → false, non-zero → true)
        Tensor bool_input = input.to_bool();  // You need to implement this
        
        // Now call the Bool version
        return dispatch_reduction<bool, OpType>(bool_input, normalized_axes, keepdim, stream);
    }

    // FP4 VALIDATION: No reductions allowed
    constexpr bool is_fp4 = std::is_same_v<T, float4_e2m1_t> || std::is_same_v<T, float4_e2m1_2x_t>;
    if constexpr (is_fp4) {
        throw std::runtime_error("Reductions (sum, min, max, etc.) are not supported for FP4 types.");
    }

    //  CRITICAL: Validate that NaN operations are only used with floating point types
    constexpr bool is_nan_op = 
        std::is_same_v<OpType<T>, NanSumOp<T>> ||
        std::is_same_v<OpType<T>, NanProductOp<T>> ||
        std::is_same_v<OpType<T>, NanMinOp<T>> ||
        std::is_same_v<OpType<T>, NanMaxOp<T>> ||
        std::is_same_v<OpType<T>, NanArgMinOp<T>> ||
        std::is_same_v<OpType<T>, NanArgMaxOp<T>>;
    
    constexpr bool is_float_type = 
        std::is_same_v<T, float> || 
        std::is_same_v<T, double> ||
        std::is_same_v<T, float16_t> ||
        std::is_same_v<T, bfloat16_t>;
    
    
    
    [[maybe_unused]] constexpr bool is_comparison_op =
        std::is_same_v<OpType<T>, MinOp<T>> ||
        std::is_same_v<OpType<T>, MaxOp<T>> ||
        std::is_same_v<OpType<T>, NanMinOp<T>> ||
        std::is_same_v<OpType<T>, NanMaxOp<T>> ||
        std::is_same_v<OpType<T>, ArgMinOp<T>> ||
        std::is_same_v<OpType<T>, ArgMaxOp<T>> ||
        std::is_same_v<OpType<T>, NanArgMinOp<T>> ||
        std::is_same_v<OpType<T>, NanArgMaxOp<T>>;
    
    // Block comparison operations on complex types - they don't have a natural ordering
    
    
    // Block NaN operations on non-float types at compile time
    if constexpr (is_nan_op && !is_float_type) {
        throw std::runtime_error(
            "NaN-aware operations are only supported for floating point types (Float16, Bfloat16, Float32, Float64). "
             "Got: " + get_dtype_name(input.dtype())
        );
    }
    
#ifdef WITH_CUDA
    if (input.is_cuda()) {
        // Route to GPU implementation
        if constexpr (is_fp4) {
             throw std::runtime_error("Reductions are not supported for FP4 types.");
        } else {
             if constexpr (std::is_same_v<OpType<T>, ArgMaxOp<T>> || 
                      std::is_same_v<OpType<T>, ArgMinOp<T>> || 
                      std::is_same_v<OpType<T>, NanArgMaxOp<T>> || 
                      std::is_same_v<OpType<T>, NanArgMinOp<T>>) 
            {
                return dispatch_index_reduction_gpu<T, OpType>(input, normalized_axes, keepdim, stream);//✨✨✨
            } 
            else 
            {
                return dispatch_reduction_gpu<T, OpType>(input, normalized_axes, keepdim, stream);//✨✨✨
            }
        }
    }
#endif

    // CPU path continues as before
    if constexpr (std::is_same_v<OpType<T>, ArgMaxOp<T>> || 
                  std::is_same_v<OpType<T>, ArgMinOp<T>> || 
                  std::is_same_v<OpType<T>, NanArgMaxOp<T>> || 
                  std::is_same_v<OpType<T>, NanArgMinOp<T>>) 
    {
        Shape output_shape = detail::calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
        return reduce_kernel<T, OpType, ValueIndex<T>>(input, normalized_axes, output_shape);
    } 
    else 
    {
        // Use Op::AccT to get the correct accumulator type (e.g., bool for AllOp/AnyOp)
        using Op = OpType<T>;
        Shape output_shape = detail::calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
        return reduce_kernel<T, OpType, typename Op::AccT>(input, normalized_axes, output_shape);
    }
}

// =================================================================
// --- MEAN REDUCTION DISPATCHER WITH TYPE VALIDATION ---
// =================================================================

template <typename T, template <typename> class SumOpType>
Tensor dispatch_mean_kernel(const Tensor& input, const std::vector<int64_t>& normalized_axes, bool keepdim, cudaStream_t stream) {//✨✨✨
    // constexpr bool is_complex_type =
    //     std::is_same_v<T, complex32_t> ||
    //     std::is_same_v<T, complex64_t> ||
    //     std::is_same_v<T, complex128_t>;

    // if constexpr (is_complex_type) {
    //     throw std::runtime_error(
    //         "statistical operations are not supported for complex types for now"
            
    //         "Got: " + get_dtype_name(input.dtype())
    //     );
    // }
    //  CRITICAL: Validate NaN-aware mean operations
    constexpr bool is_nan_sum = std::is_same_v<SumOpType<T>, NanSumOp<T>>;
    constexpr bool is_float_type = 
        std::is_same_v<T, float> || 
        std::is_same_v<T, double> ||
        std::is_same_v<T, float16_t> ||
        std::is_same_v<T, bfloat16_t>;
    
    if constexpr (is_nan_sum && !is_float_type) { 
        throw std::runtime_error(
            "NaN-aware mean is only supported for floating point types (Float16, Bfloat16, Float32, Float64). "
             "Got: " + get_dtype_name(input.dtype())
        );
    }
    
#ifdef WITH_CUDA
    if (input.is_cuda()) {
        if constexpr (std::is_same_v<T, float4_e2m1_t> || std::is_same_v<T, float4_e2m1_2x_t>) {
             throw std::runtime_error("Mean reduction is not supported for FP4 types.");
        } else {
             return dispatch_mean_gpu<T, SumOpType>(input, normalized_axes, keepdim, stream);
        }
    }
#endif

    // CPU implementation continues as before...
    int64_t reduced_count = detail::calculate_reduced_count(input.shape().dims, normalized_axes);

    if (reduced_count == 0) {
        throw std::runtime_error("Cannot compute mean: reduced count is zero.");
    }

    if (input.shape().dims.size() > MAX_DIMS) {
        throw std::runtime_error("Tensor rank exceeds maximum supported dimensions (16)");
    }

    Shape output_shape = detail::calculate_output_shape(input.shape().dims, normalized_axes, keepdim);

    if constexpr (std::is_integral_v<T>) {
        // Integers output Float64
        Tensor output({output_shape}, TensorOptions().with_dtype(Dtype::Float64).with_device(input.device()).with_req_grad(input.requires_grad()));
        
        const T* input_data = input.data<T>();
        const std::vector<int64_t>& input_dims = input.shape().dims;
        const std::vector<int64_t>& input_strides = input.stride().strides;
        
        const int64_t num_slices = output.numel();
        const bool rank_preserved = input_dims.size() == output_shape.dims.size();
        
        std::vector<int64_t> reduced_dims;
        for(size_t dim = 0; dim < input_dims.size(); ++dim) {
            bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
            if (is_reduced) {
                reduced_dims.push_back(input_dims[dim]);
            }
        }
        
        double* output_data = output.data<double>();
        //SumOpType<T> op;
        
        const size_t ndim = output_shape.dims.size();
        #pragma omp parallel for
        for (int64_t output_index = 0; output_index < num_slices; ++output_index) {
            int64_t accumulator = 0;
            
            int64_t out_coords_buf[MAX_DIMS];
            unravel_index_stack(output_index, output_shape.dims.data(), ndim, out_coords_buf);
            
            for (int64_t i = 0; i < reduced_count; ++i) {
                int64_t slice_coords_buf[MAX_DIMS];
                int64_t full_input_coords_buf[MAX_DIMS];
                
                unravel_index_stack(i, reduced_dims.data(), reduced_dims.size(), slice_coords_buf);
                
                int out_coord_idx = 0;
                int slice_coord_idx = 0;
                
                for (size_t dim = 0; dim < input_dims.size(); ++dim) {
                    bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
                    if (is_reduced) {
                        full_input_coords_buf[dim] = slice_coords_buf[slice_coord_idx++];
                    } else {
                        if (rank_preserved) {
                            full_input_coords_buf[dim] = out_coords_buf[dim];
                        } else {
                            full_input_coords_buf[dim] = out_coords_buf[out_coord_idx];
                        }
                        out_coord_idx++;
                    }
                }
                
                int64_t input_lin_idx = ravel_index_stack(full_input_coords_buf, input_strides.data(), input_dims.size());
                T input_value = input_data[input_lin_idx];
                
                accumulator += input_value;
            }
            
            output_data[output_index] = static_cast<double>(accumulator) / static_cast<double>(reduced_count);
        }
        
        return output;
        
        } else {
    // Floating point: use double accumulation for FP16/BF16
    using AccT = typename std::conditional<
        should_use_double_accumulation<T>(),
        double,  
        T        
    >::type;
    
    Tensor sum_result = reduce_kernel<T, SumOpType, AccT>(input, normalized_axes, output_shape);
    
    using SumT = typename std::conditional<
        should_use_double_accumulation<T>(),
        double,  
        T        
    >::type;
    
    T* sum_data = sum_result.data<T>();
    
    //  FIX: For NaN-aware mean, count only non-NaN values
    [[maybe_unused]] SumT divisor;
    if constexpr (is_nan_sum) {
        // Count non-NaN values in the input tensor
        const T* input_data = input.data<T>();
        const std::vector<int64_t>& input_dims = input.shape().dims;
        const std::vector<int64_t>& input_strides = input.stride().strides;
        
        std::vector<int64_t> reduced_dims;
        for(size_t dim = 0; dim < input_dims.size(); ++dim) {
            bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
            if (is_reduced) {
                reduced_dims.push_back(input_dims[dim]);
            }
        }
        
        const bool rank_preserved = input_dims.size() == output_shape.dims.size();
        const int64_t num_slices = sum_result.numel();
        
        // Create a tensor to store valid counts for each output position
        std::vector<int64_t> valid_counts(num_slices, 0);
        
        const size_t ndim = output_shape.dims.size();
        #pragma omp parallel for
        for (int64_t output_index = 0; output_index < num_slices; ++output_index) {
            int64_t valid_count = 0;
            int64_t out_coords_buf[MAX_DIMS];
            unravel_index_stack(output_index, output_shape.dims.data(), ndim, out_coords_buf);
            
            for (int64_t i = 0; i < reduced_count; ++i) {
                int64_t slice_coords_buf[MAX_DIMS];
                int64_t full_input_coords_buf[MAX_DIMS];
                
                unravel_index_stack(i, reduced_dims.data(), reduced_dims.size(), slice_coords_buf);
                
                int out_coord_idx = 0;
                int slice_coord_idx = 0;
                
                for (size_t dim = 0; dim < input_dims.size(); ++dim) {
                    bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) != normalized_axes.end();
                    if (is_reduced) {
                        full_input_coords_buf[dim] = slice_coords_buf[slice_coord_idx++];
                    } else {
                        if (rank_preserved) {
                            full_input_coords_buf[dim] = out_coords_buf[dim];
                        } else {
                            full_input_coords_buf[dim] = out_coords_buf[out_coord_idx];
                        }
                        out_coord_idx++;
                    }
                }
                
                int64_t input_lin_idx = ravel_index_stack(full_input_coords_buf, input_strides.data(), input_dims.size());
                T input_value = input_data[input_lin_idx];
                
                
                // Check if value is not NaN
                if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                    if (!std::isnan(static_cast<float>(input_value))) {
                        valid_count++;
                    }
                } else {
                    if (!safe_isnan(input_value)) {
                        valid_count++;
                    }
                }
            }
            
            valid_counts[output_index] = valid_count;
        }
        
        // Now divide each sum by its corresponding valid count
        #pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(sum_result.numel()); ++i) {
            if (valid_counts[i] > 0) {
                SumT val = static_cast<SumT>(sum_data[i]);
                // Handle complex types specially to avoid constructor ambiguity
                if constexpr (std::is_same_v<SumT, complex32_t> || std::is_same_v<SumT, complex64_t> || std::is_same_v<SumT, complex128_t>) {
                    val /= SumT(static_cast<double>(valid_counts[i]), 0.0);
                } else if constexpr (std::is_same_v<SumT, float4_e2m1_2x_t> || std::is_same_v<SumT, float4_e2m1_t>) {
                    val /= static_cast<SumT>(static_cast<float>(valid_counts[i]));
                } else {
                    val /= static_cast<SumT>(valid_counts[i]);  // Divide by non-NaN count
                }
                
                if constexpr (std::is_same_v<T, float16_t>) {
                    sum_data[i] = static_cast<T>(static_cast<float>(val));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    sum_data[i] = static_cast<T>(static_cast<float>(val));
                } else {
                    sum_data[i] = val;
                }
            } else {
                // All values were NaN - result is NaN
                if constexpr (std::is_same_v<T, float16_t>) {
                    sum_data[i] = static_cast<T>(std::nanf(""));
                } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                } else {
                    sum_data[i] = std::numeric_limits<T>::quiet_NaN();
                }
            }
        }
        
    } else {
        // Regular mean: divide by total reduced count
        SumT divisor;
        if constexpr (std::is_same_v<SumT, complex32_t> || std::is_same_v<SumT, complex64_t> || std::is_same_v<SumT, complex128_t>) {
            divisor = SumT(static_cast<double>(reduced_count), 0.0);
        } else if constexpr (std::is_same_v<SumT, float4_e2m1_2x_t> || std::is_same_v<SumT, float4_e2m1_t>) {
            divisor = static_cast<SumT>(static_cast<float>(reduced_count));
        } else {
            divisor = static_cast<SumT>(reduced_count);
        }
        
        #pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(sum_result.numel()); ++i) {
            SumT val = static_cast<SumT>(sum_data[i]);
            val /= divisor;

            if constexpr (std::is_same_v<T, float16_t>) {
                sum_data[i] = static_cast<T>(static_cast<float>(val));
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                sum_data[i] = static_cast<T>(static_cast<float>(val));
            } else {
                sum_data[i] = val;
            }
        }
    }
    
    return sum_result;


        // Final result must be cast back to the original Tensor type (T) if AccT was double.
        // The reduce_kernel returns a Tensor<T> or Tensor<double>, but the output Dtype is T.
        // The previous code had a bug here.
        // We ensure the output Tensor's data type matches the original T
        // if constexpr (should_use_double_accumulation<T>()) {
        //     Tensor final_output({output_shape}, TensorOptions().with_dtype(input.dtype()).with_req_grad(false));
        //     T* final_output_data = final_output.data<T>();
            
        //     #pragma omp parallel for
        //     for (int64_t i = 0; i < static_cast<int64_t>(sum_result.numel()); ++i) {
        //         // Safe conversion from SumT (double) back to output type (T)
        //         if constexpr (std::is_same_v<T, float16_t>) {
        //             final_output_data[i] = static_cast<T>(static_cast<float>(sum_data[i]));
        //         } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        //             final_output_data[i] = static_cast<T>(static_cast<float>(sum_data[i]));
        //         } else {
        //             final_output_data[i] = static_cast<T>(sum_data[i]);
        //         }
        //     }
        //     return final_output;
        // } else {
        //     return sum_result;
        // }
    }
}

//----------------------------------------------------------------------
// VARIANCE REDUCTION DISPATCHER (Two-pass algorithm)
//----------------------------------------------------------------------

template <typename T, template <typename> class VarianceOpType>
Tensor dispatch_variance_kernel(const Tensor& input, 
                                const std::vector<int64_t>& normalized_axes, 
                                bool keepdim,
                                int64_t correction, cudaStream_t stream) {
    // Determine if this is NaN-aware variance
    if constexpr (std::is_same_v<T, bool>) {
        throw std::runtime_error(
             "reduce_var: Bool dtype not supported for statistical operations."
        );
    }
    // constexpr bool is_complex_type =
    //     std::is_same_v<T, complex32_t> ||
    //     std::is_same_v<T, complex64_t> ||
    //     std::is_same_v<T, complex128_t>;
    // if constexpr (is_complex_type) {
    //     throw std::runtime_error(
    //         "statistical operations are not supported for complex types for now"
            
    //         "Got: " + get_dtype_name(input.dtype())
    //     );
    // }
    constexpr bool is_nan_aware = std::is_same_v<VarianceOpType<T>, NanVarianceOp<T>>;
    
    constexpr bool is_float_type = 
        std::is_same_v<T, float> || 
        std::is_same_v<T, double> ||
        std::is_same_v<T, float16_t> ||
        std::is_same_v<T, bfloat16_t>;
    
    if constexpr (is_nan_aware && !is_float_type) {
        throw std::runtime_error(
            "NaN-aware variance is only supported for floating point types (Float16, Bfloat16, Float32, Float64). "
             "Got: " + get_dtype_name(input.dtype())
        );
    }
#ifdef WITH_CUDA
    if (input.is_cuda()) {
        if constexpr (std::is_same_v<T, float4_e2m1_t> || std::is_same_v<T, float4_e2m1_2x_t>) {
             throw std::runtime_error("Variance reduction is not supported for FP4 types.");
        } else {
            return dispatch_variance_gpu<T, VarianceOpType>(
                input, normalized_axes, keepdim, correction, stream);
        }
    }
#endif
    
    //  STEP 1: Compute mean with keepdim=true (required for broadcasting)
    Tensor mean_tensor = is_nan_aware 
        ? dispatch_mean_kernel<T, NanSumOp>(input, normalized_axes, true, stream)
        : dispatch_mean_kernel<T, SumOp>(input, normalized_axes, true, stream);
    
    //  STEP 2: Calculate output shape and metadata
    Shape output_shape = calculate_output_shape(input.shape().dims, normalized_axes, keepdim);
    int64_t reduced_count = calculate_reduced_count(input.shape().dims, normalized_axes);

    if (reduced_count == 0) {
        throw std::runtime_error("Cannot compute variance: reduced count is zero.");
    }

    if (input.shape().dims.size() > MAX_DIMS) {
        throw std::runtime_error("Tensor rank exceeds maximum supported dimensions (16)");
    }
    
    // Determine output dtype
    Dtype output_dtype;
    if constexpr (std::is_integral_v<T>) {
        output_dtype = Dtype::Float64;
    } else {
        output_dtype = input.dtype();
    }
    
    Tensor output({output_shape}, TensorOptions()
        .with_dtype(output_dtype)
        .with_device(input.device())
        .with_req_grad(input.requires_grad()));
    
    //  STEP 3: Prepare data pointers
    const T* input_data = input.data<T>();
    
    //  CRITICAL FIX: For integers, mean is stored as double
    using MeanCppT = typename std::conditional<
        std::is_integral_v<T>,
        double,
        T
    >::type;
    
    const MeanCppT* mean_data = mean_tensor.data<MeanCppT>();
    
    using AccT = typename std::conditional<
        should_use_double_accumulation<T>(),
        double,
        typename std::conditional<
            std::is_integral_v<T>,
            double,
            T
        >::type
    >::type;
    
    using OutputT = typename std::conditional<
        std::is_integral_v<T>,
        double,
        T
    >::type;
    
    OutputT* output_data = output.data<OutputT>();
    
    const std::vector<int64_t>& input_dims = input.shape().dims;
    const std::vector<int64_t>& input_strides = input.stride().strides;
    const int64_t num_slices = output.numel();
    const bool rank_preserved = input_dims.size() == output_shape.dims.size();
    
    // Get mean tensor strides for indexing
    const std::vector<int64_t>& mean_strides = mean_tensor.stride().strides;
    
    // Calculate reduced_dims
    std::vector<int64_t> reduced_dims;
    for(size_t dim = 0; dim < input_dims.size(); ++dim) {
        bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) 
                         != normalized_axes.end();
        if (is_reduced) {
            reduced_dims.push_back(input_dims[dim]);
        }
    }
    
    const size_t ndim = output_shape.dims.size();
    #pragma omp parallel for
    for (int64_t output_index = 0; output_index < num_slices; ++output_index) {
        AccT accumulator = AccT(0.0f);
        int64_t valid_count = 0;  // Only used for NaN-aware variance
        
        // Calculate output coordinates
        int64_t out_coords_buf[MAX_DIMS];
        unravel_index_stack(output_index, output_shape.dims.data(), ndim, out_coords_buf);
        
        //  Map output coordinates to mean tensor coordinates
        // Since mean was computed with keepdim=true, it has same rank as input
        int64_t mean_coords_buf[MAX_DIMS];
        int out_coord_idx = 0;
        
        for (size_t dim = 0; dim < input_dims.size(); ++dim) {
            bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) 
                             != normalized_axes.end();
            if (is_reduced) {
                mean_coords_buf[dim] = 0;  // Mean tensor has size 1 in reduced dimensions
            } else {
                if (rank_preserved) {
                    mean_coords_buf[dim] = out_coords_buf[dim];
                } else {
                    mean_coords_buf[dim] = out_coords_buf[out_coord_idx];
                }
                out_coord_idx++;
            }
        }
        
        // Get the pre-computed mean value for this slice
        int64_t mean_lin_idx = ravel_index_stack(mean_coords_buf, mean_strides.data(), input_dims.size());
        AccT mean_val = static_cast<AccT>(mean_data[mean_lin_idx]);
        
        // Check if mean is NaN
        bool mean_is_nan = false;
        if constexpr (std::is_floating_point_v<T> || 
                      std::is_same_v<T, float16_t> || 
                      std::is_same_v<T, bfloat16_t>) {
            if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                mean_is_nan = std::isnan(static_cast<float>(mean_val));
            } else {
                mean_is_nan = std::isnan(mean_val);
            }
        }
        
        //  Accumulate squared deviations
        for (int64_t i = 0; i < reduced_count; ++i) {
            int64_t slice_coords_buf[MAX_DIMS];
            int64_t full_input_coords_buf[MAX_DIMS];
            
            unravel_index_stack(i, reduced_dims.data(), reduced_dims.size(), slice_coords_buf);
            
            out_coord_idx = 0;
            int slice_coord_idx = 0;
            
            for (size_t dim = 0; dim < input_dims.size(); ++dim) {
                bool is_reduced = std::find(normalized_axes.begin(), normalized_axes.end(), (int64_t)dim) 
                                 != normalized_axes.end();
                if (is_reduced) {
                    full_input_coords_buf[dim] = slice_coords_buf[slice_coord_idx++];
                } else {
                    if (rank_preserved) {
                        full_input_coords_buf[dim] = out_coords_buf[dim];
                    } else {
                        full_input_coords_buf[dim] = out_coords_buf[out_coord_idx];
                    }
                    out_coord_idx++;
                }
            }
            
            int64_t input_lin_idx = ravel_index_stack(full_input_coords_buf, input_strides.data(), input_dims.size());
            T input_value = input_data[input_lin_idx];
            
            // Check if value is NaN
            bool is_nan = false;
            if constexpr (std::is_floating_point_v<T> || 
                          std::is_same_v<T, float16_t> || 
                          std::is_same_v<T, bfloat16_t>) {
                if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
                    is_nan = std::isnan(static_cast<float>(input_value));
                } else {
                    is_nan = std::isnan(input_value);
                }
            }
            
            if constexpr (is_nan_aware) {
                //  NaN-aware: Skip NaN values and count valid ones
                if (!is_nan) {
                    AccT val_acc = static_cast<AccT>(input_value);
                    AccT diff = val_acc - mean_val;
                    accumulator += diff * diff;
                    valid_count++;  // Only increment for valid values
                }
            } else {
                //  Regular variance: Propagate NaN immediately
                if (is_nan || mean_is_nan) {
                    accumulator = std::numeric_limits<AccT>::quiet_NaN();
                    break;  // Early exit on NaN
                }
                
                AccT val_acc = static_cast<AccT>(input_value);
                AccT diff = val_acc - mean_val;
                accumulator += diff * diff;
                // Don't increment valid_count here!
            }
        }
        
        //  STEP 5: Compute divisor and variance
        int64_t divisor;
        if constexpr (is_nan_aware) {
            divisor = valid_count - correction;  // Use counted valid values
        } else {
            divisor = reduced_count - correction;  // Use total count
        }
        
        // Compute final variance
        AccT variance;
        if (safe_isnan(accumulator)) {
            variance = accumulator;
        } else if (divisor <= 0) {
            variance = std::numeric_limits<AccT>::quiet_NaN();
        } else {
            if constexpr (std::is_same_v<AccT, complex32_t> || std::is_same_v<AccT, complex64_t> || std::is_same_v<AccT, complex128_t>) {
                variance = accumulator / AccT(static_cast<double>(divisor), 0.0);
            } else if constexpr (std::is_same_v<AccT, float4_e2m1_2x_t> || std::is_same_v<AccT, float4_e2m1_t>) {
                variance = accumulator / static_cast<AccT>(static_cast<float>(divisor));
            } else {
                variance = accumulator / static_cast<AccT>(divisor);
            }
        }
        
        //  STEP 6: Convert back to output type
        if constexpr (std::is_same_v<T, float16_t>) {
            output_data[output_index] = static_cast<OutputT>(
                static_cast<T>(static_cast<float>(variance))
            );
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            output_data[output_index] = static_cast<OutputT>(
                static_cast<T>(static_cast<float>(variance))
            );
        } else {
            output_data[output_index] = static_cast<OutputT>(variance);
        }
    }
    
    return output;
}
} // namespace detail
} // namespace OwnTensor
#endif // OWNTENSOR_REDUCTIONS_IMPL_H

