#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include <random>
#include "device/DeviceCore.h"
#include "device/DeviceTransfer.h"



#include <cuda_runtime.h>
#include <curand.h>
#include "ops/helpers/ConversionKernels.cuh"

namespace OwnTensor
{
    // Helper for CUDA RNG
    void cuda_rand_uniform(float* data, size_t count, unsigned long seed, cudaStream_t stream)
    {//✨✨✨
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandSetStream(gen, stream);//✨✨✨
        curandGenerateUniform(gen, data, count);
        curandDestroyGenerator(gen);
    }

    void cuda_rand_uniform(double* data, size_t count, unsigned long seed, cudaStream_t stream)
    {//✨✨✨
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandSetStream(gen, stream);//✨✨✨
        curandGenerateUniformDouble(gen, data, count);
        curandDestroyGenerator(gen);
    }

    void cuda_rand_normal(float* data, size_t count, unsigned long seed, float sd, cudaStream_t stream)
    {//✨✨✨
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandSetStream(gen, stream);//✨✨✨
        curandGenerateNormal(gen, data, count, 0.0f, float(sd));
        curandDestroyGenerator(gen);
    }

    void cuda_rand_normal(double* data, size_t count, unsigned long seed, double sd, cudaStream_t stream)
    {//✨✨✨
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        curandSetStream(gen, stream);//✨✨✨
        curandGenerateNormalDouble(gen, data, count, 0.0, sd);
        curandDestroyGenerator(gen);
    }


    Tensor Tensor::zeros(Shape shape, TensorOptions opts)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU implementation - handles all 7 types automatically
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    tensor.fill(T(0.0f));
                });
        }
        else
        {
            // GPU implementation - optimized with cudaMemset
#ifdef WITH_CUDA
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
            cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream);//✨✨✨
#else
            throw std::runtime_error("CUDA not available");
#endif
        }
        return tensor;
    }

    Tensor Tensor::empty(Shape shape, TensorOptions opts)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU implementation - handles all 7 types automatically
            dispatch_by_dtype(opts.dtype, [&](auto [[maybe_unused]] dummy)
                {
                    // using T = decltype(dummy);
                    // tensor.fill(T(0.0f));
                });
        }
        else
        {
            // GPU implementation - optimized with cudaMemset
#ifdef WITH_CUDA
            // cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
            // cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream);//✨✨✨
#else
            throw std::runtime_error("CUDA not available");
#endif
        }
        return tensor;
    }

    Tensor Tensor::ones(Shape shape, TensorOptions opts)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU implementation - handles all 7 types automatically
            // dispatch_by_dtype(opts.dtype, [&](auto dummy) {
            //     using T = decltype(dummy);
            //     tensor.fill(T(1.0f));
            // });
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        // Special handling for bool: ones = true
                        tensor.fill(true);
                    }
                    else
                    {
                        tensor.fill(T(1.0f));
                    }
                });
        }
        else
        {
            // GPU implementation - handles all 7 types automatically
#ifdef WITH_CUDA
            [[maybe_unused]] cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        // For bool on GPU, use memset with 1
                        cudaMemset(tensor.data(), 1, tensor.numel());
                    }
                    else
                    {
                        std::vector<T> ones_data(tensor.numel(), T(1.0f));
                        cudaMemcpy(tensor.data(), ones_data.data(),
                            tensor.numel() * sizeof(T), cudaMemcpyHostToDevice);
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }
        return tensor;
    }

    Tensor Tensor::full(Shape shape, TensorOptions opts, float value)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        // For bool: any nonzero value = true
                        tensor.fill(value != 0.0f);
                    }
                    else
                    {
                        tensor.fill(static_cast<T>(value));
                    }
                });
        }
        else
        {
#ifdef WITH_CUDA
            [[maybe_unused]] cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨
            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, bool>)
                    {
                        uint8_t bool_val = (value != 0.0f) ? 1 : 0;
                        cudaMemset(tensor.data(), bool_val, tensor.numel());
                    }
                    else
                    {
                        std::vector<T> fill_data(tensor.numel(), static_cast<T>(value));
                        cudaMemcpy(tensor.data(), fill_data.data(),
                            tensor.numel() * sizeof(T), cudaMemcpyHostToDevice);
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return tensor;
    }

    template <typename U>
    Tensor Tensor::rand(Shape shape, TensorOptions opts,unsigned long seed, U lower, U upper)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU random
            //std::random_device rd;
            std::mt19937 gen(seed);

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_floating_point_v<T>)
                    {
                        std::uniform_real_distribution<T> dist(lower, upper);
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = dist(gen);
                        }
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = static_cast<T>(dist(gen));
                        }
                    }
                    else
                    {
                        throw std::runtime_error("rand only supports floating point types");
                    }
                });
        }
        else
        {
            // GPU random
#ifdef WITH_CUDA
            // std::random_device rd;
            // unsigned long seed = rd();
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, float>)
                    {
                        cuda_rand_uniform(static_cast<float*>(tensor.data()), tensor.numel(), seed, stream);
                    }
                    else if constexpr (std::is_same_v<T, double>)
                    {
                        cuda_rand_uniform(static_cast<double*>(tensor.data()), tensor.numel(), seed, stream);
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        // 1. Allocate temporary float buffer on GPU
                        float* temp_data;
                        cudaMallocAsync(&temp_data, tensor.numel() * sizeof(float), stream);
                        cuda_rand_uniform(temp_data, tensor.numel(), seed, stream);
                        convert_type_cuda(temp_data, static_cast<T*>(tensor.data()), tensor.numel(), stream);
                        cudaFreeAsync(temp_data, stream);
                    }
                    else
                    {
                        throw std::runtime_error("GPU rand only supports float/double/half/bfloat16");
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return tensor;
    }

    template <typename U>
    Tensor Tensor::randn(Shape shape, TensorOptions opts,unsigned long seed , U sd)
    {
        Tensor tensor(shape, opts);

        if (opts.device.is_cpu())
        {
            // CPU random
            //std::random_device rd;
            std::mt19937 gen(seed);

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_floating_point_v<T>)
                    {
                        std::normal_distribution<T> dist(0.0, sd);
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = dist(gen);
                        }
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        std::normal_distribution<float> dist(0.0f, float(sd));
                        T* data = static_cast<T*>(tensor.data());
                        for (size_t i = 0; i < tensor.numel(); ++i)
                        {
                            data[i] = static_cast<T>(dist(gen));
                        }
                    }
                    else
                    {
                        throw std::runtime_error("randn only supports floating point types");
                    }
                });
        }
        else
        {
            // GPU random
#ifdef WITH_CUDA
            //std::random_device rd;
            //unsigned long seed = rd();
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();//✨✨✨

            dispatch_by_dtype(opts.dtype, [&](auto dummy)
                {
                    using T = decltype(dummy);
                    if constexpr (std::is_same_v<T, float>)
                    {
                        cuda_rand_normal(static_cast<float*>(tensor.data()), tensor.numel(), seed, sd, stream);//✨✨✨
                    }
                    else if constexpr (std::is_same_v<T, double>)
                    {
                        cuda_rand_normal(static_cast<double*>(tensor.data()), tensor.numel(), seed, sd, stream);//✨✨✨
                    }
                    else if constexpr (std::is_same_v<T, OwnTensor::float16_t> || std::is_same_v<T, OwnTensor::bfloat16_t>)
                    {
                        // 1. Allocate temporary float buffer on GPU
                        float* temp_data;
                        cudaMallocAsync(&temp_data, tensor.numel() * sizeof(float), stream);
                        cuda_rand_normal(temp_data, tensor.numel(), seed, float(sd), stream);
                        convert_type_cuda(temp_data, static_cast<T*>(tensor.data()), tensor.numel(), stream);
                        cudaFreeAsync(temp_data, stream);
                    }
                    else
                    {
                        throw std::runtime_error("GPU randn only supports float/double");
                    }
                });
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return tensor;
    }

    template Tensor Tensor::rand<float>(Shape shape, TensorOptions opts,unsigned long seed, float lower, float upper);
    template Tensor Tensor::rand<double>(Shape shape, TensorOptions opts,unsigned long seed, double lower, double upper);

    template Tensor Tensor::randn<float>(Shape shape, TensorOptions opts,unsigned long seed, float sd);
    template Tensor Tensor::randn<double>(Shape shape, TensorOptions opts,unsigned long seed, double sd);

    // ======================================================================
    // multinomial — sample indices from a probability distribution
// ======================================================================
    Tensor Tensor::multinomial(const Tensor& input, int64_t num_samples,
                               bool replacement, unsigned long seed)
    {
        // If seed is 0 (default), use random_device for non-deterministic sampling.
        // Otherwise use the caller-provided seed (e.g. 42 + ddp_rank for DDP).
        if (seed == 0) {
            std::random_device rd;
            seed = rd();
        }
        // --- Validation ---
        const auto& sh = input.shape();
        if (sh.dims.size() != 1 && sh.dims.size() != 2) {
            throw std::runtime_error("multinomial: input must be 1-D or 2-D tensor");
        }

        bool is_1d   = (sh.dims.size() == 1);
        int64_t nrows = is_1d ? 1 : sh.dims[0];
        int64_t ncols = is_1d ? sh.dims[0] : sh.dims[1];

        if (num_samples <= 0) {
            throw std::runtime_error("multinomial: num_samples must be > 0");
        }

        // --- Bring input to CPU as float for sampling ---
        Tensor cpu_input = input;
        if (cpu_input.device().is_cuda()) {
            cpu_input = cpu_input.to_cpu();
        }
        if (cpu_input.dtype() != Dtype::Float32 && cpu_input.dtype() != Dtype::Float64) {
            cpu_input = cpu_input.as_type(Dtype::Float32);
        }

        // --- Allocate output on CPU (Int64) ---
        Shape out_shape = is_1d ? Shape{{num_samples}} : Shape{{nrows, num_samples}};
        Tensor output(out_shape, Dtype::Int64, DeviceIndex(Device::CPU), false);
        int64_t* out_ptr = output.data<int64_t>();

        // --- Sample per row ---
        std::mt19937 gen(seed);

        for (int64_t row = 0; row < nrows; ++row) {
            // Build weight vector for this row
            std::vector<double> weights(ncols);

            dispatch_by_dtype(cpu_input.dtype(), [&](auto dummy) {
                using T = decltype(dummy);
                if constexpr (std::is_floating_point_v<T>) {
                    const T* row_data = cpu_input.data<T>() + row * ncols;
                    for (int64_t c = 0; c < ncols; ++c) {
                        double w = static_cast<double>(row_data[c]);
                        if (w < 0.0 || !std::isfinite(w)) {
                            throw std::runtime_error(
                                "multinomial: input must be non-negative and finite");
                        }
                        weights[c] = w;
                    }
                } else {
                    throw std::runtime_error(
                        "multinomial: input must be a floating-point tensor");
                }
            });

            // Check non-zero sum
            double total = 0.0;
            int64_t non_zero_count = 0;
            for (int64_t c = 0; c < ncols; ++c) {
                total += weights[c];
                if (weights[c] > 0.0) ++non_zero_count;
            }
            if (total == 0.0) {
                throw std::runtime_error(
                    "multinomial: rows must have a non-zero sum");
            }

            if (!replacement && num_samples > non_zero_count) {
                throw std::runtime_error(
                    "multinomial: cannot sample " + std::to_string(num_samples) +
                    " without replacement from row with " +
                    std::to_string(non_zero_count) + " non-zero elements");
            }

            // Sample indices
            int64_t* row_out = out_ptr + row * num_samples;

            if (replacement) {
                // With replacement: single distribution, draw num_samples times
                std::discrete_distribution<int64_t> dist(weights.begin(), weights.end());
                for (int64_t s = 0; s < num_samples; ++s) {
                    row_out[s] = dist(gen);
                }
            } else {
                // Without replacement: draw one, zero-out, rebuild distribution
                std::vector<double> w_copy = weights;
                for (int64_t s = 0; s < num_samples; ++s) {
                    std::discrete_distribution<int64_t> dist(w_copy.begin(), w_copy.end());
                    int64_t idx = dist(gen);
                    row_out[s] = idx;
                    w_copy[idx] = 0.0;  // prevent re-sampling
                }
            }
        }

        // --- Move output to same device as input ---
        if (input.device().is_cuda()) {
#ifdef WITH_CUDA
            output = output.to_cuda(input.device().index);
#else
            throw std::runtime_error("CUDA not available");
#endif
        }

        return output;
    }

    Tensor Tensor::cat(const std::vector<Tensor>& tensors, int64_t dim) {
        if (tensors.empty()) {
            throw std::runtime_error("Tensor::cat expects a non-empty list of tensors");
        }
        
        // 1. Validate inputs and calculate output shape
        const Tensor& t0 = tensors[0];
        int64_t ndim = t0.ndim();
        
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::runtime_error("Tensor::cat: invalid dimension " + std::to_string(dim));
        }

        Shape out_shape = t0.shape();
        int64_t total_dim_size = 0;
        
        for (const auto& t : tensors) {
            if (t.ndim() != ndim) {
                throw std::runtime_error("Tensor::cat: all tensors must have same number of dimensions");
            }
            if (t.dtype() != t0.dtype()) {
                throw std::runtime_error("Tensor::cat: all tensors must have same dtype");
            }
            if (t.device().device != t0.device().device || t.device().index != t0.device().index) {
                // strict device check for now
                throw std::runtime_error("Tensor::cat: all tensors must be on same device");
            }
            
            for (int64_t i = 0; i < ndim; ++i) {
                if (i != dim && t.shape().dims[i] != t0.shape().dims[i]) {
                    throw std::runtime_error("Tensor::cat: sizes do not match except at dimension " + std::to_string(dim));
                }
            }
            total_dim_size += t.shape().dims[dim];
        }
        
        out_shape.dims[dim] = total_dim_size;
        
        // 2. Allocate output tensor
        Tensor result(out_shape, t0.dtype(), t0.device(), t0.requires_grad()); 
        
        // 3. Copy data
        // Optimization: If dim=0 and all inputs contiguous, simple memcpy.
        bool all_contiguous = true;
        for(const auto& t : tensors) if(!t.is_contiguous()) all_contiguous = false;
        
        size_t offset_bytes = 0;
        size_t element_size = t0.dtype_size(t0.dtype());
        
        if (dim == 0 && all_contiguous) {
             uint8_t* out_ptr = static_cast<uint8_t*>(result.data());
             
             for (const auto& t : tensors) {
                 size_t bytes = t.numel() * element_size;
                 device::copy_memory(out_ptr + offset_bytes, result.device().device,
                                     t.data(), t.device().device,
                                     bytes);
                 offset_bytes += bytes;
             }
        } else {
            // General case: calculate offsets.
            int64_t prob_outer_size = 1;
            for(int64_t i=0; i<dim; ++i) prob_outer_size *= out_shape.dims[i];
            
            int64_t prob_inner_size = 1;
            for(int64_t i=dim+1; i<ndim; ++i) prob_inner_size *= out_shape.dims[i];
            
            size_t inner_bytes = prob_inner_size * element_size;
            
            uint8_t* out_ptr = static_cast<uint8_t*>(result.data());
            
            // Current offset in the 'dim' dimension
            int64_t dim_offset = 0;
            
            for(const auto& t : tensors) {
                // Make contiguous if not
                Tensor t_cont = t.contiguous();
                const uint8_t* in_ptr = static_cast<const uint8_t*>(t_cont.data());
                
                int64_t dim_size = t.shape().dims[dim];
                size_t chunk_bytes = dim_size * inner_bytes;
                
                for(int64_t i=0; i<prob_outer_size; ++i) {
                     // Dest address: 
                     // row i starts at: i * (total_dim_size * inner_bytes)
                     // plus (dim_offset * inner_bytes)
                     
                     size_t out_idx_bytes = i * (total_dim_size * inner_bytes) + (dim_offset * inner_bytes);
                     size_t in_idx_bytes = i * chunk_bytes;
                     
                     device::copy_memory(out_ptr + out_idx_bytes, result.device().device,
                                         in_ptr + in_idx_bytes, t_cont.device().device,
                                         chunk_bytes);
                }
                
                dim_offset += dim_size;
            }
        }
        
        return result;
    }

}