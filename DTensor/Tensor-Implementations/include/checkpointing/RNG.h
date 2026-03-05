#pragma once

#include <random>
#include <vector>
#include <memory>

#ifdef WITH_CUDA
#include <curand.h>
#endif

namespace OwnTensor {

/**
 * @brief Container for RNG states (CPU and GPU).
 */
struct RNGState {
    // Fixed size state for std::mt19937 to avoid heap allocation
    // On this platform, sizeof(std::mt19937) is 5000 bytes, likely due to 64-bit uint_fast32_t.
    // We use a safe upper bound.
    uint32_t cpu_state_data[2000]; 
#ifdef WITH_CUDA
    unsigned long long gpu_seed;
    unsigned long long gpu_offset;
#endif
};

/**
 * @brief Global RNG management.
 */
class RNG {
public:
    /**
     * @brief Get the thread-local CPU generator.
     */
    static std::mt19937& get_cpu_generator();

    /**
     * @brief Get the current RNG state for the current thread.
     */
    static RNGState get_state();

    /**
     * @brief Restore RNG state for the current thread.
     */
    static void set_state(const RNGState& state);

    /**
     * @brief Set the seed for all generators in the current thread.
     */
    static void set_seed(unsigned long seed);

#ifdef WITH_CUDA
    /**
     * @brief Get the thread-local curand generator.
     */
    static curandGenerator_t get_gpu_generator();
    
    /**
     * @brief Increment the GPU offset counter after generating random numbers.
     * @param count Number of random values generated.
     */
    static void increment_gpu_offset(size_t count);
#endif

private:
    static thread_local std::unique_ptr<std::mt19937> cpu_gen_;
#ifdef WITH_CUDA
    static thread_local curandGenerator_t gpu_gen_;
    static thread_local unsigned long long gpu_seed_;
    static thread_local unsigned long long gpu_offset_;
    static thread_local bool gpu_gen_initialized_;
#endif
};

/**
 * @brief RAII guard to save and restore RNG state.
 */
class RNGStateGuard {
public:
    RNGStateGuard() : saved_state_(RNG::get_state()) {}
    ~RNGStateGuard() {
        RNG::set_state(saved_state_);
    }

    // No copy or move
    RNGStateGuard(const RNGStateGuard&) = delete;
    RNGStateGuard& operator=(const RNGStateGuard&) = delete;

private:
    RNGState saved_state_;
};

} // namespace OwnTensor