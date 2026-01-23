/**
 * @file hooks_example.cpp
 * @brief Example: Using Hooks with DTensor/TP Models
 * 
 * This example demonstrates:
 * 1. Pre-Hooks: Modify or inspect gradients BEFORE they are applied
 * 2. Post-Accumulation Hooks: Trigger actions AFTER gradients are accumulated
 *    (useful for DDP synchronization or gradient logging)
 * 
 * Compile: (from DTensor_v2.0 directory)
 *   make hooks_example
 * 
 * Run:
 *   mpirun -np 2 ./nn/examples/hooks_example
 */

#include <iostream>
#include <memory>
#include <cmath>
#include <mpi.h>
#include <nccl.h>

// Tensor library with hooks support
#include "core/Tensor.h"
#include "autograd/Hooks.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"

// DTensor components
#include <unparalleled/unparalleled.h>

using namespace OwnTensor;

// =============================================================================
// Example 1: Gradient Clipping Pre-Hook
// =============================================================================
/**
 * Pre-hooks are called BEFORE the gradient is used in backward.
 * Use case: Gradient clipping to prevent exploding gradients.
 */
class GradientClipHook : public FunctionPreHook {
public:
    explicit GradientClipHook(float max_norm) : max_norm_(max_norm) {}
    
    Tensor operator()(const Tensor& grad) override {
        // Compute L2 norm of gradient
        auto grad_cpu = grad.to_cpu();
        const float* g = grad_cpu.data<float>();
        size_t numel = grad_cpu.numel();
        
        float norm_sq = 0.0f;
        for (size_t i = 0; i < numel; ++i) {
            norm_sq += g[i] * g[i];
        }
        float norm = std::sqrt(norm_sq);
        
        if (norm > max_norm_) {
            float scale = max_norm_ / norm;
            std::cout << "  [GradClipHook] Norm " << norm 
                      << " exceeds " << max_norm_ 
                      << ", scaling by " << scale << "\n";
            
            // Scale the gradient (create a new tensor with scaled values)
            std::vector<float> scaled(numel);
            for (size_t i = 0; i < numel; ++i) {
                scaled[i] = g[i] * scale;
            }
            
            // Create new tensor with scaled values
            Tensor clipped = Tensor::zeros(grad.shape(), 
                TensorOptions().with_device(grad.device()).with_dtype(grad.dtype()));
            cudaMemcpy(clipped.data<float>(), scaled.data(), 
                       numel * sizeof(float), cudaMemcpyHostToDevice);
            return clipped;
        }
        return grad;  // Return unchanged if within bounds
    }
    
private:
    float max_norm_;
};

// =============================================================================
// Example 2: Gradient Logging Post-Accumulation Hook
// =============================================================================
/**
 * Post-accumulation hooks are called AFTER the gradient is fully accumulated
 * into a leaf tensor's .grad field.
 * 
 * Use case 1: Logging gradient statistics for debugging.
 * Use case 2: DDP synchronization (AllReduce gradients after accumulation).
 */
class GradientLogHook : public PostAccumulateGradHook {
public:
    explicit GradientLogHook(const std::string& param_name) : name_(param_name) {}
    
    void operator()(const Tensor& grad) override {
        auto grad_cpu = grad.to_cpu();
        const float* g = grad_cpu.data<float>();
        size_t numel = grad_cpu.numel();
        
        // Compute statistics
        float min_val = g[0], max_val = g[0], sum = 0.0f;
        for (size_t i = 0; i < numel; ++i) {
            min_val = std::min(min_val, g[i]);
            max_val = std::max(max_val, g[i]);
            sum += std::abs(g[i]);
        }
        float mean_abs = sum / numel;
        
        std::cout << "  [GradLog] " << name_ 
                  << " | min: " << min_val 
                  << " | max: " << max_val 
                  << " | mean_abs: " << mean_abs << "\n";
    }
    
private:
    std::string name_;
};

// =============================================================================
// Example 3: DDP-style AllReduce Hook (for Data Parallelism)
// =============================================================================
/**
 * This hook triggers gradient synchronization across all ranks.
 * In pure Tensor Parallelism, this is NOT needed because sync() is explicit.
 * In Data Parallelism (DDP), this is the core mechanism.
 */
class DDPSyncHook : public PostAccumulateGradHook {
public:
    DDPSyncHook(std::shared_ptr<ProcessGroupNCCL> pg, const std::string& name)
        : pg_(pg), name_(name) {}
    
    void operator()(const Tensor& grad) override {
        // In a real DDP implementation, you would:
        // 1. Call NCCL AllReduce on the gradient
        // 2. Scale by 1/world_size
        
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cout << "  [DDPSync] Rank " << rank 
                  << " would AllReduce gradient for " << name_ << "\n";
        
        // Example (pseudo-code):
        // pg_->allreduce(grad, ReduceOp::SUM);
        // grad *= (1.0f / world_size);
    }
    
private:
    std::shared_ptr<ProcessGroupNCCL> pg_;
    std::string name_;
};

// =============================================================================
// Main: Demonstrate Hooks Usage
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Set GPU for this rank
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    int device_id = rank % device_count;
    cudaSetDevice(device_id);
    cudaFree(0);  // Initialize CUDA context
    
    if (rank == 0) {
        std::cout << "╔══════════════════════════════════════════════════════╗\n";
        std::cout << "║       Hooks Example for DTensor/TP Models            ║\n";
        std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    }
    
    // Initialize NCCL
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    try {
        // Setup DTensor infrastructure (matching test_customdnn_mlp.cpp pattern)
        auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
        auto pg = init_process_group(world_size, rank);
        
        // Create a simple parameter tensor with requires_grad=true
        TensorOptions opts = TensorOptions()
            .with_device(DeviceIndex(Device::CUDA, device_id))
            .with_req_grad(true);
        
        Tensor weight = Tensor::randn<float>(Shape{{4, 4}}, opts);
        weight.set_requires_grad(true);  // Explicitly enable autograd
        
        // =========================================================================
        // Register Hooks on the Tensor
        // =========================================================================
        if (rank == 0) {
            std::cout << "=== Registering Hooks ===\n";
        }
        
        // 1. Pre-hook: Gradient clipping
        weight.register_hook(std::make_unique<GradientClipHook>(1.0f));
        if (rank == 0) std::cout << "Registered GradientClipHook (max_norm=1.0)\n";
        
        // 2. Post-accumulation hook: Gradient logging
        weight.register_post_acc_hook(std::make_unique<GradientLogHook>("weight"));
        if (rank == 0) std::cout << "Registered GradientLogHook\n";
        
        // 3. (Optional) DDP sync hook - uncomment for data parallelism
        // weight.register_post_acc_hook(std::make_unique<DDPSyncHook>(pg, "weight"));
        
        if (rank == 0) std::cout << "\n";
        
        // =========================================================================
        // Forward Pass (using autograd operations from test_hooks.cpp pattern)
        // =========================================================================
        if (rank == 0) {
            std::cout << "=== Forward Pass ===\n";
        }
        
        // Use simple operations that properly build the autograd graph
        // y = weight * weight + weight  (similar to test_hooks.cpp)
        Tensor w_sq = autograd::mul(weight, weight);
        Tensor y = autograd::add(w_sq, weight);
        Tensor loss = autograd::sum(y);
        
        if (rank == 0) {
            std::cout << "weight shape: [4, 4]\n";
            std::cout << "Computing: y = weight * weight + weight\n";
            std::cout << "loss = sum(y)\n\n";
        }
        
        // =========================================================================
        // Backward Pass - Hooks will be triggered here!
        // =========================================================================
        if (rank == 0) {
            std::cout << "=== Backward Pass (Hooks Triggered) ===\n";
        }
        
        loss.backward();
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // =========================================================================
        // Verify Gradient
        // =========================================================================
        if (rank == 0) {
            std::cout << "\n=== Gradient Verification ===\n";
            Tensor grad = weight.grad_view();
            auto grad_cpu = grad.to_cpu();
            const float* g = grad_cpu.data<float>();
            
            std::cout << "First 4 gradient values: ";
            for (int i = 0; i < 4; ++i) {
                std::cout << g[i] << " ";
            }
            std::cout << "\n";
        }
        
        if (rank == 0) {
            std::cout << "\n✅ Hooks example complete!\n";
            std::cout << "\nKey Takeaways:\n";
            std::cout << "  - Pre-hooks: Use for gradient modification (clipping, scaling)\n";
            std::cout << "  - Post-acc hooks: Use for logging, DDP sync, or custom actions\n";
            std::cout << "  - For pure TP: Hooks are optional (use explicit sync())\n";
            std::cout << "  - For DDP: Post-acc hooks are essential for gradient sync\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] ERROR: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    MPI_Finalize();
    return 0;
}
