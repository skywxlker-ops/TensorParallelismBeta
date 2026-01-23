/**
 * @file test_4layer_mlp.cpp
 * @brief 4-Layer Tensor Parallel MLP Test with Autograd
 * 
 * Architecture:
 *   Block 1: X -> ColLinear(W1) -> ReLU -> RowLinear(W2) -> H1 (replicated after sync)
 *   Block 2: H1 -> ColLinear(W3) -> ReLU -> RowLinear(W4) -> Y (replicated after sync)
 * 
 * This tests the full forward + backward through 4 tensor-parallel layers.
 * 
 * Compile: make test_4layer_mlp
 * Run: mpirun -np 2 ./test_4layer_mlp
 */

#include <unparalleled/unparalleled.h>
#include "nn/nn.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cudaSetDevice(rank);
    
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    auto pg = init_process_group(world_size, rank);
    
    if (rank == 0) {
        std::cout << "\n╔══════════════════════════════════════╗" << std::endl;
        std::cout << "║   4-Layer Tensor Parallel MLP Test   ║" << std::endl;
        std::cout << "╚══════════════════════════════════════╝\n" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
    }
    
    // Hyperparameters
    int M = 4;     // Batch size
    int K = 16;    // Input/Output features
    int H1 = 32;   // Hidden features for layer 1 (column-parallel output)
    int H2 = 32;   // Hidden features for layer 2 (column-parallel output)
    
    assert(H1 % world_size == 0 && "H1 must be divisible by world_size");
    assert(H2 % world_size == 0 && "H2 must be divisible by world_size");
    
    if (rank == 0) {
        std::cout << "Config: M=" << M << ", K=" << K << ", H1=" << H1 << ", H2=" << H2 << std::endl;
    }
    
    // =========================================================================
    // Initialize Weights and Inputs
    // =========================================================================
    
    // Layouts
    Layout x_layout = Layout::replicated(*mesh, {M, K});
    Layout target_layout = Layout::replicated(*mesh, {M, K});
    
    // Layer 1: Column-Parallel (shard output dim)
    Layout w1_layout(*mesh, {K, H1}, 1);   // [K, H1] sharded dim 1
    // Layer 2: Row-Parallel (shard input dim)
    Layout w2_layout(*mesh, {H1, K}, 0);   // [H1, K] sharded dim 0
    // Layer 3: Column-Parallel
    Layout w3_layout(*mesh, {K, H2}, 1);   // [K, H2] sharded dim 1
    // Layer 4: Row-Parallel
    Layout w4_layout(*mesh, {H2, K}, 0);   // [H2, K] sharded dim 0
    
    // Input X
    auto X = DTensor::randn({M, K}, mesh, pg, x_layout);
    
    // Weights with He initialization scale
    float scale1 = std::sqrt(2.0f / K);
    float scale2 = std::sqrt(2.0f / H1);
    float scale3 = std::sqrt(2.0f / K);
    float scale4 = std::sqrt(2.0f / H2);
    
    auto W1 = DTensor::randn({K, H1}, mesh, pg, w1_layout);
    auto W2 = DTensor::randn({H1, K}, mesh, pg, w2_layout);
    auto W3 = DTensor::randn({K, H2}, mesh, pg, w3_layout);
    auto W4 = DTensor::randn({H2, K}, mesh, pg, w4_layout);
    
    W1.scale(scale1);
    W2.scale(scale2);
    W3.scale(scale3);
    W4.scale(scale4);
    
    W1.set_requires_grad(true);
    W2.set_requires_grad(true);
    W3.set_requires_grad(true);
    W4.set_requires_grad(true);
    
    // Target Y_true (zeros for simplicity)
    auto Target = DTensor::zeros({M, K}, mesh, pg, target_layout);
    
    if (rank == 0) std::cout << "\nInitialization complete." << std::endl;
    
    // =========================================================================
    // Forward Pass
    // =========================================================================
    if (rank == 0) std::cout << "\n=== Forward Pass ===" << std::endl;
    
    // Layer 1: Column-Parallel Matmul
    if (rank == 0) std::cout << "Layer 1: X @ W1 (column-parallel)..." << std::endl;
    auto H1_out = X.matmul(W1);
    
    // Activation: ReLU
    if (rank == 0) std::cout << "Activation: ReLU..." << std::endl;
    auto H1_act = H1_out.relu();
    
    // Layer 2: Row-Parallel Matmul (includes AllReduce in sync)
    if (rank == 0) std::cout << "Layer 2: H1 @ W2 (row-parallel + sync)..." << std::endl;
    auto Y1 = H1_act.matmul(W2);
    
    // Layer 3: Column-Parallel Matmul
    if (rank == 0) std::cout << "Layer 3: Y1 @ W3 (column-parallel)..." << std::endl;
    auto H2_out = Y1.matmul(W3);
    
    // Activation: ReLU
    if (rank == 0) std::cout << "Activation: ReLU..." << std::endl;
    auto H2_act = H2_out.relu();
    
    // Layer 4: Row-Parallel Matmul (includes AllReduce in sync)
    if (rank == 0) std::cout << "Layer 4: H2 @ W4 (row-parallel + sync)..." << std::endl;
    auto Y = H2_act.matmul(W4);
    
    // =========================================================================
    // Loss Computation
    // =========================================================================
    if (rank == 0) std::cout << "\n=== Loss Computation ===" << std::endl;
    
    auto Loss = Y.mse_loss(Target);
    float loss_val = Loss.getData()[0];
    
    if (rank == 0) {
        std::cout << "MSE Loss: " << std::fixed << std::setprecision(6) << loss_val << std::endl;
    }
    
    // =========================================================================
    // Backward Pass
    // =========================================================================
    if (rank == 0) std::cout << "\n=== Backward Pass ===" << std::endl;
    
    W1.zero_grad();
    W2.zero_grad();
    W3.zero_grad();
    W4.zero_grad();
    
    Loss.backward();
    
    if (rank == 0) std::cout << "Backward pass complete." << std::endl;
    
    // =========================================================================
    // Verify Gradients
    // =========================================================================
    if (rank == 0) std::cout << "\n=== Gradient Verification ===" << std::endl;
    
    auto grad_W1 = W1.grad();
    auto grad_W2 = W2.grad();
    auto grad_W3 = W3.grad();
    auto grad_W4 = W4.grad();
    
    // Expected local shapes after sharding
    int H1_local = H1 / world_size;
    int H2_local = H2 / world_size;
    
    bool ok_w1 = (grad_W1.numel() == static_cast<size_t>(K * H1_local));
    bool ok_w2 = (grad_W2.numel() == static_cast<size_t>(H1_local * K));
    bool ok_w3 = (grad_W3.numel() == static_cast<size_t>(K * H2_local));
    bool ok_w4 = (grad_W4.numel() == static_cast<size_t>(H2_local * K));
    
    if (rank == 0) {
        std::cout << "Grad W1 shape: [" << K << ", " << H1_local << "] | OK: " << (ok_w1 ? "YES" : "NO") << std::endl;
        std::cout << "Grad W2 shape: [" << H1_local << ", " << K << "] | OK: " << (ok_w2 ? "YES" : "NO") << std::endl;
        std::cout << "Grad W3 shape: [" << K << ", " << H2_local << "] | OK: " << (ok_w3 ? "YES" : "NO") << std::endl;
        std::cout << "Grad W4 shape: [" << H2_local << ", " << K << "] | OK: " << (ok_w4 ? "YES" : "NO") << std::endl;
    }
    
    // Check gradient values (should be non-zero)
    auto grad_W1_cpu = grad_W1.to_cpu();
    auto grad_W4_cpu = grad_W4.to_cpu();
    
    float grad_w1_sum = 0.0f, grad_w4_sum = 0.0f;
    for (size_t i = 0; i < grad_W1_cpu.numel(); ++i) {
        grad_w1_sum += std::abs(grad_W1_cpu.data<float>()[i]);
    }
    for (size_t i = 0; i < grad_W4_cpu.numel(); ++i) {
        grad_w4_sum += std::abs(grad_W4_cpu.data<float>()[i]);
    }
    
    bool has_gradients = (grad_w1_sum > 1e-8f && grad_w4_sum > 1e-8f);
    
    if (rank == 0) {
        std::cout << "\nGrad W1 L1-norm: " << grad_w1_sum << std::endl;
        std::cout << "Grad W4 L1-norm: " << grad_w4_sum << std::endl;
        std::cout << "Gradients non-zero: " << (has_gradients ? "YES" : "NO") << std::endl;
    }
    
    // =========================================================================
    // Result
    // =========================================================================
    bool all_ok = ok_w1 && ok_w2 && ok_w3 && ok_w4 && has_gradients;
    
    if (rank == 0) {
        std::cout << "\n╔══════════════════════════════════════╗" << std::endl;
        std::cout << "║  Test Result: " << (all_ok ? "PASSED ✓" : "FAILED ✗") << "                 ║" << std::endl;
        std::cout << "╚══════════════════════════════════════╝\n" << std::endl;
    }
    
    // Cleanup
    MPI_Barrier(MPI_COMM_WORLD);
    pg.reset();
    MPI_Finalize();
    
    return all_ok ? 0 : 1;
}
