#include <unparalleled/unparalleled.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>

/**
 * Simple 2-Layer MLP using DTensor with Autograd
 * 
 * Architecture:
 * Input X [M, K] (Replicated)
 * Layer 1: W1 [K, N] (Column-Parallel) -> H [M, N/P] (Sharded dim 1)
 * Activation: ReLU(H) -> H_act [M, N/P]
 * Layer 2: W2 [N, K] (Row-Parallel) -> Y_partial [M, K] (Replicated after AllReduce)
 * Loss: MSE(Y, Target)
 */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cudaSetDevice(rank);
    
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    auto pg = init_process_group(world_size, rank);
    
    if (rank == 0) {
        std::cout << "\n=== DTensor MLP Autograd Test ===" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
    }
    
    // Hyperparameters
    int M = 4;    // Batch size
    int K = 8;    // Input/Output features
    int N = 16;   // Hidden features (must be divisible by world_size)
    
    assert(N % world_size == 0);
    
    // 1. Initialize Weights and Inputs
    Layout x_layout = Layout::replicated(*mesh, {M, K});
    Layout w1_layout(*mesh, {K, N}, 1);  // Column-sharded
    Layout w2_layout(*mesh, {N, K}, 0);  // Row-sharded
    Layout target_layout = Layout::replicated(*mesh, {M, K});
    
    // Input X
    auto X = DTensor::randn({M, K}, mesh, pg, x_layout);
    
    // Weights
    auto W1 = DTensor::randn({K, N}, mesh, pg, w1_layout);
    auto W2 = DTensor::randn({N, K}, mesh, pg, w2_layout);
    
    W1.set_requires_grad(true);
    W2.set_requires_grad(true);
    
    // Target Y_true
    auto Target = DTensor::zeros({M, K}, mesh, pg, target_layout);
    
    if (rank == 0) std::cout << "Initialization complete." << std::endl;
    
    // 2. Forward Pass
    if (rank == 0) std::cout << "Starting Forward Pass..." << std::endl;
    
    // Layer 1: Column-Parallel Matmul
    auto H = X.matmul(W1);
    
    // Activation: ReLU
    auto H_act = H.relu();
    
    // Layer 2: Row-Parallel Matmul
    auto Y_pred = H_act.matmul(W2);
    
    // 3. Loss Computation
    auto Loss = Y_pred.mse_loss(Target);
    float loss_val = Loss.getData()[0];
    
    if (rank == 0) {
        std::cout << "Forward Pass complete. Loss: " << std::fixed << std::setprecision(6) << loss_val << std::endl;
    }
    
    // 4. Backward Pass
    if (rank == 0) std::cout << "Starting Backward Pass..." << std::endl;
    
    W1.zero_grad(); 
    W2.zero_grad();
    
    Loss.backward();
    
    if (rank == 0) std::cout << "Backward Pass complete." << std::endl;
    
    // 5. Verify Gradients
    auto grad_W1 = W1.grad();
    auto grad_W2 = W2.grad();
    
    // Basic verification: gradients should be non-empty and have correct shard shapes
    bool ok_w1 = (grad_W1.numel() == static_cast<size_t>(K * (N / world_size)));
    bool ok_w2 = (grad_W2.numel() == static_cast<size_t>((N / world_size) * K));
    
    if (rank == 0) {
        std::cout << "\n=== Verification ===" << std::endl;
        std::cout << "Grad W1 shape: [" << K << ", " << N/world_size << "] | OK: " << (ok_w1 ? "YES" : "NO") << std::endl;
        std::cout << "Grad W2 shape: [" << N/world_size << ", " << K << "] | OK: " << (ok_w2 ? "YES" : "NO") << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    pg.reset();
    MPI_Finalize();
    
    return (ok_w1 && ok_w2) ? 0 : 1;
}
