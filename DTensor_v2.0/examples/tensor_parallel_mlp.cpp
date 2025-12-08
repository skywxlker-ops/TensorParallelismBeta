/**
 * Example: Tensor Parallel MLP using shard(), replicate(), and sync()
 * 
 * This demonstrates how the distribution primitives work together
 * in a typical tensor parallel forward pass.
 * 
 * MLP: Y = GELU(X @ W1) @ W2
 * 
 * Column-Parallel for W1: X [M, K] @ W1 [K, N/P] -> H [M, N/P]
 * Row-Parallel for W2:    H [M, N/P] @ W2 [N/P, K] -> Y [M, K] (needs AllReduce)
 */

#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Setup
    auto device_mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
    // Dimensions
    const int M = 4;      // batch size
    const int K = 8;      // input features
    const int N = 16;     // hidden dim (will be sharded: N/P per GPU)
    
    // ============================================================
    // STEP 1: Create input X on ROOT GPU, then REPLICATE to all
    // ============================================================
    DTensor X(device_mesh, pg);
    std::vector<float> x_data(M * K);
    
    // Only root GPU has the actual data initially
    if (rank == 0) {
        for (int i = 0; i < M * K; i++) x_data[i] = 0.1f * (i + 1);
    } else {
        // Other GPUs have placeholder data (will be overwritten by replicate)
        for (int i = 0; i < M * K; i++) x_data[i] = 0.0f;
    }
    
    Layout x_layout = Layout::replicated(device_mesh, {M, K});
    X.setData(x_data, x_layout);
    
    // Use REPLICATE primitive to broadcast X from root to all GPUs
    X.replicate(0);  // root = 0
    
    if (rank == 0) {
        std::cout << "=== Input X (replicated via broadcast) ===" << std::endl;
        std::cout << "Shape: [" << M << ", " << K << "]" << std::endl;
    }
    
    // ============================================================
    // STEP 2: Create W1 (column-sharded: each GPU has K x N/P)
    // ============================================================
    DTensor W1(device_mesh, pg);
    int local_N = N / world_size;
    std::vector<float> w1_data(K * local_N);
    for (int i = 0; i < K * local_N; i++) w1_data[i] = 0.01f * (rank + 1);
    
    // Use sharded layout
    Layout w1_layout(device_mesh, {K, N}, ShardingType::SHARDED, 1);  // shard on dim 1
    W1.setData(w1_data, w1_layout);
    
    if (rank == 0) {
        std::cout << "=== W1 (sharded on dim 1) ===" << std::endl;
        std::cout << "Global: [" << K << ", " << N << "], Local: [" << K << ", " << local_N << "]" << std::endl;
    }
    
    // ============================================================
    // STEP 3: Column-Parallel MatMul: H = X @ W1
    // Output H is SHARDED (each GPU has M x N/P)
    // ============================================================
    DTensor H = X.matmul(W1);
    
    if (rank == 0) {
        std::cout << "\n=== After Column-Parallel MatMul ===" << std::endl;
        std::cout << "H is SHARDED: [" << M << ", " << local_N << "] per GPU" << std::endl;
    }
    
    // ============================================================
    // STEP 4: Create W2 (row-sharded: each GPU has N/P x K)
    // ============================================================
    DTensor W2(device_mesh, pg);
    std::vector<float> w2_data(local_N * K);
    for (int i = 0; i < local_N * K; i++) w2_data[i] = 0.02f;
    
    Layout w2_layout(device_mesh, {N, K}, ShardingType::SHARDED, 0);  // shard on dim 0
    W2.setData(w2_data, w2_layout);
    
    // ============================================================
    // STEP 5: Row-Parallel MatMul: Y = H @ W2 + AllReduce
    // This requires communication to sum partial results
    // ============================================================
    DTensor Y = H.matmul(W2);
    
    if (rank == 0) {
        std::cout << "\n=== After Row-Parallel MatMul ===" << std::endl;
        std::cout << "Y shape: [" << M << ", " << K << "] (synched after AllReduce)" << std::endl;
        
        auto y_data = Y.getData();
        std::cout << "Y values (first 5): ";
        for (int i = 0; i < 5; i++) std::cout << y_data[i] << " ";
        std::cout << std::endl;
    }
    
    // ============================================================
    // STEP 6: Simulate backward pass - sync gradients
    // ============================================================
    // In backward pass, gradients need to be summed across GPUs
    DTensor grad_Y(device_mesh, pg);
    std::vector<float> grad_data(M * K);
    // Generate pseudo-random gradients based on rank and index
    for (int i = 0; i < M * K; i++) {
        grad_data[i] = 0.1f * ((i % 7) + 1) * (rank + 1) + 0.05f * (i % 3);
    }
    
    Layout grad_layout = Layout::replicated(device_mesh, {M, K});
    grad_Y.setData(grad_data, grad_layout);
    
    if (rank == 0) {
        std::cout << "\n=== Before sync() ===" << std::endl;
        auto data = grad_Y.getData();
        std::cout << "Rank 0 grads (first 5): ";
        for (int i = 0; i < 5; i++) std::cout << data[i] << " ";
        std::cout << std::endl;
    }
    
    // Average gradients across all GPUs
    grad_Y.sync();
    
    if (rank == 0) {
        std::cout << "\n=== After sync() - Gradients added ===" << std::endl;
        auto data = grad_Y.getData();
        std::cout << "Averaged grads (first 5): ";
        for (int i = 0; i < 5; i++) std::cout << data[i] << " ";
        std::cout << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
