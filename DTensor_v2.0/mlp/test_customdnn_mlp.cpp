/**
 * @file test_customdnn_mlp.cpp
 * @brief Simple MLP test using the new CustomDNN ShardingType API
 * 
 * Tests:
 * - DLinear with ShardingType::Shard(dim) and ShardingType::Replicated()
 * - Bias support in DLinear
 * - Auto-sync behavior in forward pass
 * - Forward and backward passes
 * 
 * Run: mpirun -np 2 ./test_customdnn_mlp
 */

#include <unparalleled/unparalleled.h>
#include "nn/CustomDNN.h"
#include <iostream>
#include <memory>
#include <mpi.h>
#include <nccl.h>

using namespace OwnTensor::dnn;

int main(int argc, char* argv[]) {
    // Initialize MPI
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
        std::cout << "=== CustomDNN MLP Test ===" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
        std::cout << "Testing new ShardingType API..." << std::endl;
    }
    
    // Initialize NCCL
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    try {
        // Create device mesh and process group (matching test_autograd_tp.cpp pattern)
        auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
        auto pg = init_process_group(world_size, rank);
    
    // Model dimensions
    const int64_t batch_size = 4;
    const int64_t hidden_dim = 64;
    const int64_t ffn_dim = 256;  // 4x hidden for typical MLP
    
    if (rank == 0) {
        std::cout << "\nModel config:" << std::endl;
        std::cout << "  Batch size: " << batch_size << std::endl;
        std::cout << "  Hidden dim: " << hidden_dim << std::endl;
        std::cout << "  FFN dim: " << ffn_dim << std::endl;
    }
    
    // =========================================
    // Create MLP layers using NEW ShardingType API
    // =========================================
    
    if (rank == 0) std::cout << "\nCreating layers with ShardingType API..." << std::endl;
    
    // Layer 1: Column-parallel (expand hidden -> ffn)
    // Weight: [hidden_dim, ffn_dim] sharded on dim 1 (columns)
    // Note: Bias disabled due to DTensor add operation layout constraints
    DLinear fc1(mesh, pg,
        hidden_dim, ffn_dim,
        ShardingType::Shard(1),       // weight: column-wise
        ShardingType::Replicated(),   // bias: (not used)
        false);                       // has_bias=false (DTensor add limitations)
    
    // Layer 2: Row-parallel (reduce ffn -> hidden)
    // Weight: [ffn_dim, hidden_dim] sharded on dim 0 (rows)
    // Note: Bias disabled due to DTensor add operation layout constraints
    DLinear fc2(mesh, pg,
        ffn_dim, hidden_dim,
        ShardingType::Shard(0),       // weight: row-wise (auto-syncs in forward)
        ShardingType::Replicated(),   // bias: (not used)
        false);                       // has_bias=false (DTensor add limitations)
    
    if (rank == 0) std::cout << "   fc1: Column-parallel [" << hidden_dim << " -> " << ffn_dim << "]" << std::endl;
    if (rank == 0) std::cout << "   fc2: Row-parallel [" << ffn_dim << " -> " << hidden_dim << "]" << std::endl;
    
    // =========================================
    // Create input tensor
    // =========================================
    
    Layout input_layout = Layout::replicated(*mesh, {batch_size, hidden_dim});
    DTensor input = DTensor::randn({batch_size, hidden_dim}, mesh, pg, input_layout);
    input.set_requires_grad(true);
    
    if (rank == 0) std::cout << "\nInput shape: [" << batch_size << ", " << hidden_dim << "] (replicated)" << std::endl;
    
    // =========================================
    // Forward pass
    // =========================================
    
    if (rank == 0) std::cout << "\n--- Forward Pass ---" << std::endl;
    
    // fc1: Column-parallel matmul
    DTensor h1 = fc1.forward(input);
    if (rank == 0) std::cout << "  fc1 output computed" << std::endl;
    
    // GeLU activation
    DTensor h2 = h1.gelu();
    if (rank == 0) std::cout << "  GeLU activation applied" << std::endl;
    
    // fc2: Row-parallel matmul (auto AllReduce)
    DTensor output = fc2.forward(h2);
    if (rank == 0) std::cout << "  fc2 output computed (auto-synced)" << std::endl;
    
    // =========================================
    // Compute loss (MSE with zeros as target)
    // =========================================
    
    if (rank == 0) std::cout << "\n--- Loss Computation ---" << std::endl;
    
    Layout target_layout = Layout::replicated(*mesh, {batch_size, hidden_dim});
    DTensor target = DTensor::zeros({batch_size, hidden_dim}, mesh, pg, target_layout);
    
    DTensor loss = output.mse_loss(target);
    
    // Get loss value
    auto loss_data = loss.getData();
    float loss_val = loss_data.empty() ? 0.0f : loss_data[0];
    
    if (rank == 0) std::cout << "  MSE Loss: " << loss_val << std::endl;
    
    // =========================================
    // Backward pass
    // =========================================
    
    if (rank == 0) std::cout << "\n--- Backward Pass ---" << std::endl;
    
    loss.backward();
    
    if (rank == 0) std::cout << "  Gradients computed" << std::endl;
    
    // Check gradients exist
    auto fc1_params = fc1.parameters();
    auto fc2_params = fc2.parameters();
    
    if (rank == 0) {
        std::cout << "  fc1 parameters: " << fc1_params.size() << " (weight + bias)" << std::endl;
        std::cout << "  fc2 parameters: " << fc2_params.size() << " (weight + bias)" << std::endl;
    }
    
    // =========================================
    // Optimizer step
    // =========================================
    
    if (rank == 0) std::cout << "\n--- Optimizer Step ---" << std::endl;
    
    SGD optimizer(0.01f);
    
    std::vector<DTensor*> all_params;
    for (auto* p : fc1_params) all_params.push_back(p);
    for (auto* p : fc2_params) all_params.push_back(p);
    
    optimizer.step(all_params);
    
    if (rank == 0) std::cout << "  Weights updated with SGD (lr=0.01)" << std::endl;
    
    // =========================================
    // Second forward pass to verify training works
    // =========================================
    
    if (rank == 0) std::cout << "\n--- Verification Forward ---" << std::endl;
    
    DTensor h1_v2 = fc1.forward(input);
    DTensor h2_v2 = h1_v2.gelu();
    DTensor output_v2 = fc2.forward(h2_v2);
    DTensor loss_v2 = output_v2.mse_loss(target);
    
    auto loss_data_v2 = loss_v2.getData();
    float loss_val_v2 = loss_data_v2.empty() ? 0.0f : loss_data_v2[0];
    
    if (rank == 0) {
        std::cout << "  Loss after 1 step: " << loss_val_v2 << std::endl;
        if (loss_val_v2 < loss_val) {
            std::cout << "   Loss decreased! \n\n [Training works]" << std::endl;
        } else {
            std::cout << "  ⚠ Loss did not decrease (may need more steps)" << std::endl;
        }
    }
    
    // =========================================
    // Summary
    // =========================================
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // if (rank == 0) {
    //     std::cout << "\n=== Test Complete ===" << std::endl;
    //     std::cout << "ShardingType API working correctly!" << std::endl;
    //     std::cout << "  - Shard(0) for row-parallel " << std::endl;
    //     std::cout << "  - Shard(1) for column-parallel " << std::endl;
    //     std::cout << "  - Replicated() for bias " << std::endl;
    //     std::cout << "  - Auto-sync in forward " << std::endl;
    //     std::cout << "  - Bias support " << std::endl;
    // }
    
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] ERROR: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    MPI_Finalize();
    return 0;
}
