/**
 * @file dmlp_example.cpp
 * @brief Example: 4-Layer Tensor Parallel MLP using DModule infrastructure
 * 
 * This example demonstrates using DColumnLinear and DRowLinear
 * to build a 4-layer MLP with tensor parallelism.
 * Uses seeded weights for reproducibility (matches tensor_parallel_mlp_seed.cpp).
 * 
 * Architecture:
 *   Block 1: X -> DColumnLinear -> DRowLinear -> Y1 (replicated)
 *   Block 2: Y1 -> DColumnLinear -> DRowLinear -> Y2 (replicated)
 * 
 * Compile: make dmlp_example
 * Run: mpirun -np 2 ./examples/dmlp_example
 */

#include "nn/DistributedNN.h"
#include <iostream>
#include <mpi.h>

using namespace OwnTensor;
using namespace OwnTensor::dnn;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // === Configuration (matches mlp_seed.cpp) ===
    const int64_t B = 1;   // Batch size
    const int64_t T = 3;   // Sequence length
    const int64_t C = 2;   // Model dimension
    const int64_t F = 4;   // Hidden dimension (will be sharded)
    
    // === Seeded data (matches mlp_seed.cpp) ===
    auto make_fixed_data = [](int64_t size, float base = 0.1f) {
        std::vector<float> data(size);
        for (int64_t i = 0; i < size; i++) {
            data[i] = base * (i + 1);
        }
        return data;
    };
    
    std::vector<float> x_data = make_fixed_data(B * T * C, 0.1f);
    std::vector<float> w1_data = make_fixed_data(B * C * F, 0.1f);
    std::vector<float> w2_data = make_fixed_data(B * F * C, 0.1f);
    std::vector<float> w3_data = make_fixed_data(B * C * F, 0.1f);
    std::vector<float> w4_data = make_fixed_data(B * F * C, 0.1f);
    
    // === Setup Device Mesh ===
    std::vector<int> ranks_vec(world_size);
    for (int i = 0; i < world_size; i++) ranks_vec[i] = i;
    DeviceMesh mesh({world_size}, ranks_vec);
    auto pg = mesh.get_process_group(0);
    
    if (rank == 0) {
        std::cout << "╔══════════════════════════════════════════════════════╗\n";
        std::cout << "║   DModule MLP: 4-Layer Tensor Parallel MLP (Seeded)  ║\n";
        std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
        std::cout << "Configuration: B=" << B << ", T=" << T 
                  << ", C=" << C << ", F=" << F << "\n\n";
    }
    
    int64_t F_local = F / world_size;
    
    // === Create Layers with seeded weights ===
    DColumnLinear fc1(mesh, pg, B, T, C, F, w1_data);  // Up proj, shards w1
    DRowLinear fc2(mesh, pg, B, T, F, C, w2_data);     // Down proj, shards w2
    DColumnLinear fc3(mesh, pg, B, T, C, F, w3_data);  // Up proj, shards w3
    DRowLinear fc4(mesh, pg, B, T, F, C, w4_data);     // Down proj, shards w4
    
    // Collect all parameters
    std::vector<DTensor*> all_params;
    for (auto* p : fc1.parameters()) all_params.push_back(p);
    for (auto* p : fc2.parameters()) all_params.push_back(p);
    for (auto* p : fc3.parameters()) all_params.push_back(p);
    for (auto* p : fc4.parameters()) all_params.push_back(p);
    
    if (rank == 0) {
        std::cout << "Model created with " << all_params.size() 
                  << " parameter tensors (weights + biases)\n\n";
    }
    
    // === Create Input ===
    Layout x_layout(mesh, {B, T, C});
    DTensor X(mesh, pg, x_layout, "X");
    if (rank == 0) X.setData(x_data);
    X.replicate(0);
    
    if (rank == 0) {
        std::cout << "Input X:\n";
        X.display();
    }
    
    // === Pre-allocate intermediate tensors (caller owns them) ===
    Layout h_layout(mesh, {B, T, F_local});  // Sharded hidden
    Layout y_layout(mesh, {B, T, C});         // Replicated output
    
    DTensor H1(mesh, pg, h_layout, "H1");
    DTensor Y1(mesh, pg, y_layout, "Y1");
    DTensor H2(mesh, pg, h_layout, "H2");
    DTensor Y2(mesh, pg, y_layout, "Y2");
    
    // === Forward Pass ===
    fc1.forward(H1, X);
    
    if (rank == 0) {
        std::cout << "\nH1 (after fc1, sharded):\n";
        H1.display();
    }
    
    fc2.forward(Y1, H1);
    
    if (rank == 0) {
        std::cout << "\nY1 (after fc2, replicated):\n";
        Y1.display();
    }
    
    fc3.forward(H2, Y1);
    fc4.forward(Y2, H2);
    
    if (rank == 0) {
        std::cout << "\nOutput Y2 (after 4 layers):\n";
        Y2.display();
    }
    
    // === Create Target and Compute MSE Loss ===
    DTensor target(mesh, pg, y_layout, "target");
    std::vector<float> target_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    if (rank == 0) target.setData(target_data);
    target.replicate(0);
    
    if (rank == 0) {
        std::cout << "\nTarget:\n";
        target.display();
    }
    
    // Compute MSE loss
    Layout loss_layout(mesh, {1});
    DTensor loss(mesh, pg, loss_layout, "loss");
    dmse_loss(loss, Y2, target);
    
    if (rank == 0) {
        std::cout << "\nMSE Loss:\n";
        loss.display();
    }
    
    // === Backward Pass from Loss ===
    loss.backward();
    
    if (rank == 0) {
        std::cout << "\n=== Gradients Computed ===\n";
        
        std::cout << "\n--- Block 1 ---\n";
        std::cout << "fc1.weight gradient:\n";
        fc1.weight->local_tensor().grad_view().display();
        
        std::cout << "fc2.weight gradient:\n";
        fc2.weight->local_tensor().grad_view().display();
        
        std::cout << "\n--- Block 2 ---\n";
        std::cout << "fc3.weight gradient:\n";
        fc3.weight->local_tensor().grad_view().display();
        
        std::cout << "fc4.weight gradient:\n";
        fc4.weight->local_tensor().grad_view().display();
    }
    
    // === SGD Step ===
    float lr = 0.01f;
    for (DTensor* p : all_params) {
        Tensor& t = p->mutable_tensor();
        if (t.requires_grad() && t.has_grad()) {
            Tensor grad = t.grad_view();
            t += -lr * grad;  // Uses GPU tensor ops (operator overloads)
        }
    }
    
    if (rank == 0) {
        std::cout << "\n SGD step complete (lr=" << lr << ")\n";
    }
    
    // Explicit cleanup before MPI_Finalize to avoid CUDA context issues
    // Note: This is a workaround for DTensor's CUDA memory cleanup ordering
    fc1.weight.reset();
    fc1.bias.reset();
    fc2.weight.reset();
    fc2.bias.reset();
    fc3.weight.reset();
    fc3.bias.reset();
    fc4.weight.reset();
    fc4.bias.reset();
    
    MPI_Finalize();
    return 0;
}
