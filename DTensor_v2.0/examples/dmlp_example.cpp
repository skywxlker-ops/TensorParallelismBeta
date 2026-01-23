/**
 * @file dmlp_example.cpp
 * @brief Example: 4-Layer Tensor Parallel MLP using DModule 
 *   Block 1: X -> DColumnLinear -> DRowLinear -> Y1 (replicated)
 *   Block 2: Y1 -> DColumnLinear -> DRowLinear -> Y2 (replicated)
 * 
 * Compile: make dmlp_example
 * Run: mpirun -np 2 ./examples/dmlp_example
 */

#include <mpi.h>
#include "DTensor_v2.0/mlp_adhi/DistributedNN.h"

using namespace OwnTensor;
using namespace OwnTensor::dnn;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    const int64_t B = 1;   // Batch size
    const int64_t T = 3;   // Sequence length
    const int64_t C = 2;   // Model dimension
    const int64_t F = 4;   // Hidden dimension (to be sharded)
    

    // Data loading - only on root (rank 0)
    // DModule constructors now broadcast a flag to ensure all ranks participate in sharding
    std::vector<float> x_data, w1_data, w2_data, w3_data, w4_data, target_data;
    if (rank == 0) {
        x_data = make_fixed_data(B * T * C, 0.1f);
        w1_data = make_fixed_data(B * C * F, 0.1f);
        w2_data = make_fixed_data(B * F * C, 0.1f);
        w3_data = make_fixed_data(B * C * F, 0.1f);
        w4_data = make_fixed_data(B * F * C, 0.1f);
        target_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        
        // Or load from CSV:
        // x_data = load_csv("data/input.csv");
        // w1_data = load_csv("weights/w1.csv");
        // w2_data = load_csv("weights/w2.csv");
        // w3_data = load_csv("weights/w3.csv");
        // w4_data = load_csv("weights/w4.csv");
        // target_data = load_csv("data/target.csv");
    }
    
    
    std::vector<int> ranks_vec(world_size);
    for (int i = 0; i < world_size; i++) ranks_vec[i] = i;
    DeviceMesh mesh({world_size}, ranks_vec);
    auto pg = mesh.get_process_group(0);
    
    if (rank == 0) {
        std::cout << "╔══════════════════════════╗\n";
        std::cout << "║   DModule 4 Layer MLP    ║\n";
        std::cout << "╚══════════════════════════╝\n\n";
        std::cout << "Configuration: B=" << B << ", T=" << T 
                  << ", C=" << C << ", F=" << F << "\n\n";
    }
    

    DColumnLinear fc1(mesh, pg, B, T, C, F, w1_data);
    DRowLinear fc2(mesh, pg, B, T, F, C, w2_data);
    DColumnLinear fc3(mesh, pg, B, T, C, F, w3_data);
    DRowLinear fc4(mesh, pg, B, T, F, C, w4_data);
    

    std::vector<DTensor*> all_params;
    for (auto* p : fc1.parameters()) all_params.push_back(p);
    for (auto* p : fc2.parameters()) all_params.push_back(p);
    for (auto* p : fc3.parameters()) all_params.push_back(p);
    for (auto* p : fc4.parameters()) all_params.push_back(p);
    
    if (rank == 0) {
        std::cout << "Model created with " << all_params.size() 
                  << " parameter tensors\n\n";
    }
    

    Layout x_layout(mesh, {B, T, C});
    DTensor X(mesh, pg, x_layout, "X");

    X.replicate(0);
    
    if (rank == 0) {
        std::cout << "Input X:\n";
        X.display();
    }
    

    DTensor H1 = fc1.forward(X);
    DTensor Y1 = fc2.forward(H1);
    DTensor H2 = fc3.forward(Y1);
    DTensor Y2 = fc4.forward(H2);
    
    if (rank == 0) {
        std::cout << "\nOutput Y2 (after 4 layers):\n";
        Y2.display();
    }
    

    Layout y_layout(mesh, {B, T, C});
    DTensor target(mesh, pg, y_layout, "target");
    if (rank == 0) target.setData(target_data);
    target.replicate(0);
    
    DTensor loss = dmse_loss(Y2, target);
    
    if (rank == 0) {
        std::cout << "\nMSE Loss:\n";
        loss.display();
    }
    
    loss.backward();
    
    if (rank == 0) {
        std::cout << "\n=== Gradients Computed ===\n";
        std::cout << "fc1.weight gradient:\n";
        fc1.weight->local_tensor().grad_view().display();
        std::cout << "fc4.weight gradient:\n";
        fc4.weight->local_tensor().grad_view().display();
    }
    
 
    float lr = 0.01f;
    for (DTensor* p : all_params) {
        Tensor& t = p->mutable_tensor();
        if (t.requires_grad() && t.has_grad()) {
            t += -lr * t.grad_view();
        }
    }
    
    if (rank == 0) {
        std::cout << "\n SGD step complete (lr=" << lr << ")\n";
    }

    fc1.weight.reset(); fc1.bias.reset();
    fc2.weight.reset(); fc2.bias.reset();
    fc3.weight.reset(); fc3.bias.reset();
    fc4.weight.reset(); fc4.bias.reset();
    
    MPI_Finalize();
    return 0;
}
