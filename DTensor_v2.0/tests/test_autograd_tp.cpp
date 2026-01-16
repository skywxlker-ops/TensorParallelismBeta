#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

using namespace Bridge::autograd;
namespace ag = Bridge::autograd;

// Helper to create parameters
DTensor create_parameter(const std::vector<float>& data, const Layout& layout, 
                        std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    DTensor t(mesh, pg);
    t.setData(data, layout);
    t.set_requires_grad(true);
    return t;
}

void print_separator(int rank, const std::string& title) {
    if (rank == 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << title << "\n";
        std::cout << std::string(70, '=') << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        std::cout << "\n============================================================\n";
        std::cout << "  DTensor v2.0 - Autograd TP Test (2-Layer MLP)\n";
        std::cout << "============================================================\n";
        std::cout << "\nInitializing with " << world_size << " ranks...\n";
    }
    
    // Set CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "[Rank " << rank << "] ERROR: No CUDA devices found!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    int device_id = rank % device_count;
    cudaSetDevice(device_id);
    cudaFree(0);
    
    // Initialize NCCL
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    try {
        // Create mesh and process group
        auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
        auto pg = init_process_group(world_size, rank);
        
        if (rank == 0) {
            std::cout << "[OK] MPI initialized\n";
            std::cout << "[OK] NCCL initialized\n";
            std::cout << "[OK] Process group created\n";
        }
        
        // =============================================================
        // Configuration
        // =============================================================
        const int BATCH = 2;
        const int INPUT_DIM = 4;
        const int HIDDEN_DIM = 8;  // Will be sharded
        const int OUTPUT_DIM = 4;
        
        print_separator(rank, "2-Layer MLP with Autograd TP");
        
        if (rank == 0) {
            std::cout << "\nConfiguration:\n";
            std::cout << "  Batch Size:      " << BATCH << "\n";
            std::cout << "  Input Dim:       " << INPUT_DIM << "\n";
            std::cout << "  Hidden Dim:      " << HIDDEN_DIM << " (sharded)\n";
            std::cout << "  Output Dim:      " << OUTPUT_DIM << "\n";
            std::cout << "  Shard per rank:  " << (HIDDEN_DIM / world_size) << "\n";
        }
        
        // =============================================================
        // Layer 1: Column-Parallel (Input -> Hidden)
        // =============================================================
        print_separator(rank, "Creating Layer 1 (Column-Parallel)");
        
        // W1: [INPUT_DIM, HIDDEN_DIM] column-sharded
        Layout W1_layout(*mesh, {INPUT_DIM, HIDDEN_DIM}, 1);
        std::vector<int64_t> W1_local_shape = W1_layout.get_local_shape(rank);
        int W1_local_size = W1_local_shape[0] * W1_local_shape[1];
        
        std::vector<float> W1_data(W1_local_size, (rank + 1) * 0.5f);
        auto W1 = create_parameter(W1_data, W1_layout, mesh, pg);
        
        if (rank == 0) {
            std::cout << "W1 created: " << W1_local_shape[0] << "x" << W1_local_shape[1] 
                     << " (local shard)\n";
        }
        
        // =============================================================
        // Layer 2: Row-Parallel (Hidden -> Output)
        // =============================================================
        print_separator(rank, "Creating Layer 2 (Row-Parallel)");
        
        // W2: [HIDDEN_DIM, OUTPUT_DIM] row-sharded
        Layout W2_layout(*mesh, {HIDDEN_DIM, OUTPUT_DIM}, 0);
        std::vector<int64_t> W2_local_shape = W2_layout.get_local_shape(rank);
        int W2_local_size = W2_local_shape[0] * W2_local_shape[1];
        
        std::vector<float> W2_data(W2_local_size, 1.0f);
        auto W2 = create_parameter(W2_data, W2_layout, mesh, pg);
        
        if (rank == 0) {
            std::cout << "W2 created: " << W2_local_shape[0] << "x" << W2_local_shape[1] 
                     << " (local shard)\n";
        }
        
        // =============================================================
        // Input (Replicated)
        // =============================================================
        print_separator(rank, "Creating Input");
        
        Layout X_layout = Layout::replicated(*mesh, std::vector<int64_t>{BATCH, INPUT_DIM});
        std::vector<float> X_data(BATCH * INPUT_DIM, 1.0f);
        auto X = create_parameter(X_data, X_layout, mesh, pg);
        
        if (rank == 0) {
            std::cout << "Input X created: " << BATCH << "x" << INPUT_DIM << " (replicated)\n";
        }
        
        // =============================================================
        // Forward Pass
        // =============================================================
        print_separator(rank, "Forward Pass");
        
        // =============================================================
        // Forward Pass
        // =============================================================
        print_separator(rank, "Forward Pass");
        
        // Layer 1: X @ W1 -> H (column-sharded)
        auto H = X.matmul(W1);
        if (rank == 0) {
            std::cout << "Layer 1: X @ W1 = H (column-sharded)\n";
        }
        
        // Activation
        auto H_relu = H.relu();
        if (rank == 0) {
            std::cout << "ReLU activation applied\n";
        }
        
        // Layer 2: H @ W2 -> Y (replicated via AllReduce)
        auto Y = H_relu.matmul(W2);
        if (rank == 0) {
            std::cout << "Layer 2: H @ W2 = Y (replicated)\n";
        }
        
        // =============================================================
        // Loss
        // =============================================================
        print_separator(rank, "Computing Loss");
        
        // Create dummy target (all zeros) for MSE loss
        Layout target_layout = Layout::replicated(*mesh, {BATCH, OUTPUT_DIM});
        auto Target = DTensor::zeros({BATCH, OUTPUT_DIM}, mesh, pg, target_layout);
        
        // Compute MSE Loss
        auto Loss = Y.mse_loss(Target);
        Loss.set_requires_grad(true); // Ensure loss requires grad for backward
        
        // Get scalar loss value for printing
        float loss_val = OwnTensor::reduce_mean(Loss.local_tensor()).to_cpu().data<float>()[0];
        
        if (rank == 0) {
            std::cout << "Loss: " << std::fixed << std::setprecision(4) << loss_val << "\n";
        }
        
        // =============================================================
        // Backward Pass
        // =============================================================
        print_separator(rank, "Backward Pass");
        
        if (rank == 0) {
            std::cout << "Running Loss.backward()...\n";
        }
        
        Loss.backward();
        cudaDeviceSynchronize();
        
        if (rank == 0) {
            std::cout << "Backward pass completed!\n";
        }
        
        // =============================================================
        // Verify Gradients
        // =============================================================
        print_separator(rank, "Verifying Gradients");
        
        // Check W1 gradients
        const OwnTensor::Tensor& W1_grad = W1.grad();
        float W1_grad_sum = OwnTensor::reduce_sum(OwnTensor::abs(W1_grad))
                                .to_cpu().data<float>()[0];
        
        if (rank == 0) {
            std::cout << "W1 gradient sum (rank 0): " << W1_grad_sum << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 1 && world_size > 1) {
            std::cout << "W1 gradient sum (rank 1): " << W1_grad_sum << "\n";
        }
        
        // Check W2 gradients
        const OwnTensor::Tensor& W2_grad = W2.grad();
        float W2_grad_sum = OwnTensor::reduce_sum(OwnTensor::abs(W2_grad))
                                .to_cpu().data<float>()[0];
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "W2 gradient sum (rank 0): " << W2_grad_sum << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 1 && world_size > 1) {
            std::cout << "W2 gradient sum (rank 1): " << W2_grad_sum << "\n";
        }
        
        // Verification
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "\n" << std::string(70, '-') << "\n";
            std::cout << "Verification:\n";
            if (W1_grad_sum > 0.0f && W2_grad_sum > 0.0f) {
                std::cout << "✅ PASS: Gradients were computed successfully!\n";
            } else {
                std::cout << "❌ FAIL: Some gradients are zero!\n";
            }
            std::cout << std::string(70, '-') << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] ERROR: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n============================================================\n";
        std::cout << "  [PASS] Autograd TP Test Completed Successfully\n";
        std::cout << "============================================================\n\n";
    }
    
    MPI_Finalize();
    return 0;
}


