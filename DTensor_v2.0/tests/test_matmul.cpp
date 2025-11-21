#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "tensor/placement.h"
#include "tensor/layout.h"
#include "process_group/process_group.h"
#include "bridge/tensor_ops_bridge.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <mpi.h>

// Helper to generate random data
std::vector<float> randomData(int size, int seed = 42) {
    std::vector<float> data(size);
    srand(seed);
    for (int i = 0; i < size; ++i) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
    }
    return data;
}

// Helper to check if two vectors are approximately equal
bool isClose(const std::vector<float>& a, const std::vector<float>& b, float rtol = 1e-3, float atol = 1e-4) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        float threshold = atol + rtol * std::abs(b[i]);
        if (diff > threshold) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] 
                      << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

// Test column-parallel matmul
// X [M, K] (replicated) @ W [K, N] (column-sharded) -> Y [M, N] (column-sharded)
void test_column_parallel_matmul(int rank, int world_size) {
    std::cout << "[Rank " << rank << "] Testing Column-Parallel MatMul...\n";
    
    // Matrix dimensions
    int M = 4;  // Batch/sequence dimension
    int K = 8;  // Hidden dimension
    int N = 16; // Output dimension (will be sharded)
    
    // Create 1D DeviceMesh
    auto device_mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    
    // Create ProcessGroup
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
    // Create X (replicated)
    DTensor X(device_mesh, pg);
    Layout X_layout = Layout::replicated(device_mesh, {M, K});
    std::vector<float> X_data = randomData(M * K, 100);
    X.setData(X_data, X_layout);
    
    // Create W (column-sharded)
    DTensor W(device_mesh, pg);
    std::vector<std::shared_ptr<Placement>> W_placements = {
        std::make_shared<Shard>(1)  // Shard on dimension 1 (columns)
    };
    Layout W_layout(device_mesh, {K, N}, W_placements);
    
    // Each rank gets N/world_size columns
    int N_local = N / world_size;
    std::vector<float> W_data_local = randomData(K * N_local, 200 + rank);
    W.setData(W_data_local, W_layout);
    
    // Perform column-parallel matmul
    auto start = std::chrono::high_resolution_clock::now();
    DTensor Y = X.matmul(W);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify result shape
    auto Y_layout = Y.get_layout();
    auto Y_local_shape = Y_layout.get_local_shape(rank);
    
    if (Y_local_shape[0] == M && Y_local_shape[1] == N_local) {
        std::cout << "[Rank " << rank << "] ✓ Column-Parallel MatMul PASSED"
                  << " | Output shape: [" << Y_local_shape[0] << ", " << Y_local_shape[1] << "]"
                  << " | Time: " << duration.count() / 1000.0 << " ms"
                  << " | No communication needed ✅\n";
    } else {
        std::cout << "[Rank " << rank << "] ✗ Column-Parallel MatMul FAILED"
                  << " | Expected shape: [" << M << ", " << N_local << "]"
                  << " | Got: [" << Y_local_shape[0] << ", " << Y_local_shape[1] << "]\n";
    }
}

// Test row-parallel matmul
// X [M, K] (column-sharded) @ W [K, N] (row-sharded) -> Y [M, N] (replicated via AllReduce)
void test_row_parallel_matmul(int rank, int world_size) {
    std::cout << "[Rank " << rank << "] Testing Row-Parallel MatMul...\n";
    
    // Matrix dimensions
    int M = 4;
    int K = 16; // Will be sharded
    int N = 8;
    
    // Create 1D DeviceMesh
    auto device_mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    
    // Create ProcessGroup
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
    // Create X (column-sharded, dimension 1)
    DTensor X(device_mesh, pg);
    std::vector<std::shared_ptr<Placement>> X_placements = {
        std::make_shared<Shard>(1)  // Shard on dimension 1 (columns)
    };
    Layout X_layout(device_mesh, {M, K}, X_placements);
    int K_local = K / world_size;
    std::vector<float> X_data_local = randomData(M * K_local, 300 + rank);
    X.setData(X_data_local, X_layout);
    
    // Create W (row-sharded, dimension 0)
    DTensor W(device_mesh, pg);
    std::vector<std::shared_ptr<Placement>> W_placements = {
        std::make_shared<Shard>(0)  // Shard on dimension 0 (rows)
    };
    Layout W_layout(device_mesh, {K, N}, W_placements);
    std::vector<float> W_data_local = randomData(K_local * N, 400 + rank);
    W.setData(W_data_local, W_layout);
    
    // Perform row-parallel matmul (includes AllReduce)
    auto start = std::chrono::high_resolution_clock::now();
    DTensor Y = X.matmul(W);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Verify result is replicated and has correct shape
    auto Y_layout = Y.get_layout();
    auto Y_local_shape = Y_layout.get_local_shape(rank);
    bool is_replicated = Y_layout.is_fully_replicated();
    
    if (Y_local_shape[0] == M && Y_local_shape[1] == N && is_replicated) {
        std::cout << "[Rank " << rank << "] ✓ Row-Parallel MatMul PASSED"
                  << " | Output shape: [" << Y_local_shape[0] << ", " << Y_local_shape[1] << "]"
                  << " | Time: " << duration.count() / 1000.0 << " ms"
                  << " | Includes AllReduce ⚡\n";
    } else {
        std::cout << "[Rank " << rank << "] ✗ Row-Parallel MatMul FAILED"
                  << " | Expected replicated [" << M << ", " << N << "]"
                  << " | Got: [" << Y_local_shape[0] << ", " << Y_local_shape[1] << "]"
                  << " | Is replicated: " << (is_replicated ? "yes" : "no") << "\n";
    }
}

// Test MLP-style forward pass (column -> row pattern)
void test_mlp_forward(int rank, int world_size) {
    if (rank == 0) {
        std::cout << "\n=== Testing MLP Forward Pass (Column -> Row) ===\n";
    }
    
    int batch_size = 8;
    int hidden_dim = 32;
    int intermediate_dim = 64;
    
    // Create DeviceMesh and ProcessGroup
    auto device_mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
    // Input X [batch_size, hidden_dim] - replicated
    DTensor X(device_mesh, pg);
    Layout X_layout = Layout::replicated(device_mesh, {batch_size, hidden_dim});
    std::vector<float> X_data = randomData(batch_size * hidden_dim, 500);
    X.setData(X_data, X_layout);
    
    // --- Layer 1: Column-Parallel ---
    // W1 [hidden_dim, intermediate_dim] - column sharded
    DTensor W1(device_mesh, pg);
    std::vector<std::shared_ptr<Placement>> W1_placements = {
        std::make_shared<Shard>(1)
    };
    Layout W1_layout(device_mesh, {hidden_dim, intermediate_dim}, W1_placements);
    int intermediate_local = intermediate_dim / world_size;
    std::vector<float> W1_data = randomData(hidden_dim * intermediate_local, 600 + rank);
    W1.setData(W1_data, W1_layout);
    
    auto t1_start = std::chrono::high_resolution_clock::now();
    DTensor hidden = X.matmul(W1);  // [batch, intermediate_dim] column-sharded
    auto t1_end = std::chrono::high_resolution_clock::now();
    auto t1_dur = std::chrono::duration_cast<std::chrono::microseconds>(t1_end - t1_start);
    
    // --- Layer 2: Row-Parallel ---
    // W2 [intermediate_dim, hidden_dim] - row sharded
    DTensor W2(device_mesh, pg);
    std::vector<std::shared_ptr<Placement>> W2_placements = {
        std::make_shared<Shard>(0)
    };
    Layout W2_layout(device_mesh, {intermediate_dim, hidden_dim}, W2_placements);
    std::vector<float> W2_data = randomData(intermediate_local * hidden_dim, 700 + rank);
    W2.setData(W2_data, W2_layout);
    
    auto t2_start = std::chrono::high_resolution_clock::now();
    DTensor output = hidden.matmul(W2);  // [batch, hidden_dim] replicated (AllReduce)
    auto t2_end = std::chrono::high_resolution_clock::now();
    auto t2_dur = std::chrono::duration_cast<std::chrono::microseconds>(t2_end - t2_start);
    
    if (rank == 0) {
        std::cout << "✓ MLP Forward Pass Complete\n";
        std::cout << "  Layer 1 (Column-Parallel): " << t1_dur.count() / 1000.0 << " ms (no comm)\n";
        std::cout << "  Layer 2 (Row-Parallel): " << t2_dur.count() / 1000.0 << " ms (with AllReduce)\n";
        std::cout << "  Total: " << (t1_dur.count() + t2_dur.count()) / 1000.0 << " ms\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "This test requires exactly 2 GPUs. Run with: mpirun -np 2 ./test_matmul\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "Tensor Parallel MatMul Tests (2 GPUs)\n";
        std::cout << "========================================\n\n";
    }
    
    try {
        // Test 1: Column-Parallel MatMul
        test_column_parallel_matmul(rank, world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) std::cout << "\n";
        
        // Test 2: Row-Parallel MatMul
        test_row_parallel_matmul(rank, world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Test 3: Full MLP Forward Pass
        test_mlp_forward(rank, world_size);
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "\n========================================\n";
            std::cout << "All MatMul tests completed!\n";
            std::cout << "========================================\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] Error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Finalize();
    return 0;
}
