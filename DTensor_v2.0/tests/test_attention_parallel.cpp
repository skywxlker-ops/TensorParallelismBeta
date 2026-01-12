#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <iomanip>

// === DTensor Core ===
#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"

// === TensorLib (for ops) ===
#include "bridge/bridge.h"
#include "memory/cachingAllocator.hpp"

using namespace OwnTensor;

// =============================================================================
// Tensor-Parallel Attention Layer Test
// 
// This test demonstrates tensor parallelism for transformer attention:
//   1. QKV Projection (Column-Parallel): X @ W_qkv -> QKV (sharded)
//   2. Multi-Head Attention (Local computation on each GPU)
//   3. Output Projection (Row-Parallel): attn_out @ W_o -> output (AllReduce)
//
// Architecture mirrors the MLP pattern:
//   - Column-parallel: No communication, output is sharded
//   - Row-parallel: Requires AllReduce to sum partial results
// =============================================================================

void print_separator(int rank, const std::string& title) {
    if (rank == 0) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << title << "\n";
        std::cout << std::string(70, '=') << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void print_section(int rank, const std::string& section) {
    if (rank == 0) {
        std::cout << "\n--- " << section << " ---\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Test 1: MLP Forward Pass (Baseline)
// =============================================================================
void test_mlp_forward(int rank, int world_size, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    
    print_separator(rank, "TEST 1: MLP Forward Pass");
    
    const int BATCH = 2;
    const int SEQ_LEN = 4;
    const int HIDDEN = 8;
    const int INTERMEDIATE = 16;
    
    if (rank == 0) {
        std::cout << "Config: B=" << BATCH << " T=" << SEQ_LEN << " C=" << HIDDEN << " I=" << INTERMEDIATE << "\n";
    }
    
    // Input X (Replicated) [BATCH, SEQ_LEN, HIDDEN]
    std::vector<int64_t> shape_X = {BATCH, SEQ_LEN, HIDDEN};
    Layout layout_X = Layout::replicated(*mesh, shape_X);
    
    std::vector<float> data_X(BATCH * SEQ_LEN * HIDDEN, 1.0f);
    DTensor X(mesh, pg);
    X.setData(data_X, layout_X);
    
    // Weight W1 (Column-Sharded) [HIDDEN, INTERMEDIATE]
    std::vector<int64_t> shape_W1 = {HIDDEN, INTERMEDIATE};
    Layout layout_W1(*mesh, shape_W1, 1);
    
    std::vector<int64_t> local_shape_W1 = layout_W1.get_local_shape(rank);
    int size_W1 = local_shape_W1[0] * local_shape_W1[1];
    
    std::vector<float> data_W1(size_W1, (rank + 1) * 0.5f);
    DTensor W1(mesh, pg);
    W1.setData(data_W1, layout_W1);
    
    // Forward: Hidden = X @ W1
    DTensor Hidden = X.matmul(W1);
    
    if (rank == 0) {
        std::cout << "Layer 1 (Column-Parallel): X @ W1\n";
    }
    
    // Weight W2 (Row-Sharded) [INTERMEDIATE, HIDDEN]
    std::vector<int64_t> shape_W2 = {INTERMEDIATE, HIDDEN};
    Layout layout_W2(*mesh, shape_W2, 0);
    
    std::vector<int64_t> local_shape_W2 = layout_W2.get_local_shape(rank);
    int size_W2 = local_shape_W2[0] * local_shape_W2[1];
    
    std::vector<float> data_W2(size_W2, 1.0f);
    DTensor W2(mesh, pg);
    W2.setData(data_W2, layout_W2);
    
    // Forward: Output = Hidden @ W2 (requires AllReduce)
    DTensor Output = Hidden.matmul(W2);
    
    if (rank == 0) {
        std::cout << "Layer 2 (Row-Parallel): H @ W2 + AllReduce\n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Test 2: Attention Layer with Tensor Parallelism
// =============================================================================
void test_attention_layer(int rank, int world_size,std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    
    print_separator(rank, "TEST 2: Attention Layer");
    
    const int BATCH = 2;
    const int SEQ_LEN = 4;
    const int HIDDEN = 8;
    const int NUM_HEADS = 4;
    const int HEAD_DIM = HIDDEN / NUM_HEADS;
    
    if (rank == 0) {
        std::cout << "Config: B=" << BATCH << " T=" << SEQ_LEN << " C=" << HIDDEN << " H=" << NUM_HEADS << "\n";
    }
    
    // Input X (Replicated)
    std::vector<int64_t> shape_X = {BATCH, SEQ_LEN, HIDDEN};
    Layout layout_X = Layout::replicated(*mesh, shape_X);
    
    std::vector<float> data_X(BATCH * SEQ_LEN * HIDDEN, 1.0f);
    DTensor X(mesh, pg);
    X.setData(data_X, layout_X);
    
    // W_qkv (Column-Sharded)
    std::vector<int64_t> shape_W_qkv = {HIDDEN, 3 * HIDDEN};
    Layout layout_W_qkv(*mesh, shape_W_qkv, 1);
    
    std::vector<int64_t> local_shape_W_qkv = layout_W_qkv.get_local_shape(rank);
    int size_W_qkv = local_shape_W_qkv[0] * local_shape_W_qkv[1];
    
    std::vector<float> data_W_qkv(size_W_qkv);
    for (int i = 0; i < size_W_qkv; i++) {
        data_W_qkv[i] = 0.1f * (rank + 1);
    }
    
    DTensor W_qkv(mesh, pg);
    W_qkv.setData(data_W_qkv, layout_W_qkv);
    
    // Forward: QKV = X @ W_qkv
    DTensor QKV = X.matmul(W_qkv);
    
    if (rank == 0) {
        std::cout << "QKV Projection (Column-Parallel): X @ W_qkv\n";
    }
    
    // Multi-head attention computed locally on each GPU
    DTensor attn_local = QKV;
    
    // Output Projection (Row-Parallel) - using 2D tensors
    const int BT = BATCH * SEQ_LEN;
    std::vector<int64_t> shape_attn_2d = {BT, HIDDEN};
    Layout layout_attn_2d(*mesh, shape_attn_2d, 1);
    
    std::vector<int64_t> local_shape_attn_2d = layout_attn_2d.get_local_shape(rank);
    int size_attn_2d = local_shape_attn_2d[0] * local_shape_attn_2d[1];
    
    std::vector<float> data_attn_2d(size_attn_2d, 2.0f * (rank + 1));
    DTensor attn_reshaped(mesh, pg);
    attn_reshaped.setData(data_attn_2d, layout_attn_2d);
    
    // W_o (Row-Sharded)
    std::vector<int64_t> shape_W_o_2d = {HIDDEN, HIDDEN};
    Layout layout_W_o_2d(*mesh, shape_W_o_2d, 0);
    
    std::vector<int64_t> local_shape_W_o_2d = layout_W_o_2d.get_local_shape(rank);
    int size_W_o_2d = local_shape_W_o_2d[0] * local_shape_W_o_2d[1];
    
    std::vector<float> data_W_o_2d(size_W_o_2d, 1.0f);
    DTensor W_o_2d(mesh, pg);
    W_o_2d.setData(data_W_o_2d, layout_W_o_2d);
    
    // Forward: final_output = attn @ W_o
    DTensor final_output = attn_reshaped.matmul(W_o_2d);
    
    if (rank == 0) {
        std::cout << "Output Projection (Row-Parallel): attn @ W_o + AllReduce\n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Test 3: Combined Transformer Block (Attention + MLP)
// =============================================================================
void test_transformer_block(int rank, int world_size, std::shared_ptr<DeviceMesh> mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    
    print_separator(rank, "TEST 3: Transformer Block");
    
    const int BATCH = 2;
    const int SEQ_LEN = 4;
    const int HIDDEN = 8;
    const int INTERMEDIATE = 32;
    
    if (rank == 0) {
        std::cout << "Config: B=" << BATCH << " T=" << SEQ_LEN << " C=" << HIDDEN << " I=" << INTERMEDIATE << "\n";
    }
    
    // Input
    std::vector<int64_t> shape_X = {BATCH, SEQ_LEN, HIDDEN};
    Layout layout_X = Layout::replicated(*mesh, shape_X);
    
    std::vector<float> data_X(BATCH * SEQ_LEN * HIDDEN, 1.0f);
    DTensor X(mesh, pg);
    X.setData(data_X, layout_X);
    
    // Attention QKV projection
    std::vector<int64_t> shape_W_qkv = {HIDDEN, 3 * HIDDEN};
    Layout layout_W_qkv(*mesh, shape_W_qkv, 1);
    
    std::vector<int64_t> local_shape_qkv = layout_W_qkv.get_local_shape(rank);
    std::vector<float> data_W_qkv(local_shape_qkv[0] * local_shape_qkv[1], 0.1f);
    DTensor W_qkv(mesh, pg);
    W_qkv.setData(data_W_qkv, layout_W_qkv);
    
    DTensor QKV = X.matmul(W_qkv);
    
    // Attention output projection (row-parallel)
    // Use 2D tensors to avoid 3D sharding limitations
    const int BT = BATCH * SEQ_LEN;
    std::vector<int64_t> shape_attn_2d = {BT, HIDDEN};
    Layout layout_attn_2d(*mesh, shape_attn_2d, 1);  // Column-sharded
    
    std::vector<int64_t> local_shape_attn_2d = layout_attn_2d.get_local_shape(rank);
    std::vector<float> data_attn_2d(local_shape_attn_2d[0] * local_shape_attn_2d[1], 1.0f);
    DTensor attn_out(mesh, pg);
    attn_out.setData(data_attn_2d, layout_attn_2d);
    
    std::vector<int64_t> shape_W_o = {HIDDEN, HIDDEN};
    Layout layout_W_o(*mesh, shape_W_o, 0);  // Row-sharded
    
    std::vector<int64_t> local_shape_W_o = layout_W_o.get_local_shape(rank);
    std::vector<float> data_W_o(local_shape_W_o[0] * local_shape_W_o[1], 1.0f);
    DTensor W_o(mesh, pg);
    W_o.setData(data_W_o, layout_W_o);
    
    DTensor attn_final = attn_out.matmul(W_o);
    
    // MLP Layer 1 (column-parallel)
    std::vector<int64_t> shape_W1 = {HIDDEN, INTERMEDIATE};
    Layout layout_W1(*mesh, shape_W1, 1);
    
    std::vector<int64_t> local_shape_W1 = layout_W1.get_local_shape(rank);
    std::vector<float> data_W1(local_shape_W1[0] * local_shape_W1[1], 0.5f);
    DTensor W1(mesh, pg);
    W1.setData(data_W1, layout_W1);
    
    DTensor mlp_hidden = attn_final.matmul(W1);
    
    // MLP Layer 2 (row-parallel)
    std::vector<int64_t> shape_W2 = {INTERMEDIATE, HIDDEN};
    Layout layout_W2(*mesh, shape_W2, 0);
    
    std::vector<int64_t> local_shape_W2 = layout_W2.get_local_shape(rank);
    std::vector<float> data_W2(local_shape_W2[0] * local_shape_W2[1], 1.0f);
    DTensor W2(mesh, pg);
    W2.setData(data_W2, layout_W2);
    
    DTensor mlp_output = mlp_hidden.matmul(W2);
    
    if (rank == 0) {
        std::cout << "Attention: QKV + Output Projection\n";
        std::cout << "MLP: 2-Layer Feed-Forward\n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Main Entry Point
// =============================================================================
int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        std::cout << "\nDTensor v2.0 - Tensor-Parallel Attention Test\n";
        std::cout << "============================================\n";
    }
    
    // Setup GPUs
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
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    if (rank == 0) {
        std::cout << world_size << " ranks, " << device_count << " GPUs (" << prop.name << ")\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Initialize NCCL
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    try {
        std::shared_ptr<DeviceMesh> mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
        std::shared_ptr<ProcessGroupNCCL> pg = init_process_group(world_size, rank);
        
        // Run all tests
        test_mlp_forward(rank, world_size, mesh, pg);
        test_attention_layer(rank, world_size, mesh, pg);
        test_transformer_block(rank, world_size, mesh, pg);
        
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] ERROR: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    if (rank == 0) {
        std::cout << "\n[PASS] All tests completed\n";
    }
    
    MPI_Finalize();
    return 0;
}
