
#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>
#include <chrono>
#include <iomanip>

// Timing helper function
double getElapsedTime(const std::chrono::high_resolution_clock::time_point& start,
                      const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Test configuration structure
struct BenchmarkConfig {
    int batch_size;
    int seq_len;
    int hidden_dim;
    int ffn_dim;
    int warmup_iters;
    int bench_iters;
};

// Run single-GPU MLP benchmark (no tensor parallelism)
void benchmark_single_gpu(const BenchmarkConfig& config, int rank) {
    if (rank != 0) return;  // Only run on rank 0
    
    const int B = config.batch_size;
    const int T = config.seq_len;
    const int C = config.hidden_dim;
    const int F = config.ffn_dim;
    
    // Merge batch and sequence length into one dimension for simplicity
    const int BT = B * T;
    
    cudaSetDevice(0);
    
    // Create tensors using OwnTensor library directly
    OwnTensor::TensorOptions opts;
    opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, 0))
               .with_dtype(OwnTensor::Dtype::Float32);
    
    // Create input tensor [BT, C]
    OwnTensor::Shape x_shape;
    x_shape.dims = {BT, C};
    OwnTensor::Tensor X(x_shape, opts);
    std::vector<float> x_data(BT * C);
    for (int i = 0; i < BT * C; i++) x_data[i] = 0.01f * (i % 100 + 1);
    X.set_data(x_data);
    
    // Create W1 weight [C, F]
    OwnTensor::Shape w1_shape;
    w1_shape.dims = {C, F};
    OwnTensor::Tensor W1(w1_shape, opts);
    std::vector<float> w1_data(C * F);
    for (int i = 0; i < C * F; i++) w1_data[i] = 0.01f * (i % F + 1);
    W1.set_data(w1_data);
    
    // Create W2 weight [F, C]
    OwnTensor::Shape w2_shape;
    w2_shape.dims = {F, C};
    OwnTensor::Tensor W2(w2_shape, opts);
    std::vector<float> w2_data(F * C);
    for (int i = 0; i < F * C; i++) w2_data[i] = 0.02f;
    W2.set_data(w2_data);
    
    // Warmup iterations
    for (int i = 0; i < config.warmup_iters; i++) {
        OwnTensor::Tensor H = TensorOpsBridge::matmul(X, W1);   // [BT, C] x [C, F] = [BT, F]
        OwnTensor::Tensor Y = TensorOpsBridge::matmul(H, W2);   // [BT, F] x [F, C] = [BT, C]
    }
    cudaDeviceSynchronize();
    
    // Benchmark iterations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < config.bench_iters; i++) {
        OwnTensor::Tensor H = TensorOpsBridge::matmul(X, W1);
        OwnTensor::Tensor Y = TensorOpsBridge::matmul(H, W2);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time_ms = getElapsedTime(start, end);
    double avg_time_ms = total_time_ms / config.bench_iters;
    double throughput = (config.bench_iters * B) / (total_time_ms / 1000.0);  // samples/sec
    
    // Calculate total parameters: W1[C,F] + W2[F,C] = 2*C*F
    double params_millions = (2.0 * config.hidden_dim * config.ffn_dim) / 1e6;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(12) << config.hidden_dim
              << std::setw(12) << config.ffn_dim
              << std::setw(15) << params_millions
              << std::setw(15) << avg_time_ms 
              << std::setw(18) << throughput << std::endl;
}

// Run 2-GPU MLP benchmark with tensor parallelism
void benchmark_tensor_parallel(const BenchmarkConfig& config, int rank, int world_size) {
    const int B = config.batch_size;
    const int T = config.seq_len;
    const int C = config.hidden_dim;
    const int F = config.ffn_dim;
    
    // Merge batch and sequence length into one dimension for simplicity
    const int BT = B * T;
    
    auto device_mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    auto pg = init_process_group(world_size, rank);
    
    // Create input tensor [BT, C] (replicated across both GPUs)
    Layout x_layout = Layout::replicated(*device_mesh, {BT, C});
    DTensor X(device_mesh, pg);
    std::vector<float> x_data;
    
    if (rank == 0) {
        x_data.resize(BT * C);
        for (int i = 0; i < BT * C; i++) x_data[i] = 0.01f * (i % 100 + 1);
    }
    X.setDataFromRoot(x_data, x_layout, 0);  // Broadcast from root during initialization
    
    // Create W1 weight [C, F] (column-parallel: sharded on output dimension)
    DTensor W1(device_mesh, pg);
    std::vector<float> w1_full_data;
    if (rank == 0) {
        w1_full_data.resize(C * F);
        for (int i = 0; i < C * F; i++) w1_full_data[i] = 0.01f * (i % F + 1);
    }
    Layout w1_layout = Layout::replicated(*device_mesh, {C, F});  // Start as replicated
    W1.setDataFromRoot(w1_full_data, w1_layout, 0);             // Load on root, broadcast
    W1.shard(1, 0);  // Shard on dimension 1 (output dim: F)
    
    // Create W2 weight [F, C] (row-parallel: sharded on input dimension)
    DTensor W2(device_mesh, pg);
    std::vector<float> w2_full_data;
    if (rank == 0) {
        w2_full_data.resize(F * C);
        for (int i = 0; i < F * C; i++) w2_full_data[i] = 0.02f;
    }
    Layout w2_layout = Layout::replicated(*device_mesh, {F, C});  // Start as replicated
    W2.setDataFromRoot(w2_full_data, w2_layout, 0);             // Load on root, broadcast
    W2.shard(0, 0);  // Shard on dimension 0 (input dim: F)
    
    // Warmup iterations
    for (int i = 0; i < config.warmup_iters; i++) {
        DTensor H = X.matmul(W1);  // Column-parallel matmul (GELU removed as not implementing)
        DTensor Y = H.matmul(W2);  // [BT, F/2] x [F/2, C] = [BT, C] (row-parallel)
        Y.sync();  // AllReduce to gather results
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Benchmark iterations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < config.bench_iters; i++) {
            DTensor H = X.matmul(W1);  // Column-parallel matmul (GELU removed as not implementing)
        DTensor Y = H.matmul(W2);  // Row-parallel matmul
        Y.sync();  // AllReduce to gather results
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time_ms = getElapsedTime(start, end);
    double avg_time_ms = total_time_ms / config.bench_iters;
    double throughput = (config.bench_iters * B) / (total_time_ms / 1000.0);  // samples/sec
    
    // Calculate total parameters: W1[C,F] + W2[F,C] = 2*C*F
    double params_millions = (2.0 * config.hidden_dim * config.ffn_dim) / 1e6;
    
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(12) << config.hidden_dim
                  << std::setw(12) << config.ffn_dim
                  << std::setw(15) << params_millions
                  << std::setw(15) << avg_time_ms 
                  << std::setw(18) << throughput << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Set CUDA device based on rank
    cudaSetDevice(rank % 2);
    
    // Test different hidden dimensions to show when TP becomes beneficial
    std::vector<int> hidden_dims = {768, 1024, 2048, 4096};
    const int FIXED_BATCH = 16;  // Fixed batch size
    
    if (rank == 0) {
        std::cout << "\nMLP Benchmark: Single GPU vs 2-GPU TP" << std::endl;
        std::cout << "Batch size: " << FIXED_BATCH << std::endl;
        
        if (world_size == 1) {
            std::cout << "--- Single GPU (No Tensor Parallelism) ---" << std::endl;
        } else {
            std::cout << "--- 2-GPU Tensor Parallel ---" << std::endl;
        }
        
        std::cout << std::setw(12) << "Hidden Dim" 
                  << std::setw(12) << "FFN Dim"
                  << std::setw(15) << "Params (M)"
                  << std::setw(15) << "Latency (ms)" 
                  << std::setw(18) << "Throughput (samples/s)" << std::endl;
        std::cout << std::string(72, '-') << std::endl;
    }
    
    for (int hidden_dim : hidden_dims) {
        BenchmarkConfig config;
        config.seq_len = 512;
        config.hidden_dim = hidden_dim;
        config.ffn_dim = hidden_dim * 4;
        config.batch_size = FIXED_BATCH;
        config.warmup_iters = 3;
        config.bench_iters = 10;
        
        if (world_size == 1) {
            benchmark_single_gpu(config, rank);
        } else if (world_size == 2) {
            benchmark_tensor_parallel(config, rank, world_size);
        } else {
            if (rank == 0) {
                std::cerr << "Error: This benchmark requires either 1 or 2 GPUs" << std::endl;
                std::cerr << "Current world_size: " << world_size << std::endl;
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        if (world_size == 1) {
            std::cout << "\nSingle GPU benchmark completed" << std::endl;
            //std::cout << "To run 2-GPU TP: mpirun -np 2 --allow-run-as-root ./test_mlp_benchmark" << std::endl;
        } else {
            std::cout << "\n2-GPU Tensor Parallel benchmark completed" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}


// All execute together - single launch overhead
// Total overhead: ~5-10μs (vs 20-40μs before)


/*
 * ============================================================================
 * BUILD & RUN INSTRUCTIONS
 * ============================================================================
 * 
 * From DTensor_v2.0 directory:
 * 
 *   make lib                  # Build library (if needed)
 *   make test_mlp_benchmark   # Build this test
 *   mpirun -np 2 ./tests/test_mlp_benchmark
 * 
 * ============================================================================
 */
