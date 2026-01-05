
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include <cmath>

#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "memory/cachingAllocator.hpp"

using namespace OwnTensor;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// =============================================================================
// Arithmetic Intensity Result
// =============================================================================
struct AIResult {
    int batch_size;
    int hidden_dim;
    
    // MatMul 1 & 2
    double mm1_ai;
    double mm2_ai;
    
    // AllReduce bytes
    double ar_bytes;
    
    // Total MLP
    double total_flops;
    double total_bytes;
    double total_ai;
    
    // Boundedness checks
    double gpu_mem_threshold;    // Peak FLOPS / GPU Memory BW
    double interconnect_threshold; // Peak FLOPS / Interconnect BW
    std::string gpu_mem_bound;   // Compute or Memory (GPU)
    std::string interconnect_bound; // Compute or Comm (PCIe/NVLink)
};

// =============================================================================
// Calculate Arithmetic Intensity
// =============================================================================
AIResult calculate_ai(
    int batch_size, int hidden_dim, int world_size,
    double peak_tflops, double gpu_mem_bw, double interconnect_bw) {
    
    const int BT = batch_size * 512;
    const int C = hidden_dim;
    const int F = hidden_dim * 4;
    const int F_local = F / world_size;
    
    AIResult r;
    r.batch_size = batch_size;
    r.hidden_dim = hidden_dim;
    
    // MatMul 1: [BT, C] x [C, F_local] = [BT, F_local]
    double mm1_flops = 2.0 * BT * C * F_local;
    double mm1_bytes = (double)(BT * C + C * F_local + BT * F_local) * 4.0;
    r.mm1_ai = mm1_flops / mm1_bytes;
    
    // MatMul 2: [BT, F_local] x [F_local, C] = [BT, C]
    double mm2_flops = 2.0 * BT * F_local * C;
    double mm2_bytes = (double)(BT * F_local + F_local * C + BT * C) * 4.0;
    r.mm2_ai = mm2_flops / mm2_bytes;
    
    // AllReduce: [BT, C] - bidirectional
    r.ar_bytes = 2.0 * BT * C * 4.0;
    
    // Total
    r.total_flops = mm1_flops + mm2_flops;
    r.total_bytes = mm1_bytes + mm2_bytes + r.ar_bytes;
    r.total_ai = r.total_flops / r.total_bytes;
    
    // GPU Memory threshold
    r.gpu_mem_threshold = (peak_tflops * 1e12) / (gpu_mem_bw * 1e9);
    r.gpu_mem_bound = (r.total_ai < r.gpu_mem_threshold) ? "Mem-bound" : "Compute";
    
    // Interconnect threshold (for AllReduce specifically)
    // FLOPS that can be done while AllReduce happens
    r.interconnect_threshold = (peak_tflops * 1e12) / (interconnect_bw * 1e9);
    
    // Communication AI = FLOPS / AllReduce bytes
    double comm_ai = r.total_flops / r.ar_bytes;
    r.interconnect_bound = (comm_ai < r.interconnect_threshold) ? "Comm-bound" : "Overlap OK";
    
    return r;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_size != 2) {
        if (rank == 0) std::cerr << "Requires TP=2\n";
        MPI_Finalize();
        return 1;
    }
    
    int device_id = rank % 2;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    // Hardware specs
    double peak_tflops = 12.7;      // FP32 peak TFLOPS
    double gpu_mem_bw = 360.0;      // GPU memory bandwidth GB/s
    double interconnect_bw = 7.0;   // PCIe bidirectional GB/s (measured)
    
    if (rank == 0) {
        std::cout << "\n=== Arithmetic Intensity (TP=2, " << prop.name << ") ===\n";
        std::cout << "Thresholds: GPU Mem=" << std::fixed << std::setprecision(0) 
                  << (peak_tflops * 1e12) / (gpu_mem_bw * 1e9) 
                  << ", PCIe=" << (peak_tflops * 1e12) / (interconnect_bw * 1e9) << " FLOPS/byte\n\n";
        std::cout << std::setw(6) << "Batch" << std::setw(8) << "Hidden"
                  << std::setw(8) << "AI" << std::setw(12) << "GPU"
                  << std::setw(12) << "Comm\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::vector<std::pair<int, int>> configs = {
        {1, 768}, {4, 768}, {8, 2048}, {16, 4096}, {32, 4096}
    };
    
    for (auto& [bs, hdim] : configs) {
        auto r = calculate_ai(bs, hdim, world_size, peak_tflops, gpu_mem_bw, interconnect_bw);
        
        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(0);
            std::cout << std::setw(6) << r.batch_size
                      << std::setw(8) << r.hidden_dim
                      << std::setw(8) << r.total_ai
                      << std::setw(12) << r.gpu_mem_bound
                      << std::setw(12) << r.interconnect_bound << "\n";
        }
    }
    
    if (rank == 0) {
        std::cout << "\n[PASS] Arithmetic intensity analysis complete.\n";
    }
    
    MPI_Finalize();
    return 0;
}

// make test_mlp_arithmetic_intensity
// mpirun -np 2 --allow-run-as-root ./test_mlp_arithmetic_intensity


