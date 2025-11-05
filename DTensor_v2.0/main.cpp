#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <filesystem>

// === TensorLib Core ===
#include "TensorLib.h"
#include "memory/cachingAllocator.hpp"

#include "dtensor.h"
#include "process_group.h"
#include "planner.h"
#include "ckpt.h"

using namespace OwnTensor;

void worker(int rank, int world_size, const ncclUniqueId& id,
            int global_size,
            bool show_mesh, bool show_desc, bool save_ckpt, bool load_ckpt) {

    cudaSetDevice(rank);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ProcessGroup pg(rank, world_size, rank, id);
    DTensor tensor(rank, world_size, &pg);

    std::string ckpt_dir = "checkpoints";
    std::filesystem::create_directories(ckpt_dir);

    // ------------------------------------------------------------
    // Mesh or Layout Display (Meta Inspection)
    // ------------------------------------------------------------
    if (show_mesh) {
        Planner::printLayoutJSON(rank, world_size, global_size);
        cudaStreamDestroy(stream);
        return;
    }

    if (show_desc) {
        auto layout = Planner::inferLayout(global_size, world_size);
        if (rank == 0) std::cout << Planner::describePlan(layout);
        cudaStreamDestroy(stream);
        return;
    }

    // ------------------------------------------------------------
    // Tensor Data Initialization
    // ------------------------------------------------------------
    std::vector<float> local_data(global_size);

    if (load_ckpt) {
        auto [data, meta] = Checkpoint::loadLatest(ckpt_dir, rank);
        if (!data.empty()) {
            tensor.setData(data);
        } else {
            std::cerr << "[Checkpoint] Rank " << rank
                      << " had no valid checkpoint, initializing random data.\n";
            srand(time(nullptr) + rank);
            for (int i = 0; i < global_size; ++i)
                local_data[i] = static_cast<float>(rand() % 1000) / 100.0f;
            tensor.setData(local_data);
        }
    } else {
        srand(time(nullptr) + rank);
        for (int i = 0; i < global_size; ++i)
            local_data[i] = static_cast<float>(rand() % 1000) / 100.0f;

        if (rank == 0) {
            std::cout << "[DEBUG] Rank " << rank
                      << " first 10 values before upload: ";
            for (int i = 0; i < std::min(10, global_size); ++i)
                std::cout << local_data[i] << " ";
            std::cout << "\n";
        }

        tensor.setData(local_data);
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // Tensor Metadata Debug Info (TensorLib Integration Verification)
    // ------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n[TensorLib Integration Check]\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Each rank prints its tensor properties
    std::cout << "[Rank " << rank << "] Tensor metadata:"
              << " device=" << (tensor.getData().empty() ? "Unknown" : "CUDA")
              << ", dtype=float32"
              << ", elements=" << global_size
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // Distributed Computation: AllReduce Test
    // ------------------------------------------------------------
    if (rank == 0) std::cout << "\n==== BEFORE ALLREDUCE ====\n";
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[Rank " << rank << "] ";
    tensor.print();

    tensor.allReduce();

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) std::cout << "\n==== AFTER ALLREDUCE ====\n";
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[Rank " << rank << "] ";
    tensor.print();

    // ------------------------------------------------------------
    // Optional Checkpoint Save
    // ------------------------------------------------------------
    if (save_ckpt) {
        Checkpoint::save(ckpt_dir, tensor.getData(), rank);
    }

    cudaStreamDestroy(stream);
}

// ===============================================================
// Main Entry
// ===============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Updated header line for TensorLib integration
    if (rank == 0)
        std::cout << "[Init] Using DTensor (TensorLib backend + CUDA + NCCL)\n";

    cudaFree(0); // Force CUDA context init

    bool show_mesh = false;
    bool show_desc = false;
    bool save_ckpt = false;
    bool load_ckpt = false;
    int global_size = 8; // Default tensor length

    // ------------------------------------------------------------
    // Parse CLI Arguments
    // ------------------------------------------------------------
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--mesh") == 0)
            show_mesh = true;
        else if (std::strcmp(arg, "--describe") == 0)
            show_desc = true;
        else if (std::strcmp(arg, "--save") == 0)
            save_ckpt = true;
        else if (std::strcmp(arg, "--load") == 0)
            load_ckpt = true;
        else if (std::strncmp(arg, "--len=", 6) == 0)
            global_size = std::atoi(arg + 6);
        else if (std::strncmp(arg, "--size=", 7) == 0)
            global_size = std::atoi(arg + 7);
        else if (std::strncmp(arg, "--", 2) == 0) {
            int val = std::atoi(arg + 2); // shorthand --1024
            if (val > 0)
                global_size = val;
        }
    }

    if (global_size <= 0) {
        if (rank == 0)
            std::cerr << "[Warning] Invalid tensor size. Using default: 8\n";
        global_size = 8;
    }

    if (rank == 0)
        std::cout << "[Config] Tensor length set to " << global_size << "\n";

    // ------------------------------------------------------------
    // Initialize NCCL and Worker
    // ------------------------------------------------------------
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    worker(rank, world_size, id, global_size, show_mesh, show_desc, save_ckpt, load_ckpt);

    // ------------------------------------------------------------
    // Finalize and Report Allocator Stats
    // ------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n=== Allocator Stats ===" << std::endl;
        gAllocator.printStats();
    }

    MPI_Finalize();
    return 0;
}

