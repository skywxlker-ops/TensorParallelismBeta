// #include <mpi.h>
// #include <nccl.h>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>
// #include <memory>
// #include <cstring>
// #include <cstdlib>
// #include <ctime>
// #include <filesystem>

// // === TensorLib Core ===
// #include "TensorLib.h"
// #include "memory/cachingAllocator.hpp"

// #include "dtensor.h"
// #include "process_group.h"
// #include "planner.h"
// #include "ckpt.h"
// #include "bridge/tensor_ops_bridge.h"

// using namespace OwnTensor;

// // =============================================================
// // Worker Function
// // =============================================================
// void worker(int rank, int world_size, const ncclUniqueId& id,
//             int rows, int cols,
//             bool show_mesh, bool show_desc, bool save_ckpt, bool load_ckpt) {

//     cudaSetDevice(rank);
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     ProcessGroup pg(rank, world_size, rank, id);
//     DTensor tensor(rank, world_size, &pg);

//     std::string ckpt_dir = "checkpoints";
//     std::filesystem::create_directories(ckpt_dir);

//     int total_elems = rows * cols;
//     std::vector<float> local_data(total_elems);

//     // ------------------------------------------------------------
//     // Checkpoint Load or Random Init
//     // ------------------------------------------------------------
//     if (load_ckpt) {
//         std::string path = ckpt_dir + "/tensor_rank" + std::to_string(rank) + ".bin";
//         tensor.loadCheckpoint(path);
//         std::cout << "[Rank " << rank << "] Loaded checkpoint from " << path << std::endl;
//     } else {
//         srand(time(nullptr) + rank * 77);
//         for (int i = 0; i < total_elems; ++i)
//             local_data[i] = static_cast<float>(rand() % 1000) / 100.0f;

//         tensor.setData(local_data);
//         tensor = tensor.reshape(rows, cols);
//     }

//     cudaDeviceSynchronize();
//     MPI_Barrier(MPI_COMM_WORLD);

//     if (rank == 0) {
//         std::cout << "\n[TensorLib Integration Check]\n";
//         std::cout << "[Rank 0] Tensor metadata: device=CUDA, dtype=float32, shape="
//                   << rows << "x" << cols << "\n";
//     }

//     MPI_Barrier(MPI_COMM_WORLD);

//     // ------------------------------------------------------------
//     // AllReduce Test on 2D Tensor
//     // ------------------------------------------------------------
//     if (rank == 0) std::cout << "\n==== BEFORE ALLREDUCE (2D Tensor) ====\n";
//     MPI_Barrier(MPI_COMM_WORLD);
//     tensor.print();

//     tensor.allReduce();

//     cudaDeviceSynchronize();
//     MPI_Barrier(MPI_COMM_WORLD);

//     if (rank == 0) std::cout << "\n==== AFTER ALLREDUCE (2D Tensor) ====\n";
//     tensor.print();

//     // ------------------------------------------------------------
//     // TensorOpsBridge Functional Tests
//     // ------------------------------------------------------------
//     if (rank == 0) std::cout << "\n==== TensorOpsBridge Functional Tests (Post-AllReduce) ====\n";

//     DTensor other(rank, world_size, &pg);
//     std::vector<float> local_data_B(total_elems);
//     srand(time(nullptr) + rank * 101);
//     for (int i = 0; i < total_elems; ++i)
//         local_data_B[i] = static_cast<float>(rand() % 1000) / 100.0f;
//     other.setData(local_data_B);
//     other = other.reshape(rows, cols);

//     DTensor add_res = tensor.add(other);
//     DTensor sub_res = tensor.sub(other);
//     DTensor mul_res = tensor.mul(other);
//     DTensor div_res = tensor.div(other);
//     DTensor matmul_res = tensor.matmul(other);

//     if (rank == 0) {
//         std::cout << "\nA:\n";
//         tensor.print();
//         std::cout << "B:\n";
//         other.print();

//         std::cout << "A + B:\n";
//         add_res.print();
//         std::cout << "A - B:\n";
//         sub_res.print();
//         std::cout << "A * B:\n";
//         mul_res.print();
//         std::cout << "A / B:\n";
//         div_res.print();
//         std::cout << "A x B (matmul):\n";
//         matmul_res.print();
//     }

//     // ------------------------------------------------------------
//     // Save Checkpoint (optional)
//     // ------------------------------------------------------------
//     if (save_ckpt) {
//         std::string path = ckpt_dir + "/tensor_rank" + std::to_string(rank) + ".bin";
//         tensor.saveCheckpoint(path);
//     }

//     cudaStreamDestroy(stream);
// }

// // =============================================================
// // Main Entry
// // =============================================================
// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);
//     int world_size, rank;
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     if (rank == 0)
//         std::cout << "[Init] Using DTensor (TensorLib backend + CUDA + NCCL)\n";

//     cudaFree(0); // Force CUDA context init

//     // Defaults
//     bool show_mesh = false;
//     bool show_desc = false;
//     bool save_ckpt = false;
//     bool load_ckpt = false;
//     int rows = 2, cols = 2;

//     // ------------------------------------------------------------
//     // CLI Argument Parsing
//     // ------------------------------------------------------------
//     for (int i = 1; i < argc; ++i) {
//         const char* arg = argv[i];
//         if (strcmp(arg, "--mesh") == 0)
//             show_mesh = true;
//         else if (strcmp(arg, "--describe") == 0)
//             show_desc = true;
//         else if (strcmp(arg, "--save") == 0)
//             save_ckpt = true;
//         else if (strcmp(arg, "--load") == 0)
//             load_ckpt = true;
//         else if (strncmp(arg, "--rows=", 7) == 0)
//             rows = atoi(arg + 7);
//         else if (strncmp(arg, "--cols=", 7) == 0)
//             cols = atoi(arg + 7);
//         else if (strncmp(arg, "--len=", 6) == 0) {
//             int len = atoi(arg + 6);
//             rows = 1;
//             cols = len;
//         }
//     }

//     if (rank == 0)
//         std::cout << "[Config] Tensor shape = " << rows << "x" << cols << "\n";

//     // ------------------------------------------------------------
//     // Initialize NCCL and Worker
//     // ------------------------------------------------------------
//     ncclUniqueId id;
//     if (rank == 0) ncclGetUniqueId(&id);
//     MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

//     worker(rank, world_size, id, rows, cols, show_mesh, show_desc, save_ckpt, load_ckpt);

//     MPI_Barrier(MPI_COMM_WORLD);
//     if (rank == 0) {
//         std::cout << "\n=== Allocator Stats ===" << std::endl;
//         gAllocator.printStats();
//     }

//     MPI_Finalize();
//     return 0;
// }

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
#include "bridge/tensor_ops_bridge.h"

using namespace OwnTensor;

// =============================================================
// Utility: Parse shape from string like "--shape=2x3x4"
// =============================================================
std::vector<int> parseShapeArg(const char* arg) {
    std::vector<int> shape;
    const char* p = arg + 8; // skip "--shape="
    std::string num;
    while (*p) {
        if (*p == 'x') {
            if (!num.empty()) {
                shape.push_back(std::stoi(num));
                num.clear();
            }
        } else {
            num.push_back(*p);
        }
        ++p;
    }
    if (!num.empty()) shape.push_back(std::stoi(num));
    return shape;
}

// =============================================================
// Worker Function
// =============================================================
void worker(int rank, int world_size, const ncclUniqueId& id,
            const std::vector<int>& shape,
            bool show_mesh, bool show_desc, bool save_ckpt, bool load_ckpt) {

    cudaSetDevice(rank);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ProcessGroup pg(rank, world_size, rank, id);
    DTensor tensor(rank, world_size, &pg);

    std::string ckpt_dir = "checkpoints";
    std::filesystem::create_directories(ckpt_dir);

    int total_elems = 1;
    for (int d : shape) total_elems *= d;

    std::vector<float> local_data(total_elems);

    // ------------------------------------------------------------
    // Load or Random Initialize
    // ------------------------------------------------------------
    if (load_ckpt) {
        std::string path = ckpt_dir + "/tensor_rank" + std::to_string(rank) + ".bin";
        tensor.loadCheckpoint(path);
        std::cout << "[Rank " << rank << "] Loaded checkpoint from " << path << std::endl;
    } else {
        srand(time(nullptr) + rank * 77);
        for (int i = 0; i < total_elems; ++i)
            local_data[i] = static_cast<float>(rand() % 1000) / 100.0f;
        tensor.setData(local_data, shape);
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // Metadata Print
    // ------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n[TensorLib Integration Check]\n";
        std::cout << "[Rank 0] Tensor metadata: device=CUDA, dtype=float32, shape=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << "x";
        }
        std::cout << "]\n";
    }

    // ------------------------------------------------------------
    // Collective Test
    // ------------------------------------------------------------
    if (rank == 0) std::cout << "\n==== BEFORE ALLREDUCE ====\n";
    MPI_Barrier(MPI_COMM_WORLD);
    tensor.print();

    tensor.allReduce();
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) std::cout << "\n==== AFTER ALLREDUCE ====\n";
    tensor.print();

    // ------------------------------------------------------------
    // Functional Ops
    // ------------------------------------------------------------
    if (rank == 0)
        std::cout << "\n==== TensorOpsBridge Functional Tests (Post-AllReduce) ====\n";

    DTensor other(rank, world_size, &pg);
    std::vector<float> local_data_B(total_elems);
    srand(time(nullptr) + rank * 101);
    for (int i = 0; i < total_elems; ++i)
        local_data_B[i] = static_cast<float>(rand() % 1000) / 100.0f;
    other.setData(local_data_B, shape);

    DTensor add_res = tensor.add(other);
    DTensor sub_res = tensor.sub(other);
    DTensor mul_res = tensor.mul(other);
    DTensor div_res = tensor.div(other);

    bool canMatmul = false;
    if (shape.size() >= 2) {
        int lastA = shape[shape.size() - 1];
        int lastB = shape[shape.size() - 2];
        canMatmul = (lastA == lastB);
    }

    if (rank == 0) {
        std::cout << "\nA:\n"; tensor.print();
        std::cout << "B:\n"; other.print();
        std::cout << "A + B:\n"; add_res.print();
        std::cout << "A - B:\n"; sub_res.print();
        std::cout << "A * B:\n"; mul_res.print();
        std::cout << "A / B:\n"; div_res.print();
    }

    if (canMatmul) {
        DTensor matmul_res = tensor.matmul(other);
        if (rank == 0) {
            std::cout << "A x B (matmul):\n"; matmul_res.print();
        }
    } else if (rank == 0) {
        std::cout << "[Info] Skipping matmul (incompatible or ND tensor)\n";
    }

    // ------------------------------------------------------------
    // Checkpoint (optional)
    // ------------------------------------------------------------
    if (save_ckpt) {
        std::string path = ckpt_dir + "/tensor_rank" + std::to_string(rank) + ".bin";
        tensor.saveCheckpoint(path);
    }

    cudaStreamDestroy(stream);
}

// =============================================================
// Main Entry
// =============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
        std::cout << "[Init] Using DTensor (TensorLib backend + CUDA + NCCL)\n";

    cudaFree(0); // Force CUDA context init

    bool show_mesh = false, show_desc = false, save_ckpt = false, load_ckpt = false;
    std::vector<int> shape = {2, 2}; // Default 2D

    // ------------------------------------------------------------
    // CLI: flexible parsing
    // ------------------------------------------------------------
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--mesh") == 0)
            show_mesh = true;
        else if (strcmp(arg, "--describe") == 0)
            show_desc = true;
        else if (strcmp(arg, "--save") == 0)
            save_ckpt = true;
        else if (strcmp(arg, "--load") == 0)
            load_ckpt = true;
        else if (strncmp(arg, "--rows=", 7) == 0)
            shape = {atoi(arg + 7), 2};
        else if (strncmp(arg, "--cols=", 7) == 0)
            shape = {2, atoi(arg + 7)};
        else if (strcmp(arg, "--3d") == 0)
            shape = {2, 2, 2};
        else if (strncmp(arg, "--shape=", 8) == 0)
            shape = parseShapeArg(arg);
    }

    if (rank == 0) {
        std::cout << "[Config] Tensor shape = ";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << "x";
        }
        std::cout << "\n";
    }

    // ------------------------------------------------------------
    // NCCL Init + Worker
    // ------------------------------------------------------------
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    worker(rank, world_size, id, shape, show_mesh, show_desc, save_ckpt, load_ckpt);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n=== Allocator Stats ===" << std::endl;
        gAllocator.printStats();
    }

    MPI_Finalize();
    return 0;
}

