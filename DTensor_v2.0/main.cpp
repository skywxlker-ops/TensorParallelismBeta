// #include <mpi.h>
// #include <nccl.h>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>
// #include <memory>
// #include <cstring>
// #include "tensor/dtensor.h"
// #include "process_group/process_group.h"
// #include "tensor/planner.h"
// #include "ckpt/ckpt.h"   // <- new include

// void worker(int rank, int world_size, const ncclUniqueId& id,
//             bool show_mesh, bool show_desc, bool save_ckpt, bool load_ckpt) {

//     cudaSetDevice(rank);
//     ProcessGroup pg(rank, world_size, rank, id);
//     DTensor tensor(rank, world_size, &pg);
//     int global_size = 8;

//     // Mesh view
//     if (show_mesh) {
//         Planner::printLayoutJSON(rank, world_size, global_size);
//         return;
//     }

//     // Planner description
//     if (show_desc) {
//         auto layout = Planner::inferLayout(global_size, world_size);
//         if (rank == 0) std::cout << Planner::describePlan(layout);
//         return;
//     }

//     // === Checkpoint filenames ===
//     std::string ckpt_file = "ckpt_rank_" + std::to_string(rank) + ".bin";

//     // === Load checkpoint if requested ===
//     if (load_ckpt) {
//         auto data = Checkpoint::load(ckpt_file, rank);
//         if (!data.empty()) tensor.setData(data);
//         else {
//             std::cerr << "[Checkpoint] Rank " << rank
//                       << " had no valid checkpoint, initializing fresh data.\n";
//             std::vector<float> local_data = {0, 1, 2, 3, 4, 5, 6, 7};
//             tensor.setData(local_data);
//         }
//     } else {
//         std::vector<float> local_data = {0, 1, 2, 3, 4, 5, 6, 7};
//         tensor.setData(local_data);
//     }

//     if (rank == 0) std::cout << "==== BEFORE ALLREDUCE ====\n";
//     tensor.print();

//     tensor.allReduce();

//     if (rank == 0) std::cout << "==== AFTER ALLREDUCE ====\n";
//     tensor.print();

//     // === Save checkpoint if requested ===
//     if (save_ckpt) {
//         Checkpoint::save(ckpt_file, tensor.getData(), rank);
//     }
// }

// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);
//     int world_size, rank;
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     bool show_mesh = false;
//     bool show_desc = false;
//     bool save_ckpt = false;
//     bool load_ckpt = false;

//     // Command-line argument handling
//     if (argc > 1) {
//         for (int i = 1; i < argc; ++i) {
//             if (std::strcmp(argv[i], "--mesh") == 0)
//                 show_mesh = true;
//             else if (std::strcmp(argv[i], "--describe") == 0)
//                 show_desc = true;
//             else if (std::strcmp(argv[i], "--save") == 0)
//                 save_ckpt = true;
//             else if (std::strcmp(argv[i], "--load") == 0)
//                 load_ckpt = true;
//         }
//     }

//     ncclUniqueId id;
//     if (rank == 0) ncclGetUniqueId(&id);
//     MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

//     worker(rank, world_size, id, show_mesh, show_desc, save_ckpt, load_ckpt);

//     MPI_Finalize();
//     return 0;
// }


// #include <mpi.h>
// #include <nccl.h>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>
// #include <memory>
// #include <cstring>
// #include <filesystem>  // for checkpoint directory creation
// #include "tensor/dtensor.h"
// #include "process_group/process_group.h"
// #include "tensor/planner.h"
// #include "ckpt/ckpt.h"

// void worker(int rank, int world_size, const ncclUniqueId& id,
//             bool show_mesh, bool show_desc, bool save_ckpt, bool load_ckpt) {

//     cudaSetDevice(rank);
//     ProcessGroup pg(rank, world_size, rank, id);
//     DTensor tensor(rank, world_size, &pg);
//     int global_size = 8;

//     // === Ensure checkpoint directory exists ===
//     std::string ckpt_dir = "checkpoints";
//     std::filesystem::create_directories(ckpt_dir);

//     // === Mesh view ===
//     if (show_mesh) {
//         Planner::printLayoutJSON(rank, world_size, global_size);
//         return;
//     }

//     // === Planner description ===
//     if (show_desc) {
//         auto layout = Planner::inferLayout(global_size, world_size);
//         if (rank == 0) std::cout << Planner::describePlan(layout);
//         return;
//     }

//     // === Load latest checkpoint if requested ===
//     if (load_ckpt) {
//         auto [data, meta] = Checkpoint::loadLatest(ckpt_dir, rank);
//         if (!data.empty()) {
//             tensor.setData(data);
//         } else {
//             std::cerr << "[Checkpoint] Rank " << rank
//                       << " had no valid checkpoint, initializing fresh data.\n";
//             std::vector<float> local_data = {0, 1, 2, 3, 4, 5, 6, 7};
//             tensor.setData(local_data);
//         }
//     } else {
//         std::vector<float> local_data = {0, 1, 2, 3, 4, 5, 6, 7};
//         tensor.setData(local_data);
//     }

//     if (rank == 0) std::cout << "==== BEFORE ALLREDUCE ====\n";
//     tensor.print();

//     tensor.allReduce();

//     if (rank == 0) std::cout << "==== AFTER ALLREDUCE ====\n";
//     tensor.print();

//     // === Save new versioned checkpoint if requested ===
//     if (save_ckpt) {
//         Checkpoint::save(ckpt_dir, tensor.getData(), rank);
//     }
// }

// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);
//     int world_size, rank;
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     bool show_mesh = false;
//     bool show_desc = false;
//     bool save_ckpt = false;
//     bool load_ckpt = false;

//     // === Command-line argument handling ===
//     if (argc > 1) {
//         for (int i = 1; i < argc; ++i) {
//             if (std::strcmp(argv[i], "--mesh") == 0)
//                 show_mesh = true;
//             else if (std::strcmp(argv[i], "--describe") == 0)
//                 show_desc = true;
//             else if (std::strcmp(argv[i], "--save") == 0)
//                 save_ckpt = true;
//             else if (std::strcmp(argv[i], "--load") == 0)
//                 load_ckpt = true;
//         }
//     }

//     ncclUniqueId id;
//     if (rank == 0) ncclGetUniqueId(&id);
//     MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

//     worker(rank, world_size, id, show_mesh, show_desc, save_ckpt, load_ckpt);

//     MPI_Finalize();
//     return 0;
// }


// #include <mpi.h>
// #include <nccl.h>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>
// #include <memory>
// #include <cstring>
// #include <filesystem>  // for checkpoint directory creation
// #include "tensor/dtensor.h"
// #include "process_group/process_group.h"
// #include "tensor/planner.h"
// #include "ckpt/ckpt.h"
// #include "memory/cachingAllocator.hpp"  // allocator header

// void worker(int rank, int world_size, const ncclUniqueId& id,
//             bool show_mesh, bool show_desc, bool save_ckpt, bool load_ckpt) {

//     // Set the device for this rank
//     cudaSetDevice(rank);
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     ProcessGroup pg(rank, world_size, rank, id);
//     DTensor tensor(rank, world_size, &pg);
//     int global_size = 8;

//     // === Ensure checkpoint directory exists ===
//     std::string ckpt_dir = "checkpoints";
//     std::filesystem::create_directories(ckpt_dir);

//     // === Mesh view ===
//     if (show_mesh) {
//         Planner::printLayoutJSON(rank, world_size, global_size);
//         cudaStreamDestroy(stream);
//         return;
//     }

//     // === Planner description ===
//     if (show_desc) {
//         auto layout = Planner::inferLayout(global_size, world_size);
//         if (rank == 0) std::cout << Planner::describePlan(layout);
//         cudaStreamDestroy(stream);
//         return;
//     }

//     // === Load latest checkpoint if requested ===
//     if (load_ckpt) {
//         auto [data, meta] = Checkpoint::loadLatest(ckpt_dir, rank);
//         if (!data.empty()) {
//             tensor.setData(data);
//         } else {
//             std::cerr << "[Checkpoint] Rank " << rank
//                       << " had no valid checkpoint, initializing fresh data.\n";
//             std::vector<float> local_data = {0, 1, 2, 3, 4, 5, 6, 7};
//             tensor.setData(local_data);
//         }
//     } else {
//         std::vector<float> local_data = {0, 1, 2, 3, 4, 5, 6, 7};
//         tensor.setData(local_data);
//     }

//     // === Sync before collective ops ===
//     cudaDeviceSynchronize();
//     MPI_Barrier(MPI_COMM_WORLD);

//     if (rank == 0) std::cout << "==== BEFORE ALLREDUCE ====\n";
//     MPI_Barrier(MPI_COMM_WORLD);

//     std::cout << "[Rank " << rank << "] ";
//     tensor.print();

//     // === Perform AllReduce ===
//     tensor.allReduce();

//     cudaDeviceSynchronize();
//     MPI_Barrier(MPI_COMM_WORLD);

//     if (rank == 0) std::cout << "==== AFTER ALLREDUCE ====\n";
//     MPI_Barrier(MPI_COMM_WORLD);

//     std::cout << "[Rank " << rank << "] ";
//     tensor.print();

//     // === Save checkpoint if requested ===
//     if (save_ckpt) {
//         Checkpoint::save(ckpt_dir, tensor.getData(), rank);
//     }

//     cudaStreamDestroy(stream);
// }

// int main(int argc, char** argv) {
//     MPI_Init(&argc, &argv);
//     int world_size, rank;
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     bool show_mesh = false;
//     bool show_desc = false;
//     bool save_ckpt = false;
//     bool load_ckpt = false;

//     // === Command-line argument handling ===
//     if (argc > 1) {
//         for (int i = 1; i < argc; ++i) {
//             if (std::strcmp(argv[i], "--mesh") == 0)
//                 show_mesh = true;
//             else if (std::strcmp(argv[i], "--describe") == 0)
//                 show_desc = true;
//             else if (std::strcmp(argv[i], "--save") == 0)
//                 save_ckpt = true;
//             else if (std::strcmp(argv[i], "--load") == 0)
//                 load_ckpt = true;
//         }
//     }

//     ncclUniqueId id;
//     if (rank == 0) ncclGetUniqueId(&id);
//     MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

//     worker(rank, world_size, id, show_mesh, show_desc, save_ckpt, load_ckpt);

//     // === Allocator stats at the end ===
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
#include "tensor/dtensor.h"
#include "process_group/process_group.h"
#include "tensor/planner.h"
#include "ckpt/ckpt.h"
#include "memory/cachingAllocator.hpp"

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

    // === Load or initialize random data ===
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

        // Debug check before upload
        if (rank == 0) {
            std::cout << "[DEBUG] Rank " << rank << " first 10 values before upload: ";
            for (int i = 0; i < std::min(10, global_size); ++i)
                std::cout << local_data[i] << " ";
            std::cout << "\n";
        }

        tensor.setData(local_data);
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) std::cout << "==== BEFORE ALLREDUCE ====\n";
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[Rank " << rank << "] ";
    tensor.print();

    tensor.allReduce();

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) std::cout << "==== AFTER ALLREDUCE ====\n";
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[Rank " << rank << "] ";
    tensor.print();

    if (save_ckpt) {
        Checkpoint::save(ckpt_dir, tensor.getData(), rank);
    }

    cudaStreamDestroy(stream);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool show_mesh = false;
    bool show_desc = false;
    bool save_ckpt = false;
    bool load_ckpt = false;
    int global_size = 8;  // default if no user input

    // === Parse CLI args ===
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mesh") == 0)
            show_mesh = true;
        else if (std::strcmp(argv[i], "--describe") == 0)
            show_desc = true;
        else if (std::strcmp(argv[i], "--save") == 0)
            save_ckpt = true;
        else if (std::strcmp(argv[i], "--load") == 0)
            load_ckpt = true;
        else if (std::strncmp(argv[i], "--", 2) == 0) {
            // e.g. --256 or --1024
            int val = std::atoi(argv[i] + 2);
            if (val > 0)
                global_size = val;
        }
    }

    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    worker(rank, world_size, id, global_size, show_mesh, show_desc, save_ckpt, load_ckpt);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n=== Allocator Stats ===" << std::endl;
        gAllocator.printStats();
    }

    MPI_Finalize();
    return 0;
}




