#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

void test_send_recv(int rank, std::shared_ptr<ProcessGroupNCCL> pg) {
    if (rank == 0) std::cout << "\n[Send/Recv Test]" << std::endl;
    
    float *data;
    CUDA_CHECK(cudaMalloc(&data, 4 * sizeof(float)));
    
    if (rank == 0) {
        float h_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        CUDA_CHECK(cudaMemcpy(data, h_data, 4 * sizeof(float), cudaMemcpyHostToDevice));
        pg->send<float>(data, 4, 1, ncclFloat)->wait();
        std::cout << "  rank 0: sent [1,2,3,4] to rank 1" << std::endl;
    } else if (rank == 1) {
        pg->recv<float>(data, 4, 0, ncclFloat)->wait();
        float h_result[4];
        CUDA_CHECK(cudaMemcpy(h_result, data, 4 * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool pass = (h_result[0] == 1.0f && h_result[1] == 2.0f && 
                     h_result[2] == 3.0f && h_result[3] == 4.0f);
        std::cout << "  rank 1: received ";
        if (pass) std::cout << "PASS" << std::endl;
        else std::cout << "FAIL" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(data));
    MPI_Barrier(MPI_COMM_WORLD);
}

void test_scatter(int rank, int world_size, std::shared_ptr<ProcessGroupNCCL> pg) {
    if (rank == 0) std::cout << "\n[Scatter Test]" << std::endl;
    
    float *send_buf = nullptr;
    float *recv_buf;
    
    CUDA_CHECK(cudaMalloc(&recv_buf, 4 * sizeof(float)));
    
    if (rank == 0) {
        // Root prepares 8 elements (4 per rank)
        CUDA_CHECK(cudaMalloc(&send_buf, 8 * sizeof(float)));
        float h_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        CUDA_CHECK(cudaMemcpy(send_buf, h_data, 8 * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Scatter: root sends 4 elements to each rank
    pg->scatter<float>(send_buf, recv_buf, 4, 0, ncclFloat)->wait();
    
    // Verify
    float h_result[4];
    CUDA_CHECK(cudaMemcpy(h_result, recv_buf, 4 * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool pass = false;
    if (rank == 0) {
        pass = (h_result[0] == 1.0f && h_result[1] == 2.0f && 
                h_result[2] == 3.0f && h_result[3] == 4.0f);
        std::cout << "  rank 0: received [1,2,3,4] ";
    } else if (rank == 1) {
        pass = (h_result[0] == 5.0f && h_result[1] == 6.0f && 
                h_result[2] == 7.0f && h_result[3] == 8.0f);
        std::cout << "  rank 1: received [5,6,7,8] ";
    }
    
    if (pass) std::cout << "PASS" << std::endl;
    else std::cout << "FAIL" << std::endl;
    
    if (send_buf) CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaFree(recv_buf));
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "This test requires exactly 2 MPI processes." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    ncclUniqueId nccl_id;
    if (rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }
    MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    auto pg = init_process_group(world_size, rank);
    CUDA_CHECK(cudaSetDevice(rank));

    if (rank == 0) {
        std::cout << "\nPoint-to-Point & Scatter Tests:\n";
    }

    test_send_recv(rank, pg);
    test_scatter(rank, world_size, pg);

    if (rank == 0) {
        std::cout << "\nDone.\n";
    }

    MPI_Finalize();
    return 0;
}
