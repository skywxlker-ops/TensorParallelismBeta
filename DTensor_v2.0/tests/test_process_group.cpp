#include "process_group.h"
#include "tensor/tensor.h" // Assuming this is the header for the Tensor library
#include <iostream>
#include <vector>

// Helper to create a ncclUniqueId
ncclUniqueId get_nccl_unique_id() {
    ncclUniqueId id;
    if (0 == 0) { // Assuming rank 0 generates the ID
        ncclGetUniqueId(&id);
    }
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    return id;
}

// Helper to print a tensor from the device
template<typename T>
void print_tensor(const tensor::Tensor<T>& t, int rank, const std::string& name) {
    if (rank != 0) return;

    std::vector<T> host_data(t.numel());
    cudaMemcpy(host_data.data(), t.data_ptr(), t.numel() * sizeof(T), cudaMemcpyHostToDevice);

    std::cout << name << " (on rank 0):" << std::endl;
    for (size_t i = 0; i < t.numel(); ++i) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (rank == 0) std::cerr << "This test requires at least 2 ranks." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int local_rank = rank % 4; // Assuming 4 GPUs per node
    cudaSetDevice(local_rank);

    ncclUniqueId id = get_nccl_unique_id();
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, local_rank, id);

    if (rank == 0) {
        std::cout << "ProcessGroup Test using corrected out-of-place collectives" << std::endl;
        std::cout << "World Size: " << world_size << std::endl << std::endl;
    }

    // ================== Test 1: AllReduce ==================
    {
        if (rank == 0) std::cout << "--- Testing AllReduce ---" << std::endl;
        std::vector<int64_t> shape = {2, 4};
        tensor::Tensor<float> send_tensor({shape}, tensor::Device::CUDA, local_rank);
        tensor::Tensor<float> recv_tensor({shape}, tensor::Device::CUDA, local_rank);

        std::vector<float> initial_data(send_tensor.numel(), (float)(rank + 1));
        cudaMemcpy(send_tensor.data_ptr(), initial_data.data(), initial_data.size() * sizeof(float), cudaMemcpyHostToDevice);

        auto work = pg->allReduce<float>(send_tensor.data_ptr(), recv_tensor.data_ptr(), send_tensor.numel(), ncclFloat);
        work->wait();

        float expected_sum = (float)(world_size * (world_size + 1)) / 2.0f;
        print_tensor(recv_tensor, rank, "AllReduce Result");
        if(rank==0) std::cout << "Expected value: " << expected_sum << std::endl;
    }

    // ================== Test 2: Broadcast ==================
    {
        if (rank == 0) std::cout << "\n--- Testing Broadcast ---" << std::endl;
        std::vector<int64_t> shape = {2, 2};
        tensor::Tensor<float> send_tensor({shape}, tensor::Device::CUDA, local_rank);
        tensor::Tensor<float> recv_tensor({shape}, tensor::Device::CUDA, local_rank);
        
        if (rank == 0) {
            std::vector<float> initial_data(send_tensor.numel(), 123.45f);
            cudaMemcpy(send_tensor.data_ptr(), initial_data.data(), initial_data.size() * sizeof(float), cudaMemcpyHostToDevice);
            print_tensor(send_tensor, rank, "Broadcast Send (Rank 0)");
        }

        auto work = pg->broadcast<float>(send_tensor.data_ptr(), recv_tensor.data_ptr(), recv_tensor.numel(), 0, ncclFloat);
        work->wait();

        if (rank == 1) { // Print from another rank to confirm
             print_tensor(recv_tensor, 0, "Broadcast Result (on Rank 1, printed by Rank 0)");
        }
    }

    // ================== Test 3: AllGather ==================
    {
        if (rank == 0) std::cout << "\n--- Testing AllGather ---" << std::endl;
        std::vector<int64_t> local_shape = {2, 2};
        std::vector<int64_t> global_shape = {2, 2 * (int64_t)world_size};

        tensor::Tensor<float> send_tensor({local_shape}, tensor::Device::CUDA, local_rank);
        tensor::Tensor<float> recv_tensor({global_shape}, tensor::Device::CUDA, local_rank);

        std::vector<float> initial_data(send_tensor.numel(), (float)(rank * 100));
        cudaMemcpy(send_tensor.data_ptr(), initial_data.data(), initial_data.size() * sizeof(float), cudaMemcpyHostToDevice);

        auto work = pg->allGather<float>(send_tensor.data_ptr(), recv_tensor.data_ptr(), send_tensor.numel(), ncclFloat);
        work->wait();
        
        print_tensor(recv_tensor, rank, "AllGather Result");
    }


    if (rank == 0) {
        std::cout << "\nAll tests completed." << std::endl;
    }

    MPI_Finalize();
    return 0;
}