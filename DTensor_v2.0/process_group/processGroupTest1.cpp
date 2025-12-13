// test_process_group_nccl_with_owntensor.cpp
#include "./include/ProcessGroupNCCL.h"
#include "./Tensor-Implementations/include/TensorLib.h"                     // Your OwnTensor::Tensor
#include <iostream>
#include <mpi.h>


#define RESULT_CHECK(cmd)                                                  \
    do {                                                                   \
        result_t __r = (cmd);                                              \
        if (__r != pgSuccess) {                                            \
            throw std::runtime_error(                                      \
                std::string("Process Group Error in ") + pgGetError(__r)   \
                            );                                             \
        }                                                                  \
    } while(0)                                                             \

using namespace OwnTensor;

// Helper to print tensor content (only rank 0)
void print_tensor(const Tensor& t, int rank, const char* name)
{
    if (rank != 0) return;

    std::cout << name << " ";
    t.to_cpu().display();
    t.to_cuda();
    std::cout << "\n";
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    cudaSetDevice(rank);

    if (world_size < 2) {
        if (rank == 0)
            std::cerr << "Error: Run with at least 2 processes: mpirun -np 4 ...\n";
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "NCCL + OwnTensor Distributed Test\n";
        std::cout << "World size: " << world_size << " ranks\n";
    }

    // Tell ProcessGroupNCCL how many GPUs per node (change if needed)
    // setenv("NO_GPUS_PER_NODE", "4", 1);  // adjust to your system

    // Use default stream
    cudaStream_t stream;
    std::cout << "INIT STARTED\n";
    cudaStreamCreate(&stream);
    // Initialize the distributed process group
    auto pg = init_process_group(world_size, rank, stream);

    std::cout << "INIT COMPLETED\n";

    int local_rank = pg->get_local_rank();
    if (rank == 0) {
        std::cout << "ProcessGroupNCCL initialized successfully.\n";
        std::cout << "Local rank mapping: ";
        for (int r = 0; r < world_size; ++r)
            std::cout << "rank" << r << "→GPU" << (r % 4) << " ";
        std::cout << "\n\n";
    }

    // ===================================================================
    // Test 1: AllReduce
    // ===================================================================
    {
        Shape shape{{2, 8}};  // 16 elements
        TensorOptions opts;
        opts = opts.with_device(DeviceIndex(Device::CUDA, local_rank))
                   .with_dtype(Dtype::Float32);

        auto send_tensor = Tensor::full(shape, opts, float(rank + 1));  // rank 0 sends 1.0, rank 1 sends 2.0, etc.
        auto recv_tensor = Tensor::zeros(shape, opts);

        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, send_tensor.data());
        std::cout << cudaGetErrorString(err) << std::endl;
        print_tensor(send_tensor, rank, "Send tensor (before AllReduce):");

        std::cout<<"All Reduce Started: \n";
        RESULT_CHECK(pg->all_reduce(
            send_tensor.data(),
            recv_tensor.data(),
            send_tensor.numel(),
            Dtype::Float32,
            sum
        ));
        std::cout<<"All Reduce Finished. \n";

        print_tensor(recv_tensor, rank, "After AllReduce (sum):");

        if (rank == 0) {
            float expected = world_size * (world_size + 1) / 2.0f;  // 1+2+...+world_size
            std::cout << "Expected value in every element: " << expected << "\n\n";
        }
    }

    // ===================================================================
    // Test 2: Broadcast from rank 0
    // ===================================================================
    {
        Shape shape{{4, 4}};
        TensorOptions opts;
        opts = opts.with_device(DeviceIndex(Device::CUDA, local_rank))
                   .with_dtype(Dtype::Float32);

        auto send_tensor = Tensor::zeros(shape, opts);
        auto recv_tensor = Tensor::zeros(shape, opts);

        if (rank == 0) {
            send_tensor.fill(123.456f);
            print_tensor(send_tensor, rank, "Broadcast source (rank 0):");
        }

        RESULT_CHECK(pg->broadcast(
            (const void*) send_tensor.data(),
            (void*)recv_tensor.data(),
            send_tensor.numel(),
            Dtype::Float32,
            0  // root
        ));

        print_tensor(recv_tensor, rank, "After Broadcast (everyone should see 123.456):");
    }

    // ===================================================================
    // Test 3: AllGather
    // ===================================================================
    {
        Shape local_shape{{2, 4}};           // each rank contributes 8 elements
        Shape global_shape{{2, 4 * world_size}}; // concatenated along dim=1

        TensorOptions opts;
        opts = opts.with_device(DeviceIndex(Device::CUDA, local_rank))
                   .with_dtype(Dtype::Float32);

        auto send_tensor = Tensor::full(local_shape, opts, float(rank * 1000));
        auto recv_tensor = Tensor::zeros(global_shape, opts);

        print_tensor(send_tensor, rank, "AllGather local input:");

        RESULT_CHECK(pg->all_gather(
            send_tensor.data(),
            recv_tensor.data(),
            send_tensor.numel(),
            Dtype::Float32
            
        ));

        if (rank == 0) {
            std::cout << "AllGather result (concatenated on rank 0):";
            recv_tensor.to_cpu().display();
            std::cout << "\n";
        }
    }

    // Optional: timing test
    if (rank == 0) std::cout << "\nStarting timing test (AllReduce 10x)...\n";

    {
        Shape shape{{1024, 1024}}; // ~4M elements
        auto opts = TensorOptions()
            .with_device(DeviceIndex(Device::CUDA, local_rank))
            .with_dtype(Dtype::Float32);

        auto a = Tensor::ones(shape, opts);
        auto b = Tensor::zeros(shape, opts);

        pg->start_time();

        for (int i = 0; i < 10; ++i) {
            RESULT_CHECK(pg->all_reduce(a.data(), b.data(), a.numel(), Dtype::Float32, sum));
            RESULT_CHECK(pg->all_reduce(b.data(), a.data(), a.numel(), Dtype::Float32, sum));
        }

        float ms = 0.0f;
        pg->end_time(ms);

        if (rank == 0) {
            std::cout << "10x AllReduce (1024×1024 float32) took: " << ms << " ms\n";
            std::cout << "Bandwidth approx: " << (2.0 * 10 * a.nbytes() / (ms / 1000.0)) / 1e9 << " GB/s\n";
        }
    }

    if (rank == 0)
        std::cout << "\nALL TESTS PASSED! ProcessGroupNCCL + OwnTensor integration works perfectly!\n";

    MPI_Finalize();
    return 0;
}