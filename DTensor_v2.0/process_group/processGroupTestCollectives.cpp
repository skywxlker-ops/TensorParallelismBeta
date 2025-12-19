#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>
#include "../Tensor-Implementations/include/TensorLib.h"
#include "ProcessGroupNCCL.h"

#define ProcessGroup_CHECK(cmd)                                                \ 
    do{                                                                        \
        result_t r = (cmd);                                                    \      
        if(cmd != pgSuccess){                                                  \
            throw std::runtime_error(std::string("PGERROR") + pgGetError(r));  \
        }                                                                      \
    }while(0)                                                                  \

int  main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    int world_size;
    int rank;
    int root = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    cudaSetDevice(rank);

    auto pg = init_process_group(world_size, rank);


    //sizes
    int m = 100;
    int n = 200;


    //declarations
    OwnTensor::TensorOptions tensor_options;
    tensor_options.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
    tensor_options.dtype = OwnTensor::Dtype::Float32;
    OwnTensor::Tensor tensor = OwnTensor::Tensor::randn(OwnTensor::Shape{{m,n}}, tensor_options);
    OwnTensor::Tensor tensor_broadcast = OwnTensor::Tensor(OwnTensor::Shape{{m,n}}, tensor_options);
    OwnTensor::Tensor tensor_all_gather = OwnTensor::Tensor(OwnTensor::Shape{{m*world_size, n}}, tensor_options);
    
    std::cout << "Tensors before ProcessGroup: " << std::endl;
    if(rank == 0){
        std::cout << "TENSOR IN GPU: " << rank << std::endl;
        tensor.to_cpu().display();
        tensor.to(tensor_options.device);
    }else{
        std::cout << "TENSOR IN GPU: " << rank << std::endl;
        tensor.to_cpu().display();
        tensor.to(tensor_options.device);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) std::cout << "\n======== TEST STARTED ========\n" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    //Collectives

    //broadcast (from both ranks to each other)
    if(rank == 0) std::cout<< "\n\n => BroadCast Testing:\n\n " << std::endl;
    ProcessGroup_CHECK(pg->broadcast(tensor.data(), tensor_broadcast.data(), tensor.numel(), tensor.dtype(), rank, true));

    
    if(rank == 0){
        std::cout << "TENSOR IN GPU: " << rank << std::endl;
        tensor_broadcast.to_cpu().display();
        tensor_broadcast.to(tensor_options.device);
    }else{
        std::cout << "TENSOR IN GPU: " << rank << std::endl;
        tensor_broadcast.to_cpu().display();
        tensor_broadcast.to(tensor_options.device);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //all_gather
    if(rank == 0) std::cout<< "\n\n => AllGather Testing:\n\n " << std::endl;
    ProcessGroup_CHECK(pg->all_gather(tensor.data(), tensor_all_gather.data(), tensor.numel()/world_size, tensor.dtype(), true));
    if(rank == 0){
        std::cout << "DEVICE IN GPU: " << rank << std::endl;
        tensor_all_gather.to_cpu().display();
        tensor_all_gather.to(tensor_options.device);
    }else{
        std::cout << "DEVICE IN GPU: " << rank << std::endl;
        tensor_all_gather.to_cpu().display();
        tensor_all_gather.to(tensor_options.device);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //reduce_scatter
    if(rank == 0) std::cout<< "\n\n => RecuceScatter Testing:\n\n " << std::endl;
    ProcessGroup_CHECK(pg->reduce_scatter(tensor.data(), tensor.data(), tensor.numel()/2, tensor.dtype(), avg, true));
    if(rank == 0){
        std::cout << "DEVICE IN GPU: " << rank << std::endl;
        tensor.to_cpu().display();
        tensor.to(tensor_options.device);
    }else{
        std::cout << "DEVICE IN GPU: " << rank << std::endl;
        tensor.to_cpu().display();
        tensor.to(tensor_options.device);
    }

    MPI_Barrier(MPI_COMM_WORLD);


    if(rank == 0) std::cout << "\n======== TEST COMPLETED ========\n" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}