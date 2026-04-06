#include "TensorLib.h"
#include <iostream>
#include <mpi.h>

int main(int args, char *argv[]){
    MPI_Init(&args, &argv);
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &rank);
    cudaSetDevice(rank);
    auto tensor1 = OwnTensor::Tensor::randn<float>({{1, 1024, 768}}, {OwnTensor::Dtype::Float32, OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, 1)});
    auto tensor2 = OwnTensor::Tensor::randn<float>({{8, 1024, 768}}, {OwnTensor::Dtype::Float32, OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, 1)});

    auto tensor3 = tensor1 + tensor2;
    // tensor3
    // std::cout << tensor1.data<float>() << std::endl;
    
    MPI_Finalize();

    return 0;

}