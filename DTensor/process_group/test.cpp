#include "ProcessGroupNCCL.h"
#include "TensorLib.h"
#include <mpi.h>
using namespace OwnTensor;
int main(int args, char **argv){
    MPI_Init(&args, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
    auto pg = init_process_group(world_size, rank);
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32).with_device(DeviceIndex(Device::CUDA, rank));
    Tensor tensor1;
    if(rank == 0) tensor1 = Tensor::rand(Shape{{50304, 768}}, opts);
    if(rank == 1) tensor1 = Tensor::randn(Shape{{50304, 768}}, opts);

    tensor1.display();

    
    pg->all_reduce(tensor1.data(), tensor1.data(), 16, tensor1.dtype(), sum, true);
    pg->blockStream();
    tensor1.display();

    return 0;


}