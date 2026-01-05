#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::shared_ptr<Work> work_obj;
   
    DeviceMesh device_mesh ({2}, {0,1});

    auto pg = device_mesh.get_process_group(0);

    cudaStream_t comm_stream;
    cudaStreamCreate(&comm_stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float duration;
    float avgduration = 0.0f;

    const int64_t X = 69;
    const int64_t Y = 67;
    const int64_t Z = 420;

    

    for (int i = 0; i < 1; i++ ){

        cudaEventRecord(start,comm_stream);     

        Layout w1_layout(device_mesh, { X , Y , Z }, 2);

        DTensor W1(device_mesh, pg, w1_layout);

        W1.rand();

        Layout W1_asS_layout(device_mesh,{ X , Y , Z/2 });

        DTensor W1_Shard(device_mesh, pg, W1_asS_layout);

        W1.shard( 0 , 1 , W1 );

        CUDA_CHECK(cudaStreamSynchronize(comm_stream));
        
        cudaEventRecord(stop,comm_stream);

        cudaEventElapsedTime(&duration, start, stop);

        avgduration += duration;

}

avgduration /= 20;
    
cudaEventDestroy(start);
cudaEventDestroy(stop);

if(rank == 0){
  std::cout<<"\n\n ========= BENCHMARKS ========= \n\n"<<std::endl;
  std::cout<<" DURATION : "<< duration <<std::endl;
  std::cout<<" X : "<< X <<std::endl;
  std::cout<<" Y : "<< Y <<std::endl;
  std::cout<<" Z : "<< Z <<std::endl;
 
}
    MPI_Finalize();
    return 0;
}