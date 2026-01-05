

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

    ncclUniqueId nccl_id;
    cudaStream_t comm_stream;
    cudaStreamCreate(&comm_stream);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;
    float avgduration;

    auto pg = device_mesh.get_process_group(0);
    
    // const int64_t B = 3;      // batch size
    // const int64_t C = 2;      // input features
    // const int64_t T = 4;      // token length
    // const int64_t F = 2*2;     // hidden dim (will be sharded: F / P per GPU

    const int64_t B = 8;      // batch size
    const int64_t C = 768;      // input features
    const int64_t T = 1024;      // token length
    const int64_t F = 768*4;     // hidden dim (will be sharded: F / P per GPU)

    for (int i = 0; i < 20; i++ ){


    cudaEventRecord(start,comm_stream);
    
    Layout x_layout( device_mesh,  { B, T, C });
    
    // Layout wqkv_layout(device_mesh, {B, C, 3 * C }, 2);

    if(rank == 0 ) std::cout << "\n x - dtensor : \n";

    DTensor X(device_mesh, pg, x_layout);

    // DTensor WQKV(device_mesh, pg, wqkv_layout);

    // std::vector<float> wqkv_data(B * C * (C * 3));

    // DTensor WQKV_shard(device_mesh, {B, T, ( 3 * C) / 2 });

    X.rand();

    if (rank == 0)  X.display();

    X.replicate(0);  // root = 0


    // WQKV.shard(2, 0 ); // shard dim = 2, root = 0

    // Layout qkvlayout (device_mesh, {B, T , C / world_size }); // can add dim = 1 as another parameter if we are doing context parallelism

    // DTensor q(device_mesh, pg, qkvlayout);
    
    // DTensor k(device_mesh, pg, qkvlayout);
    
    // DTensor v(device_mesh, pg, qkvlayout); 

    // WQKV.qkvslpit(q,k,v);

    Layout w1_layout(device_mesh, {B ,C ,F }, 2);

    if(rank == 0 ) std::cout << "\n w1 - dtensor : \n";

    DTensor W1(device_mesh, pg, w1_layout);

    int local_F = F / world_size;
        
    W1.rand();

    if (rank == 0)  W1.display();

    //  W1.rotate3D( 1, 0 );
    
    // if(rank == 0 ) std::cout << " \n w2 - rotated clockwise : \n";

    // if( rank == 0 ) W1.display();

    Layout W1_asS_layout(device_mesh,{B , C, F/2});

    DTensor W1_Shard(device_mesh, pg, W1_asS_layout);
    
    // W1_Shard.rotate3D( 1, 0 );
    
    W1_Shard.shard(2, 0, W1);  

    if (rank == 0) {std::cout << "\n w1 shard - dtensor : \n";} 

    if (rank == 0){ W1_Shard.display(); }

    // W1_Shard.rotate3D( 1, 1 );

    // if (rank == 0) {std::cout << "\n w1 shard - rotated anticlockwise : \n";} 

    // if (rank == 0){ W1_Shard.display(); }

    Layout H_layout(device_mesh, {B ,T ,F/2 });
    
    DTensor H (device_mesh, pg, H_layout);
    
    DTensor B1 (device_mesh, pg, H_layout);

    B1.rand();
    if(rank == 0 ) std::cout << "\n b1 - dtensor : \n";
    if (rank == 0)  B1.display();

    H.Linear(X,W1_Shard,B1);

    if(rank == 0 ) std::cout << "\n h - dtensor : \n";
    
    H.display(); 

    Layout w2_layout(device_mesh, {B, F, C }, 1);

    DTensor W2(device_mesh, pg, w2_layout);

    std::vector<float> w2_full_data(B * F * C );
    
    W2.rand();

    if(rank == 0 ) std::cout << " \n w2 - dtensor : \n";

    if (rank == 0)  W2.display();

    // W2.rotate3D( 2, 0 );

    // if(rank == 0 ) std::cout << " \n w2 - rotated clockwise : \n";

    // if (rank == 0)  W2.display();

    Layout W2_asS_layout(device_mesh,{B, (F/2), C});

    DTensor W2_Shard(device_mesh, pg, W2_asS_layout);
  
    // W2_Shard.rotate3D( 2, 0 );
    
    W2_Shard.shard_default(1, 0, W2);  // shard on dim 1, root = 0, parentTensor W2

    if (rank == 0) {std::cout << "\n w1 shard - dtensor: \n";} 

    if (rank == 0){ W2_Shard.display(); }

    // W2_Shard.rotate3D( 2, 1 );
    
    // if (rank == 0) {std::cout << "\n w1 shard - rotated anticlockwise : \n";} 

    // if (rank == 0){ W2_Shard.display(); }

    DTensor Y(device_mesh, pg, x_layout);

    DTensor B2(device_mesh, pg, x_layout);
    
    B2.rand();
  
    if(rank == 0 ) std::cout << "\n b2 - dtensor : \n";
    if (rank == 0)  B2.display();
    
    Y.Linear(H,W2_Shard,B2);

    if(rank == 0 ) std::cout << "\n y - dtensor : \n";

  
    if(rank == 0 )  std::cout<<"\n Y before sync \n"; 

    if (rank == 0) { Y.display(); }
    
    Y.sync();

    if (rank == 0) { std::cout<<"\n Y after sync \n"; }

    if (rank == 0) {  Y.display(); }

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
      std::cout<<" B : "<< B <<std::endl;
      std::cout<<" T : "<< T <<std::endl;
      std::cout<<" C : "<< C <<std::endl;
      std::cout<<" F : "<< F <<std::endl;    
    }

    // Layout grad_layout (device_mesh, {B, T, C });

    // DTensor grad_Y(device_mesh, pg, grad_layout);

    // grad_Y.rand();
    
    // if (rank == 0) {
        
    //     std::cout << "\n=== Before sync() ===" << std::endl;
    //     grad_Y.display();

    // }
    
    // grad_Y.sync(); // sum gradients
    
    // if (rank == 0) {
    //     std::cout << "\n=== After sync() - Gradients added ===" << std::endl;

    //        grad_Y.display();

    // }
    
    MPI_Finalize();
    return 0;
}
