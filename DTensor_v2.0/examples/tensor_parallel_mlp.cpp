

#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
   
    DeviceMesh device_mesh ({2}, {0,1});

    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
  /*  
    const int64_t B = 8;      // batch size
    const int64_t C = 768;      // input features
    const int64_t T = 1024;      // token length
    const int64_t F = 768*4;     // hidden dim (will be sharded: F / P per GPU)

  */
    const int64_t B = 2;      // batch size
    const int64_t C = 2;      // input features
    const int64_t T = 4;      // token length
    const int64_t F = 2*2;     // hidden dim (will be sharded: F / P per GPU


    Layout x_layout( device_mesh,  { B, T, C });
    
    // Layout wqkv_layout(device_mesh, {B, C, 3 * C }, 2);
    if(rank == 0 ) std::cout << "\n x - dtensor: \n";
    DTensor X(device_mesh, pg, x_layout);
    std::vector<float> x_data(B * T * C );


    // DTensor WQKV(device_mesh, pg, wqkv_layout);

    // std::vector<float> wqkv_data(B * C * (C * 3));

    // DTensor WQKV_shard(device_mesh, {B, T, ( 3 * C) / 2 });
   

    

    // if (rank == 0) {
    //     for (int i = 0; i < B * T * C ; i++) x_data[i] = 0.67f * (i + 1);
    //     // for (int i = 0; i < B * C * C ; i++) wqkv_data[i] = 0.67f * (i + 1);

    //     X.setData(x_data);
    //     // WQKV.setData(x_data);
    // } 
    
    X.rand();

    if (rank == 0)  X.display();

    X.replicate(0);  // root = 0
    // WQKV.shard(2, 0 ); // shard dim = 2, root = 0

    // Layout qkvlayout (device_mesh, {B, T , C / world_size }); // can add dim = 1 as another parameter if we are doing context parallelism

    // DTensor q(device_mesh, pg, qkvlayout);
    
    // DTensor k(device_mesh, pg, qkvlayout);
    
    // DTensor v(device_mesh, pg, qkvlayout); 

    // WQKV.qkvslpit(q,k,v);



    // if (rank == 0) {
    //     std::cout << "=== Input X (replicated via broadcast) ===" << std::endl;
    //     std::cout << "Shape: [" << X.get_layout().get_global_shape()[0] << ", " << X.get_layout().get_global_shape()[1] << ", " << X.get_layout().get_global_shape()[2] << "]" << std::endl;
    // }
    
    Layout w1_layout(device_mesh, {B ,C ,F }, 2);

    // std::cout << "Layout completed!!" << std::endl;
    if(rank == 0 ) std::cout << "\n w1 - dtensor: \n";

    DTensor W1(device_mesh, pg, w1_layout);

    // std::cout << "DTensor completed!!" << std::endl;
    int local_F = F / world_size;
    
    // Full W1 tensor created on root GPU only
    std::vector<float> w1_full_data(B * C * F );
    // if (rank == 0) {
    //     for (int i = 0; i < C * F ; i++) w1_full_data[i] = 0.01f * (i % F + 1);
    // }
    
    
    // W1.setData(w1_full_data);
    
    W1.rand();

    if (rank == 0)  W1.display();

    // std::cout << "allocation started" <<std::endl;
    Layout W1_asS_layout(device_mesh,{B, C, F/2});
    if (rank == 0) {std::cout << "\n w1 shard - dtensor: \n";}

    DTensor W1_Shard(device_mesh, pg, W1_asS_layout);
    
    W1_Shard.shard(2, 0, W1);  
    // std::cout << "allocation ended" << rank<<std::endl;
    if (rank == 0) {
        // std::cout << "=== W1 (sharded on dim 2) ===" << std::endl;
        // std::cout << "Global: [" <<  W1.get_layout().get_global_shape()[0] << ", " << W1.get_layout().get_global_shape()[1] <<", " <<W1.get_layout().get_global_shape()[2]<<"], Local: [" << W1_Shard.get_layout().get_global_shape()[0] << ", " << W1_Shard.get_layout().get_global_shape()[1] <<", "<<W1_Shard.get_layout().get_global_shape()[2]<< "]" << std::endl;
    }
    
    { W1_Shard.display(); }

    Layout H_layout(device_mesh, {B ,T ,F/2 });
    // std::cout<<"m"<<rank;
    if(rank == 0 ) std::cout << "\n h - dtensor: \n";

    DTensor H (device_mesh, pg, H_layout);

    H.matmul(X,W1_Shard);

    if (rank == 0) { H.display(); }
    //   std::cout<<"k"<<rank;
      
    // if (rank == 1) {
    //     std::cout << "\n=== After Column-Parallel MatMul ===" << std::endl;
    //     std::cout << "H is SHARDED: [" << H.get_layout().get_global_shape()[0] << ", " << H.get_layout().get_global_shape()[1] << ", "<<H.get_layout().get_global_shape()[2]<<"] per GPU" <<rank<< std::endl;
    // }
    //   std::cout<<"p";
    Layout w2_layout(device_mesh, {B, F, C }, 1);
    // std::cout<<"t";
    if(rank == 0 ) std::cout << " \n w2 - dtensor: \n";

    DTensor W2(device_mesh, pg, w2_layout);
    // std::cout<<"q";
    // Full W2 tensor created on root GPU only
    std::vector<float> w2_full_data(B * F * C );
    // if (rank == 0) {
    //     for (int i = 0; i < B * (F/2) * C; i++) w2_full_data[i] = 0.02f;
    // }
   
  
    // replicated layout (root has full data, others have placeholder)
    // W2.setData(w2_full_data);

    W2.rand();

    if (rank == 0) { W2.display(); }

    if(rank == 0) std::cout << "\n w2-shard - dtensor: \n";
    Layout W2_asS_layout(device_mesh,{B, (F/2), C});

    DTensor W2_Shard(device_mesh, pg, W2_asS_layout);
    
    W2_Shard.shard(1, 0, W2);  // shard on dim 1, root = 0, parentTensor W2
    
    { W2_Shard.display(); }

    if(rank == 0 ) std::cout << "\n y - dtensor: \n";
    
    DTensor Y(device_mesh, pg, x_layout);
    
    Y.matmul(H,W2_Shard);

    if (rank == 0) {  std::cout<<"\n Y before sync \n"; Y.display(); }

   
    
    Y.sync();

    if (rank == 0) { std::cout<<"\n Y after sync \n"; }

    

    if (rank == 0) {
        // std::cout << "\n=== After Row-Parallel MatMul ===" << std::endl;
        // // std::cout << "Y shape: [" << Y.get_layout().get_global_shape()[0] << ", " << Y.get_layout().get_global_shape()[1] << ", " << Y.get_layout().get_global_shape()[2] << "] (synched after AllReduce)" << std::endl;
        
        // auto y_data = Y.getData();
        // std::cout << "Y values (first 5): ";
        // for (int i = 0; i < 5; i++) std::cout << y_data[i] << " ";
        // std::cout << std::endl;
        Y.display();
    }
    
    Layout grad_layout (device_mesh, {B, T, C });

    DTensor grad_Y(device_mesh, pg, grad_layout);

    // std::vector<float> grad_data(B * T * C );

    // for (int i = 0; i < B * T * C ; i++) {
    //     grad_data[i] = 0.1f * ((i % 7) + 1) * (rank + 1) + 0.05f * (i % 3);
    // }
    
    grad_Y.rand();
    
    if (rank == 0) {
        std::cout << "\n=== Before sync() ===" << std::endl;
        // auto data = grad_Y.getData();
        // std::cout << "Rank 0 grads (first 5): ";
        // for (int i = 0; i < 5; i++) std::cout << data[i] << " ";
        // std::cout << std::endl;
        grad_Y.display();

    }
    
   
    grad_Y.sync(); // sum gradients
    
    if (rank == 0) {
        std::cout << "\n=== After sync() - Gradients added ===" << std::endl;
    //     auto data = grad_Y.getData();
    //     std::cout << "Averaged grads (first 5): ";
    //     for (int i = 0; i < 5; i++) std::cout << data[i] << " ";
    //     std::cout << std::endl;
           grad_Y.display();

    }
    
    MPI_Finalize();
    return 0;
}
