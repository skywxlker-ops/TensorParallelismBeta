

#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>

int main(int argc, char** argv) {
    std::cout<<"-3";
    MPI_Init(&argc, &argv);
    
    std::cout<<"-2";
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::cout<<"-1";
    DeviceMesh device_mesh ({2}, {0,1});
    std::cout<<"0";
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
    
    const int64_t B = 8;      // batch size
    const int64_t C = 768;      // input features
    const int64_t T = 1024;      // token length
    const int64_t F = 768*4;     // hidden dim (will be sharded: F / P per GPU)

  


    Layout x_layout( device_mesh,  { B, T, C });
    
    Layout wqkv_layout(device_mesh, {B, C, 3 * C }, 2);

    DTensor X(device_mesh, pg, x_layout);
    std::vector<float> x_data(B * T * C );
    std::cout<<"1";

    // DTensor WQKV(device_mesh, pg, wqkv_layout);

    // std::vector<float> wqkv_data(B * C * (C * 3));

    // DTensor WQKV_shard(device_mesh, {B, T, ( 3 * C) / 2 });
   

    

    if (rank == 0) {
        for (int i = 0; i < B * T * C ; i++) x_data[i] = 0.1f * (i + 1);
        // for (int i = 0; i < B * C * C ; i++) wqkv_data[i] = 0.67f * (i + 1);

        X.setData(x_data);
        // WQKV.setData(x_data);
    } 
        std::cout<<"2";

    X.replicate(0);  // root = 0
    // WQKV.shard(2, 0 ); // shard dim = 2, root = 0

    // Layout qkvlayout (device_mesh, {B, T , C / world_size }); // can add dim = 1 as another parameter if we are doing context parallelism

    // DTensor q(device_mesh, pg, qkvlayout);
    
    // DTensor k(device_mesh, pg, qkvlayout);
    
    // DTensor v(device_mesh, pg, qkvlayout); 

    // WQKV.qkvslpit(q,k,v);



    if (rank == 0) {
        std::cout << "=== Input X (replicated via broadcast) ===" << std::endl;
        std::cout << "Shape: [" << B << ", " << T << ", " << C << "]" << std::endl;
    }
    
    Layout w1_layout(device_mesh, {B ,C ,F }, 1);

    DTensor W1(device_mesh, pg, w1_layout);
    int local_F = F / world_size;
    
    // Full W1 tensor created on root GPU only
    std::vector<float> w1_full_data(B * C * F );
    if (rank == 0) {
        for (int i = 0; i < C * F ; i++) w1_full_data[i] = 0.01f * (i % F + 1);
    }
    

    W1.setData(w1_full_data);
    
    Layout W1_asS_layout(device_mesh,{B, C, F/2});

    DTensor W1_Shard(device_mesh, pg, W1_asS_layout);

    W1_Shard.shard(1, 0, W1);  
    
    if (rank == 0) {
        std::cout << "=== W1 (sharded on dim 1) ===" << std::endl;
        std::cout << "Global: [" <<  C << ", " << F << "], Local: [" << C << ", " << local_F << "]" << std::endl;
    }
    
    Layout H_layout(device_mesh, {B ,T ,F/2 });

    DTensor H (device_mesh, pg, H_layout);

    H.matmul(X,W1_Shard);
    
    if (rank == 0) {
        std::cout << "\n=== After Column-Parallel MatMul ===" << std::endl;
        std::cout << "H is SHARDED: [" << B << ", " << local_F << "] per GPU" << std::endl;
    }
    
    Layout w2_layout(device_mesh, {B, F, C }, 2);

    DTensor W2(device_mesh, pg, w2_layout);
    
    // Full W2 tensor created on root GPU only
    std::vector<float> w2_full_data(B * (F/2) * C );
    if (rank == 0) {
        for (int i = 0; i < B * (F/2) * C; i++) w2_full_data[i] = 0.02f;
    }
    
    // replicated layout (root has full data, others have placeholder)
    W2.setData(w2_full_data);

    Layout W2_asS_layout(device_mesh,{B, (F/2), C});

    DTensor W2_Shard(device_mesh, pg, W1_asS_layout);
    
    W2_Shard.shard(2, 0, W2);  // shard on dim 2, root = 0, parentTensor W2
    

    DTensor Y(device_mesh, pg, x_layout);;
    
    Y.matmul(H,W2_Shard);
    
    Y.sync();

    if (rank == 0) {
        std::cout << "\n=== After Row-Parallel MatMul ===" << std::endl;
        std::cout << "Y shape: [" << B << ", " << T << ", " << C << "] (synched after AllReduce)" << std::endl;
        
        auto y_data = Y.getData();
        std::cout << "Y values (first 5): ";
        for (int i = 0; i < 5; i++) std::cout << y_data[i] << " ";
        std::cout << std::endl;
    }
    
    Layout grad_layout (device_mesh, {B, T, C });

    DTensor grad_Y(device_mesh, pg, grad_layout);

    std::vector<float> grad_data(B * T * C );

    for (int i = 0; i < B * T * C ; i++) {
        grad_data[i] = 0.1f * ((i % 7) + 1) * (rank + 1) + 0.05f * (i % 3);
    }
    
    grad_Y.setData(grad_data);
    
    if (rank == 0) {
        std::cout << "\n=== Before sync() ===" << std::endl;
        auto data = grad_Y.getData();
        std::cout << "Rank 0 grads (first 5): ";
        for (int i = 0; i < 5; i++) std::cout << data[i] << " ";
        std::cout << std::endl;
    }
    
    grad_Y.sync(); // sum gradients
    
    if (rank == 0) {
        std::cout << "\n=== After sync() - Gradients added ===" << std::endl;
        auto data = grad_Y.getData();
        std::cout << "Averaged grads (first 5): ";
        for (int i = 0; i < 5; i++) std::cout << data[i] << " ";
        std::cout << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
