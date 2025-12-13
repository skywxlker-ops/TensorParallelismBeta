

#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    auto device_mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
    const int B = 8;      // batch size
    const int C = 768;      // input features
    const int T = 1024;      // token length
    const int F = 768*4;     // hidden dim (will be sharded: F / P per GPU)

    Layout x_layout(device_mesh, {B, T, C });
    Layout wqkv_layout(device_mesh, {C, 3 * C }, 1);

    DTensor X(device_mesh, pg, x_layout);
    std::vector<float> x_data(B * T * C );
    X.reshape({B, T, C});

    DTensor WQKV(device_mesh, pg, wqkv_layout);
    std::vector<float> wqkv_data(C * (C * 3));
    WQKV.reshape({C, 3 * C});

    

    if (rank == 0) {
        for (int i = 0; i < B * T * C ; i++) x_data[i] = 0.1f * (i + 1);
        for (int i = 0; i < C * 3 * C ; i++) wqkv_data[i] = 0.67f * (i + 1);

        X.setData(x_data);
        WQKV.setData(wqkv_data);
    } 
    
    X.replicate(0);  // root = 0
    WQKV.shard(2, 0 ); // shard dim = 2, root = 0

    

    if (rank == 0) {
        std::cout << "=== Input X (replicated via broadcast) ===" << std::endl;
        std::cout << "Shape: [" << B << ", " << T << ", " << C << "]" << std::endl;
    }
    

    DTensor W1(device_mesh, pg);
    int local_F = F / world_size;
    
    // Full W1 tensor created on root GPU only
    std::vector<float> w1_full_data(C * F );
    if (rank == 0) {
        for (int i = 0; i < C * F ; i++) w1_full_data[i] = 0.01f * (i % F + 1);
    }
    

    Layout w1_replicated(device_mesh, {C, F});
    W1.setData(w1_full_data, w1_replicated);
    

    W1.shard(2, 0);  
    
    if (rank == 0) {
        std::cout << "=== W1 (sharded on dim 1) ===" << std::endl;
        std::cout << "Global: [" << C << ", " << F << "], Local: [" << C << ", " << local_F << "]" << std::endl;
    }
    
   
    DTensor H = X.matmul(W1);
    
    if (rank == 0) {
        std::cout << "\n=== After Column-Parallel MatMul ===" << std::endl;
        std::cout << "H is SHARDED: [" << B << ", " << local_F << "] per GPU" << std::endl;
    }
    
    DTensor W2(device_mesh, pg);
    
    // Full W2 tensor created on root GPU only
    std::vector<float> w2_full_data(F * C);
    if (rank == 0) {
        for (int i = 0; i < F * C; i++) w2_full_data[i] = 0.02f;
    }
    
    // replicated layout (root has full data, others have placeholder)
    Layout w2_replicated = Layout::replicated(device_mesh, {F, C});
    W2.setData(w2_full_data, w2_replicated);
    
    W2.shard(1, 0);  // shard on dim 1, root = 0
    
    DTensor Y = H.matmul(W2);
    
    Y.sync();

    if (rank == 0) {
        std::cout << "\n=== After Row-Parallel MatMul ===" << std::endl;
        std::cout << "Y shape: [" << B << ", " << T << ", " << C << "] (synched after AllReduce)" << std::endl;
        
        auto y_data = Y.getData();
        std::cout << "Y values (first 5): ";
        for (int i = 0; i < 5; i++) std::cout << y_data[i] << " ";
        std::cout << std::endl;
    }
    

    DTensor grad_Y(device_mesh, pg);
    std::vector<float> grad_data(B * T * C );

    for (int i = 0; i < B * T * C ; i++) {
        grad_data[i] = 0.1f * ((i % 7) + 1) * (rank + 1) + 0.05f * (i % 3);
    }
    
    Layout grad_layout = Layout::replicated(device_mesh, {B, T, C });
    grad_Y.setData(grad_data, grad_layout);
    
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
