

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

    cudaStream_t comm_stream;
    cudaStreamCreate(&comm_stream);


    cudaEvent_t start, stop;
    cudaEvent_t forward_start, forward_stop;
    cudaEvent_t backward_start, backward_stop;
    cudaEvent_t sync_start, sync_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&forward_start);
    cudaEventCreate(&forward_stop);
    cudaEventCreate(&backward_start);
    cudaEventCreate(&backward_stop);
    cudaEventCreate(&sync_start);
    cudaEventCreate(&sync_stop);

    float duration;
    float forward_duration, backward_duration, sync_duration;
    float avgduration;
    

    // const int64_t B = 8;      // batch size
    // const int64_t C = 768;      // input features
    // const int64_t T = 1024;      // token length
    // const int64_t F = 768*4;     // hidden dim (will be sharded: F / P per GPU)

    
    const int64_t B = 1;      // batch size
    const int64_t C = 2;      // input features
    const int64_t T = 3;      // token length
    const int64_t F = 2*2;     // hidden dim (will be sharded: F / P per GPU
    
    // ========== FIXED DATA FOR COMPARISON ==========
    // Initialize with deterministic values so we can compare with sharded version
    auto make_fixed_data = [](int64_t size, float base = 0.1f) {
        std::vector<float> data(size);
        for (int64_t i = 0; i < size; i++) {
            data[i] = base * (i + 1);  // Simple sequential values
        }
        return data;
    };
    
    // Pre-compute fixed data for all tensors
    std::vector<float> x_data = make_fixed_data(B * T * C, 0.1f);     // X: [1,3,2] = 6 elements
    std::vector<float> w1_data = make_fixed_data(B * C * F, 0.1f);    // W1: [1,2,4] = 8 elements
    std::vector<float> w2_data = make_fixed_data(B * F * C, 0.1f);    // W2: [1,4,2] = 8 elements
    std::vector<float> w3_data = make_fixed_data(B * C * F, 0.1f);    // W3: [1,2,4] = 8 elements
    std::vector<float> w4_data = make_fixed_data(B * F * C, 0.1f);    // W4: [1,4,2] = 8 elements
    std::vector<float> b1_data(B * T * F, 0.0f);   // Zero biases for simplicity
    std::vector<float> b2_data(B * T * C, 0.0f);
    std::vector<float> b3_data(B * T * F, 0.0f);
    std::vector<float> b4_data(B * T * C, 0.0f);
    // ================================================
    
    auto pg = device_mesh.get_process_group(0);
    
    
    for (int i = 0; i < 1; i++ ){

    cudaEventRecord(start,comm_stream);
    cudaEventRecord(forward_start, comm_stream);
      
    Layout x_layout( device_mesh,  { B, T, C });
    
    // Layout wqkv_layout(device_mesh, {B, C, 3 * C }, 2);

    if(rank == 0 ) std::cout << "\n x - dtensor : \n";

    DTensor X(device_mesh, pg, x_layout);
    if (rank == 0) X.setData(x_data);  // Set fixed data

    // DTensor WQKV(device_mesh, pg, wqkv_layout);

    // std::vector<float> wqkv_data(B * C * (C * 3));

    // DTensor WQKV_shard(device_mesh, {B, T, ( 3 * C) / 2 });

    // Tensors are already initialized with random data in constructor

    if (rank == 0)  X.display();

    X.replicate(0);  // root = 0


    // WQKV.shard(2, 0 ); // shard dim = 2, root = 0

    // Layout qkvlayout (device_mesh, {B, T , C / world_size }); // can add dim = 1 as another parameter if we are doing context parallelism

    // DTensor q(device_mesh, pg, qkvlayout);
    
    // DTensor k(device_mesh, pg, qkvlayout);
    
    // DTensor v(device_mesh, pg, qkvlayout); 

    // WQKV.qkvslpit(q,k,v);

    Layout w1_layout(device_mesh, {B ,C ,F });

    if(rank == 0 ) std::cout << "\n w1 - dtensor : \n";

    DTensor W1(device_mesh, pg, w1_layout);
    if (rank == 0) W1.setData(w1_data);  // Set fixed data
    W1.replicate(0);  // Broadcast to all ranks

    // int local_F = F / world_size;
        
    // Tensors are already initialized with random data in constructor

    if (rank == 0)  W1.display();

    // Layout W1_asS_layout(device_mesh,{B , C, F/2});

    // DTensor W1_Shard(device_mesh, pg, W1_asS_layout);
    
    // W1_Shard.shard_fused_transpose(2, 0, W1);  

    // if (rank == 0) {std::cout << "\n w1 shard : \n";} 

    // { W1_Shard.display(); }

    Layout H_layout(device_mesh, {B ,T ,F });
    
    DTensor H (device_mesh, pg, H_layout);
    
    DTensor B1 (device_mesh, pg, H_layout);
    if (rank == 0) B1.setData(b1_data);  // Set fixed data (zeros)
    B1.replicate(0);

    // Tensors are already initialized with random data in constructor
    
    // Enable gradient tracking on weight tensors for backward pass
    W1.mutable_tensor().set_requires_grad(true);
    B1.mutable_tensor().set_requires_grad(true);
    
    if (rank == 0) std::cout << "\n b1 - dtensor : \n";
    if (rank == 0)  B1.display();

    // Use autograd-enabled linear for gradient tracking
    // === FORWARD PASS TIMING START ===
    
    H.linear_w_autograd(X, W1, B1);

    if ( rank == 0 )std::cout << "\n h1 - dtensor : \n";
    
    if (rank == 0 ) H.display(); 

    Layout w2_layout(device_mesh, {B, F, C });

    DTensor W2(device_mesh, pg, w2_layout);
    if (rank == 0) W2.setData(w2_data);  // Set fixed data
    W2.replicate(0);

    // std::vector<float> w2_full_data(B * F * C );
    
    // Tensors are already initialized with random data in constructor

    if(rank == 0 ) std::cout << " \n w2 - dtensor : \n";

    if (rank == 0)  W2.display();

    // Layout W2_asS_layout(device_mesh,{B, (F/2), C});

    // DTensor W2_Shard(device_mesh, pg, W2_asS_layout);
  
    // W2_Shard.shard_fused_transpose(1, 0, W2);  // shard on dim 1, root = 0, parentTensor W2

    // if (rank == 0) {std::cout << "\n w2 shard : \n";} 

    // W2_Shard.display(); 

    DTensor Y1(device_mesh, pg, x_layout);

    DTensor B2(device_mesh, pg, x_layout);
    if (rank == 0) B2.setData(b2_data);  // Set fixed data (zeros)
    B2.replicate(0);
    
    // Tensors are already initialized with random data in constructor
    
    // Enable gradient tracking on weight tensors for backward pass
    W2.mutable_tensor().set_requires_grad(true);
    B2.mutable_tensor().set_requires_grad(true);
  
    if(rank == 0 ) std::cout << "\n b2 - dtensor : \n";
    if (rank == 0)  B2.display();
      
    // Use autograd-enabled linear for gradient tracking
    Y1.linear_w_autograd(H, W2 , B2);

    if ( rank == 0 ) std::cout << "\n y1 - dtensor : \n";
  
    // if ( rank == 0 )std::cout<<"\n Y1 before sync \n"; 

    if ( rank == 0 )Y1.display(); 

    // === FORWARD PASS TIMING END, SYNC TIMING START ===
    cudaEventRecord(sync_start, comm_stream);
    
    // Async sync - enqueue all-reduce but don't wait yet (PyTorch-style deferred wait)
    // Y1.sync_async();
    
    // === SYNC TIMING END ===
    // Note: sync_async() returns immediately, but we record the event here
    // The actual all-reduce is still running on GPU asynchronously
    
    // Now we need the result - call wait() before displaying
    // Y1.wait();  // Deferred wait - Y1 is now replicated and ready for next block
    cudaEventRecord(sync_stop, comm_stream);
    
    if (rank == 0) { std::cout<<"\n Y1 after sync (replicated - input to Block 2) \n"; }
    
    if (rank == 0) { Y1.display(); }
    
    // ============================================================
    // BLOCK 2: Second Tensor Parallel MLP Block
    // ============================================================
    // Y1 (replicated) -> H2 (sharded) -> Y2 (partial) -> sync -> Y2 (replicated)
    
    if (rank == 0) { std::cout<<"\n\n=== BLOCK 2: Second Tensor Parallel MLP ===\n"; }
    
    // --- Layer 3: Up Projection (Column Parallel) ---
    // H2 = Y1 @ W3_Shard + B3
    // Y1: [B, T, C] replicated, W3: [B, C, F] sharded on dim 2
    
    Layout w3_layout(device_mesh, {B, C, F});
    DTensor W3(device_mesh, pg, w3_layout);
    if (rank == 0) W3.setData(w3_data);  // Set fixed data
    W3.replicate(0);
    
    // Layout W3_asS_layout(device_mesh, {B, C, (F/2)});
    // DTensor W3_Shard(device_mesh, pg, W3_asS_layout);
    
    // W3_Shard.shard_fused_transpose(2, 0, W3);  // shard on dim 2
    
    if (rank == 0) { std::cout << "\n w3 - dtensor: \n"; }
    if (rank == 0) W3.display();
    
    Layout h2_layout(device_mesh, {B, T, F});
    DTensor H2(device_mesh, pg, h2_layout);
    DTensor B3(device_mesh, pg, h2_layout);
    if (rank == 0) B3.setData(b3_data);  // Set fixed data (zeros)
    B3.replicate(0);
    
    W3.mutable_tensor().set_requires_grad(true);
    B3.mutable_tensor().set_requires_grad(true);
    
    if (rank == 0) { std::cout << "\n b3 - dtensor : \n"; }
    if (rank == 0) { B3.display(); }
    
    // H2 = Y1 @ W3_Shard + B3 (Y1 is now Y1)
    H2.linear_w_autograd(Y1, W3 , B3);
    
    if ( rank == 0 )std::cout << "\n h2 - dtensor (Block 2 intermediate) : \n";
    if (rank == 0) H2.display();
    
    // --- Layer 4: Down Projection (Row Parallel) ---
    // Y2 = H2 @ W4_Shard + B4
    // H2: [B, T, F_local] sharded, W4: [B, F, C] sharded on dim 1
    
    Layout w4_layout(device_mesh, {B, F, C});
    DTensor W4(device_mesh, pg, w4_layout);
    if (rank == 0) W4.setData(w4_data);  // Set fixed data
    W4.replicate(0);
    
    // Layout W4_asS_layout(device_mesh, {B, (F/2), C});
    // DTensor W4_Shard(device_mesh, pg, W4_asS_layout);
    
    // W4_Shard.shard_fused_transpose(1, 0, W4);  // shard on dim 1
    
    if (rank == 0) { std::cout << "\n w4 dtensor : \n"; }
    if (rank == 0) W4.display();
    
    // Y2 output - same shape as Y1: [B, T, C]
    DTensor Y2(device_mesh, pg, x_layout);
    DTensor B4(device_mesh, pg, x_layout);
    if (rank == 0) B4.setData(b4_data);  // Set fixed data (zeros)
    B4.replicate(0);
    
    W4.mutable_tensor().set_requires_grad(true);
    B4.mutable_tensor().set_requires_grad(true);
    
    if (rank == 0) { std::cout << "\n b4 dtensor : \n"; }
    if (rank == 0) { B4.display(); }
    
    // Y2 = H2 @ W4_Shard + B4
    Y2.linear_w_autograd(H2, W4, B4);
    
    if ( rank == 0 ) std::cout << "\n Y2 dtensor \n";
    if (rank == 0) Y2.display();
    
    cudaEventRecord(forward_stop, comm_stream);
    // // All-reduce Y2 to get final replicated output
    // Y2.sync_async();
    // Y2.wait();
    
    // if (rank == 0) { std::cout << "\n Y2 after sync (final output) \n"; }
    // if (rank == 0) { Y2.display(); }
    
    // === BACKWARD TIMING START ===
    // === BACKWARD PASS ===
    // Compute gradients by calling backward on the FINAL output (Y2)
    if (rank == 0) { std::cout << "\n=== Computing Gradients (Backward Pass) ===\n"; }
    
    cudaEventRecord(backward_start, comm_stream);
    Y2.backward();  // Backward from Y2 through all 4 layers
    cudaEventRecord(backward_stop, comm_stream);
    
    // Display computed gradients
    if (rank == 0) {
        std::cout << "\n=== Gradients Computed ===\n";
        
        std::cout << "\n--- Block 1 Gradients ---\n";
        if (W1.local_tensor().owns_grad()) {
            std::cout << "\nW1 gradient:\n";
            W1.local_tensor().grad_view().display();
        } else {
            std::cout << "W1 has no gradient (not a leaf or requires_grad=false)\n";
        }
        
        if (W2.local_tensor().owns_grad()) {
            std::cout << "\nW2 gradient:\n";
            W2.local_tensor().grad_view().display();
        } else {
            std::cout << "W2 has no gradient (not a leaf or requires_grad=false)\n";
        }
        
        std::cout << "\n--- Block 2 Gradients ---\n";
        if (W3.local_tensor().owns_grad()) {
            std::cout << "\nW3 gradient:\n";
            W3.local_tensor().grad_view().display();
        } else {
            std::cout << "W3 has no gradient (not a leaf or requires_grad=false)\n";
        }
        
        if (W4.local_tensor().owns_grad()) {
            std::cout << "\nW4 gradient:\n";
            W4.local_tensor().grad_view().display();
        } else {
            std::cout << "W4 has no gradient (not a leaf or requires_grad=false)\n";
        }
    }

    }
    
    cudaEventRecord(stop,comm_stream);
    
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate all durations BEFORE destroying events
    cudaEventElapsedTime(&duration, start, stop);
    duration /=1;
    
    // Synchronize and get individual phase durations
    CUDA_CHECK(cudaEventSynchronize(forward_stop));
    CUDA_CHECK(cudaEventSynchronize(backward_stop));
    cudaEventElapsedTime(&forward_duration, forward_start, forward_stop);
    cudaEventElapsedTime(&sync_duration, sync_start, sync_stop);
    cudaEventElapsedTime(&backward_duration, backward_start, backward_stop);
    
    // Now destroy all events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(forward_start);
    cudaEventDestroy(forward_stop);
    cudaEventDestroy(backward_start);
    cudaEventDestroy(backward_stop);
    cudaEventDestroy(sync_start);
    cudaEventDestroy(sync_stop);

    if(rank == 0){
      std::cout<<"\n\n ========= BENCHMARKS ========= \n\n"<<std::endl;

      std::cout<<" FORWARD DURATION  : "<< forward_duration <<" ms"<<std::endl;
      std::cout<<" SYNC DURATION     : "<< sync_duration <<" ms"<<std::endl;
      std::cout<<" BACKWARD DURATION : "<< backward_duration <<" ms"<<std::endl;
      std::cout<<" TOTAL DURATION    : "<< duration <<" ms"<<std::endl;
      std::cout<<std::endl;
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


