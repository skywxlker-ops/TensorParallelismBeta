for (int mesh_dim = 0; mesh_dim < ndim(); ++mesh_dim) {
    // Calculate color: hash of coordinates excluding mesh_dim
    int color = 0;
    int multiplier = 1;
    for (int d = ndim() - 1; d >= 0; --d) {
        if (d != mesh_dim) {
            color += my_coordinate_[d] * multiplier;
            multiplier *= mesh_shape_[d];
        }
    }
    
    // Key determines ordering within the sub-communicator
    int key = my_coordinate_[mesh_dim];
}

MPI_Comm sub_comm;
MPI_Comm_split(MPI_COMM_WORLD, color, key, &sub_comm);

ncclUniqueId create_nccl_id(int root_rank, MPI_Comm comm) {
    ncclUniqueId nccl_id;
    
    int my_rank_in_comm;
    MPI_Comm_rank(comm, &my_rank_in_comm);
    
    if (my_rank_in_comm == root_rank) {
        ncclGetUniqueId(&nccl_id);  // Root creates NCCL ID
    }
    
    // Broadcast ONLY within this sub-communicator
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 
              root_rank, comm);  // Use sub-comm, not COMM_WORLD!
    
    return nccl_id;
}


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

        const int64_t B = 2;      // batch size
        const int64_t C = 4;      // input features
        const int64_t T = 4;      // token length
        const int64_t F = 4*4;     // hidden dim (will be sharded: F / P per GPU

        Layout x_layout( device_mesh,  { B, T, C });

        if(rank == 0 ) std::cout << "\n x - dtensor: \n";
        DTensor X(device_mesh, pg, x_layout);
        std::vector<float> x_data(B * T * C );

        X.rand();

        if (rank == 0)  X.display();

        X.replicate(0);  // root = 0

        Layout w1_layout(device_mesh, {B ,C ,F }, 2);

        if(rank == 0 ) std::cout << "\n w1 - dtensor: \n";

        DTensor W1(device_mesh, pg, w1_layout);

        int local_F = F / world_size;
        
        // Full W1 tensor created on root GPU only
        std::vector<float> w1_full_data(B * C * F );
    
        W1.rand();

        if (rank == 0)  W1.display();

        Layout W1_asS_layout(device_mesh,{B, C, F/2});

        if (rank == 0) {std::cout << "\n w1 shard - dtensor: \n";}

        DTensor W1_Shard(device_mesh, pg, W1_asS_layout);
        
        W1_Shard.shard(2, 0, W1);  
        
        if (rank == 0) { W1_Shard.display(); }

        Layout H_layout(device_mesh, {B ,T ,F/2 });

        if(rank == 0 ) std::cout << "\n h - dtensor: \n";

        DTensor H (device_mesh, pg, H_layout);

        H.matmul(X,W1_Shard);

        if (rank == 0) { H.display(); }

        Layout w2_layout(device_mesh, {B, F, C }, 1);

        if(rank == 0 ) std::cout << " \n w2 - dtensor: \n";

        DTensor W2(device_mesh, pg, w2_layout);

        std::vector<float> w2_full_data(B * F * C );

        W2.rand();

        if (rank == 0) { W2.display(); }

        if(rank == 0) std::cout << "\n w2-shard - dtensor: \n";
        Layout W2_asS_layout(device_mesh,{B, (F/2), C});

        DTensor W2_Shard(device_mesh, pg, W2_asS_layout);
        
        W2_Shard.shard(1, 0, W2);  // shard on dim 1, root = 0, parentTensor W2
        
        if (rank == 0) { W2_Shard.display(); }

        if(rank == 0 ) std::cout << "\n y - dtensor: \n";
        
        DTensor Y(device_mesh, pg, x_layout);
        
        Y.matmul(H,W2_Shard);

        if (rank == 0) {  std::cout<<"\n Y before sync \n"; Y.display(); }
    
        Y.sync();

        if (rank == 0) { std::cout<<"\n Y after sync \n"; }

        if (rank == 0) {
            Y.display();
        }
        
        Layout grad_layout (device_mesh, {B, T, C });
        DTensor grad_Y(device_mesh, pg, grad_layout);

        grad_Y.rand();
        
        if (rank == 0) {
            std::cout << "\n=== Before sync() ===" << std::endl;
            grad_Y.display();
        }
        
        grad_Y.sync(); // sum gradients
        
        if (rank == 0) {
            std::cout << "\n=== After sync() - Gradients added ===" << std::endl;
            grad_Y.display();
        }
        MPI_Finalize();
        return 0;
    }