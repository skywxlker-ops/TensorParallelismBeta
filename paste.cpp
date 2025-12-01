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