#include <mpi.h>
#include <nccl.h>
#include <iostream>
#include "process_group/process_group.h"
#include "tensor/dtensor.h"

void worker(int rank, int world_size, const ncclUniqueId &id) {
    int device = rank;
    ProcessGroup pg(rank, world_size, device, id);
    DTensor tensor(world_size, 8, rank, &pg);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "\n==== BEFORE ALLREDUCE ====\n";
    MPI_Barrier(MPI_COMM_WORLD);

    tensor.copyDeviceToHost();
    std::cout << tensor.toString() << std::endl;

    tensor.allReduce();
    tensor.copyDeviceToHost();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "\n==== AFTER ALLREDUCE ====\n";
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << tensor.toString() << std::endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    worker(rank, world_size, id);

    MPI_Finalize();
    return 0;
}
