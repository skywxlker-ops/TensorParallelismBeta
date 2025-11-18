#pragma once
#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>

inline void barrierHeader(int rank, const std::string &header) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "\n" << std::string(60, '=') << "\n" << header << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

inline void gatherPrint(int rank, int world_size, const std::string &data) {
    std::vector<int> sizes(world_size);
    int local_size = data.size();
    MPI_Gather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<std::string> all(world_size);
        all[0] = data;
        for (int r = 1; r < world_size; r++) {
            all[r].resize(sizes[r]);
            MPI_Recv(all[r].data(), sizes[r], MPI_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int r = 0; r < world_size; r++)
            std::cout << all[r] << std::endl;
    } else {
        MPI_Send(data.c_str(), local_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}