#include <iostream>
#include <vector>
#include <cmath>
#include "tensor/dtensor.h"
#include "process_group/ProcessGroupNCCL.h"
#include "tensor/device_mesh.h"
#include "tensor/layout.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    cudaSetDevice(rank);
    auto pg = init_process_group(world_size, rank);
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});

    // 2D tensor: [[2, 4], [6, 8]]
    std::vector<int> shape = {2, 2};
    std::vector<float> original = {2, 4, 6, 8};

    // Create replicated tensor
    Layout rep_layout(mesh, shape, ShardingType::REPLICATED);
    DTensor rep(mesh, pg);
    rep.setData(original, rep_layout);

    // Redistribute to Partial
    Layout partial_layout(mesh, shape, ShardingType::PARTIAL, -1, "sum");
    DTensor partial = rep.redistribute(partial_layout);
    std::vector<float> p_data = partial.getData();

    // Reduce back
    DTensor recovered = partial.redistribute(rep_layout);
    std::vector<float> r_data = recovered.getData();

    // Only rank 0 prints
    if (rank == 0) {
        std::cout << "Original:  [[" << original[0] << ", " << original[1] << "], [" 
                  << original[2] << ", " << original[3] << "]]" << std::endl;
        std::cout << "Partial:   [[" << p_data[0] << ", " << p_data[1] << "], [" 
                  << p_data[2] << ", " << p_data[3] << "]] (each GPU)" << std::endl;
        std::cout << "Recovered: [[" << r_data[0] << ", " << r_data[1] << "], [" 
                  << r_data[2] << ", " << r_data[3] << "]]" << std::endl;
        
        bool pass = (r_data[0] == 2 && r_data[1] == 4 && r_data[2] == 6 && r_data[3] == 8);
        std::cout << (pass ? "PASSED" : "FAILED") << std::endl;
    }

    MPI_Finalize();
    return 0;
}
