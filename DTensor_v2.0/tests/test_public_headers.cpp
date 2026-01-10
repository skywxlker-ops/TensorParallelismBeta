/**
 * Test: Public API Test Suite
 */

#include <unparalleled/unparalleled.h>
#include <iostream>
#include <cmath>

bool float_eq(float a, float b, float eps = 1e-5) {
    return std::fabs(a - b) < eps;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cudaSetDevice(rank);
    
    int passed = 0, total = 0;
    
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    auto pg = init_process_group(world_size, rank);
    std::vector<int64_t> shape = {4, 8};
    Layout layout(*mesh, shape, 0);
    
    // Factory: zeros
    {
        auto t = DTensor::zeros(shape, mesh, pg, layout);
        auto d = t.getData();
        bool ok = !d.empty() && float_eq(d[0], 0.0f);
        if (ok) passed++; total++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Factory: ones
    {
        auto t = DTensor::ones(shape, mesh, pg, layout);
        auto d = t.getData();
        bool ok = !d.empty() && float_eq(d[0], 1.0f);
        if (ok) passed++; total++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Factory: full
    {
        auto t = DTensor::full(shape, 3.14f, mesh, pg, layout);
        auto d = t.getData();
        bool ok = !d.empty() && float_eq(d[0], 3.14f);
        if (ok) passed++; total++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Arithmetic: add
    {
        auto a = DTensor::full(shape, 2.0f, mesh, pg, layout);
        auto b = DTensor::full(shape, 3.0f, mesh, pg, layout);
        auto c = a.add(b);
        auto d = c.getData();
        bool ok = !d.empty() && float_eq(d[0], 5.0f);
        if (ok) passed++; total++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Arithmetic: mul
    {
        auto a = DTensor::full(shape, 2.0f, mesh, pg, layout);
        auto b = DTensor::full(shape, 3.0f, mesh, pg, layout);
        auto c = a.mul(b);
        auto d = c.getData();
        bool ok = !d.empty() && float_eq(d[0], 6.0f);
        if (ok) passed++; total++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "PublicAPI: " << passed << "/" << total << " tests passed" << std::endl;
    }
    
    MPI_Finalize();
    return (passed == total) ? 0 : 1;
}

// Build: cd tests && make test_public_headers
// Run:   mpirun -np 2 ./test_public_headers
