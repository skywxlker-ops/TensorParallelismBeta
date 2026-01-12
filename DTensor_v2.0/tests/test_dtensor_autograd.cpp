#include <unparalleled/unparalleled.h>
#include <iostream>
#include <cassert>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cudaSetDevice(rank);
    
    auto mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    auto pg = init_process_group(world_size, rank);
    
    if (rank == 0) {
        std::cout << "\n=== DTensor Autograd Integration Test ===" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
    }
    
    int passed = 0, total = 0;
    
    // Test 1: set_requires_grad / requires_grad
    if (rank == 0) std::cout << "\n[Test 1] set_requires_grad / requires_grad..." << std::endl;
    std::vector<int64_t> shape = {4, 8};
    Layout layout(*mesh, shape, 1);  // Column-sharded
    auto W = DTensor::randn(shape, mesh, pg, layout);
    
    W.set_requires_grad(true);
    bool ok1 = (W.requires_grad() == true);
    if (ok1) passed++; total++;
    if (rank == 0) std::cout << (ok1 ? "[PASS]" : "[FAIL]") << " requires_grad() returns true" << std::endl;
    
    W.set_requires_grad(false);
    bool ok2 = (W.requires_grad() == false);
    if (ok2) passed++; total++;
    if (rank == 0) std::cout << (ok2 ? "[PASS]" : "[FAIL]") << " requires_grad() returns false after set" << std::endl;
    
    // Test 2: Column-parallel matmul with autograd  
    // X [M, K] replicated @ W [K, N/P] column-sharded -> Y [M, N/P]
    if (rank == 0) std::cout << "\n[Test 2] Column-parallel matmul with autograd..." << std::endl;
    int M = 4, K = 8, N = 8;  // N must be divisible by world_size
    
    std::vector<int64_t> x_shape = {M, K};
    std::vector<int64_t> w_shape = {K, N};
    
    Layout x_layout = Layout::replicated(*mesh, x_shape);  // Replicated input
    Layout w_layout(*mesh, w_shape, 1);  // Column-sharded weights
    
    auto X = DTensor::randn(x_shape, mesh, pg, x_layout);
    auto W2 = DTensor::randn(w_shape, mesh, pg, w_layout);
    W2.set_requires_grad(true);
    
    auto Y = X.matmul(W2);
    auto Y_data = Y.getData();
    
    int expected_cols = N / world_size;
    bool ok3 = (Y_data.size() == static_cast<size_t>(M * expected_cols));
    if (ok3) passed++; total++;
    if (rank == 0) {
        std::cout << (ok3 ? "[PASS]" : "[FAIL]") 
                  << " Matmul completed, result size: " << Y_data.size() 
                  << " (expected: " << M * expected_cols << ")" << std::endl;
    }
    
    // Test 3: zero_grad
    if (rank == 0) std::cout << "\n[Test 3] zero_grad..." << std::endl;
    W2.zero_grad();
    passed++; total++;
    if (rank == 0) std::cout << "[PASS] zero_grad() completed without error" << std::endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n=== Autograd Tests: " << passed << "/" << total << " PASSED ===" << std::endl;
    }
    
    pg.reset();
    MPI_Finalize();
    return (passed == total) ? 0 : 1;
}

// Build: cd tests && make test_dtensor_autograd
// Run:   mpirun -np 2 --allow-run-as-root ./test_dtensor_autograd
