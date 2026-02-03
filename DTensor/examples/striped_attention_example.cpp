/**
 * Example: Striped Attention Token Permutation
 * 
 * Demonstrates permute_striped() and unpermute_striped() for
 * distributing tokens across GPUs in a balanced way for causal masking.
 * 
 * Build: make striped_attention_example
 * Run:   mpirun -np 4 ./examples/striped_attention_example
 */

#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>
#include <iomanip>

void print_sequence(const std::vector<float>& data, const std::string& label) {
    std::cout << label << ": [";
    for (size_t i = 0; i < data.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::setw(2) << (int)data[i];
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Setup
    auto device_mesh = std::make_shared<DeviceMesh>(std::vector<int>{world_size});
    
    ncclUniqueId nccl_id;
    if (rank == 0) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    auto pg = std::make_shared<ProcessGroup>(rank, world_size, rank, nccl_id);
    
    // Create a sequence of tokens: 0, 1, 2, ..., 15
    // Simulates a sequence dimension of length 16 on 4 GPUs
    const int seq_len = 16;
    const int hidden_dim = 4;  // Additional dimension to show multi-dim support
    
    if (rank == 0) {
        std::cout << "\n=== Striped Attention Permutation Demo ===" << std::endl;
        std::cout << "Sequence length: " << seq_len << std::endl;
        std::cout << "Hidden dimension: " << hidden_dim << std::endl;
        std::cout << "World size (d): " << world_size << std::endl;
        std::cout << "Chunk size (n): " << seq_len / world_size << std::endl;
        std::cout << "\n";
    }
    
    DTensor tokens(device_mesh, pg);
    std::vector<float> token_data(seq_len);
    for (int i = 0; i < seq_len; i++) token_data[i] = (float)i;
    
    Layout layout = Layout::replicated(device_mesh, {seq_len});
    tokens.setData(token_data, layout);
    
    if (rank == 0) {
        std::cout << "=== Test 1: 1D Sequence ===" << std::endl;
        print_sequence(token_data, "Original ");
    }
    
    // Apply striped permutation
    tokens.permute_striped(0);  // Permute along dim 0 (sequence dim)
    
    if (rank == 0) {
        auto permuted = tokens.getData();
        print_sequence(permuted, "Permuted ");
        
        // Show which tokens go to which GPU
        int n = seq_len / world_size;
        std::cout << "\nDistribution to GPUs:" << std::endl;
        for (int gpu = 0; gpu < world_size; gpu++) {
            std::cout << "  GPU " << gpu << ": [";
            for (int i = 0; i < n; i++) {
                if (i > 0) std::cout << ", ";
                std::cout << std::setw(2) << (int)permuted[gpu * n + i];
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Apply unpermutation to restore original order
    tokens.unpermute_striped(0);
    
    if (rank == 0) {
        auto restored = tokens.getData();
        print_sequence(restored, "Restored ");
        
        // Verify restoration
        bool match = true;
        for (int i = 0; i < seq_len; i++) {
            if ((int)restored[i] != i) {
                match = false;
                break;
            }
        }
        std::cout << "Verification: " << (match ? "PASSED " : "FAILED ") << std::endl;
    }
    
    // ============================================================
    // Test 2: 2D Tensor (Seq x Hidden)
    // ============================================================
    if (rank == 0) {
        std::cout << "\n=== Test 2: 2D Tensor [Seq, Hidden] ===" << std::endl;
    }
    
    DTensor tokens_2d(device_mesh, pg);
    std::vector<float> data_2d(seq_len * hidden_dim);
    
    // Fill with pattern: token_id * 10 + hidden_idx
    // So token 5, hidden 2 = 52
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < hidden_dim; h++) {
            data_2d[s * hidden_dim + h] = s * 10.0f + h;
        }
    }
    
    Layout layout_2d = Layout::replicated(device_mesh, {seq_len, hidden_dim});
    tokens_2d.setData(data_2d, layout_2d);
    
    if (rank == 0) {
        std::cout << "Original (first row = token 0, second = token 1, etc.):" << std::endl;
        for (int s = 0; s < 4; s++) {  // Show first 4 tokens
            std::cout << "  Token " << s << ": [";
            for (int h = 0; h < hidden_dim; h++) {
                if (h > 0) std::cout << ", ";
                std::cout << std::setw(2) << (int)data_2d[s * hidden_dim + h];
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "  ..." << std::endl;
    }
    
    // Permute along sequence dimension (dim 0)
    tokens_2d.permute_striped(0);
    
    if (rank == 0) {
        auto permuted_2d = tokens_2d.getData();
        std::cout << "\nAfter permute_striped(0) - first 4 rows:" << std::endl;
        for (int s = 0; s < 4; s++) {
            std::cout << "  Row " << s << ": [";
            for (int h = 0; h < hidden_dim; h++) {
                if (h > 0) std::cout << ", ";
                std::cout << std::setw(3) << (int)permuted_2d[s * hidden_dim + h];
            }
            std::cout << "]  <- Token " << (int)(permuted_2d[s * hidden_dim] / 10) << std::endl;
        }
        std::cout << "  ..." << std::endl;
    }
    
    // Restore
    tokens_2d.unpermute_striped(0);
    
    if (rank == 0) {
        auto restored_2d = tokens_2d.getData();
        bool match = true;
        for (int i = 0; i < seq_len * hidden_dim; i++) {
            if ((int)restored_2d[i] != (int)data_2d[i]) {
                match = false;
                break;
            }
        }
        std::cout << "\n2D Verification: " << (match ? "PASSED " : "FAILED ") << std::endl;
    }
    
    if (rank == 0) {
        std::cout << "\n=== All Tests Complete ===" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
